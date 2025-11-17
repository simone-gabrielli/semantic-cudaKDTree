#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <numeric>

#include "cukd/builder.h"
#include "cukd/spatial-kdtree.h"
#include "cukd/label_mask.h"
#include "cukd/traversal_label_pruned.h"

#define CUDA_CHECK(x)                                                                       \
  do {                                                                                      \
    cudaError_t e = (x);                                                                    \
    if (e != cudaSuccess) {                                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                                              \
    }                                                                                       \
  } while (0)

// ---------------- data & traits ----------------
struct LPoint3f {
  float3 p;
  int label;
};

struct LPoint3f_traits : public cukd::default_data_traits<float3> {
  using point_t = float3;
  static constexpr int num_dims = 3;
  enum { has_explicit_dim = false };
  __host__ __device__ static inline const float3& get_point(const LPoint3f& d) { return d.p; }
  __host__ __device__ static inline float3& get_point(LPoint3f& d) { return d.p; }
  __host__ __device__ static inline float get_coord(const LPoint3f& d, int dim) { return cukd::get_coord(d.p, dim); }
};

namespace cukd {
namespace labels {
template <>
struct default_label_traits<LPoint3f> {
  __host__ __device__ static inline int get_label(const LPoint3f& d) { return d.label; }
};
}  // namespace labels
}  // namespace cukd

// ---------------- kernels ----------------
using cukd::labels::Mask;

__global__ void kernel_fcp_label_same(const float3* queries, const int* q_labels, int M, const LPoint3f* nodes, int N, const Mask* masks, int* out_idx, float* out_d2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  Mask desired;
  desired.clear();
  if (q_labels[i] >= 0) desired.setBit(q_labels[i]);  // single-label
  float best;
  int idx = cukd::labels::fcp_label_filtered<LPoint3f, LPoint3f_traits, float3>(queries[i], nodes, N, masks, desired, best);
  out_idx[i] = idx;
  out_d2[i] = best;
}

// Unfiltered baseline (all labels allowed)
__device__ inline Mask make_all_mask() {
  Mask m;
#if CUKD_LABEL_MASK_WORDS == 1
  m.w[0] = ~(cukd_label_word_t)0;
#else
#pragma unroll
  for (int w = 0; w < CUKD_LABEL_MASK_WORDS; ++w) m.w[w] = ~(cukd_label_word_t)0;
#endif
  return m;
}
__global__ void kernel_fcp_unfiltered(const float3* queries, int M, const LPoint3f* nodes, int N, const Mask* masks, int* out_idx, float* out_d2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  float best;
  int idx = cukd::labels::fcp_label_filtered<LPoint3f, LPoint3f_traits, float3>(queries[i], nodes, N, masks, make_all_mask(), best);
  out_idx[i] = idx;
  out_d2[i] = best;
}

// -------- Batched by label: single 2D launch, constant label per row --------
__global__ void kernel_fcp_label_batched2D_constL(
  const float3* queries_sorted,
  const int* offsets,
  const int* counts,
  int numLabels,
  const LPoint3f* nodes,
  int N,
  const Mask* masks,
  int* out_idx,
  float* out_d2) {
  const int L = blockIdx.y;
  if (L >= numLabels) return;
  const int begin = offsets[L];
  const int count = counts[L];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  const int qIdx = begin + i;
  float best;
  int idx;

#if CUKD_LABEL_MASK_WORDS == 1
  // Direct single-word fast path (no per-thread label loads)
  idx = cukd::labels::fcp_label_single_word<LPoint3f, LPoint3f_traits, float3>(queries_sorted[qIdx], nodes, N, masks, L, best);
#else
  // Fallback: build 1-bit mask and use generic wrapper
  Mask desired;
  desired.clear();
  desired.setBit(L);
  idx = cukd::labels::fcp_label_filtered<LPoint3f, LPoint3f_traits, float3>(queries_sorted[qIdx], nodes, N, masks, desired, best);
#endif

  out_idx[qIdx] = idx;
  out_d2[qIdx] = best;
}

// ---------------- timing helpers ----------------
float time_kernel(std::function<void(cudaEvent_t, cudaEvent_t)> launch, int repeats = 5) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  // warmup
  for (int i = 0; i < 2; i++) {
    launch(start, stop);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  float total_ms = 0.f;
  for (int i = 0; i < repeats; i++) {
    CUDA_CHECK(cudaEventRecord(start));
    launch(start, stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms / repeats;
}

// ---------------- main ----------------
int main(int argc, char** argv) {
  // N points, M queries, labels, repeats
  int N = (argc > 1 ? std::max(1 << 18, atoi(argv[1])) : 1 << 20);
  int M = (argc > 2 ? std::max(1 << 16, atoi(argv[2])) : 1 << 20);
  int NUM_LABELS = (argc > 3 ? std::max(2, atoi(argv[3])) : 16);
  int REPEATS = (argc > 4 ? std::max(1, atoi(argv[4])) : 5);
  printf("Benchmark(SpatialKDTree dmem): N=%d, M=%d, labels=%d, repeats=%d\n", N, M, NUM_LABELS, REPEATS);

  // host data
  std::mt19937 rng(0xBADC0DE);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  std::uniform_int_distribution<int> lab(0, NUM_LABELS - 1);

  std::vector<LPoint3f> h_pts(N);
  for (int i = 0; i < N; i++) {
    h_pts[i].p = make_float3(uni(rng), uni(rng), uni(rng));
    h_pts[i].label = lab(rng);
  }

  // device points (explicit device memory)
  LPoint3f* d_points = nullptr;
  CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(LPoint3f)));
  CUDA_CHECK(cudaMemcpy(d_points, h_pts.data(), N * sizeof(LPoint3f), cudaMemcpyHostToDevice));

  // build SpatialKDTree in-place, then wrap
  cukd::SpatialKDTree<LPoint3f, LPoint3f_traits> tree;
  cukd::buildTree<LPoint3f, LPoint3f_traits>(d_points, N);
  tree.data = d_points;
  tree.numPrims = N;

  // access
  const LPoint3f* nodes = tree.data;
  const int nPts = tree.numPrims;

  // build subtree label masks
  cukd::labels::Mask* d_masks = nullptr;
  CUDA_CHECK(cudaMalloc(&d_masks, nPts * sizeof(cukd::labels::Mask)));
  cukd::labels::build_label_masks<LPoint3f>(const_cast<LPoint3f*>(nodes), nPts, d_masks);

  // queries
  std::vector<float3> h_q(M);
  std::vector<int> h_L(M);
  for (int i = 0; i < M; i++) {
    h_q[i] = make_float3(uni(rng), uni(rng), uni(rng));
    h_L[i] = lab(rng);
  }

  float3* d_q = nullptr;
  int* d_L = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, M * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_L, M * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), M * sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_L, h_L.data(), M * sizeof(int), cudaMemcpyHostToDevice));

  int* d_idx = nullptr;
  float* d_d2 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_idx, M * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_d2, M * sizeof(float)));

  // Prefer L1 cache for memory-bound kernels
  cudaFuncSetCacheConfig(kernel_fcp_unfiltered, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(kernel_fcp_label_same, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(kernel_fcp_label_batched2D_constL, cudaFuncCachePreferL1);

  dim3 BS(256), GS((M + BS.x - 1) / BS.x);

  // ---------------- timings ----------------
  auto launch_unfiltered = [&](cudaEvent_t, cudaEvent_t stop) {
    kernel_fcp_unfiltered<<<GS, BS>>>(d_q, M, nodes, nPts, d_masks, d_idx, d_d2);
    CUDA_CHECK(cudaEventRecord(stop));
  };
  auto launch_label = [&](cudaEvent_t, cudaEvent_t stop) {
    kernel_fcp_label_same<<<GS, BS>>>(d_q, d_L, M, nodes, nPts, d_masks, d_idx, d_d2);
    CUDA_CHECK(cudaEventRecord(stop));
  };

  float t_unf_ms = time_kernel(launch_unfiltered, REPEATS);
  float t_lab_ms = time_kernel(launch_label, REPEATS);

  // --------- batched by label (single 2D launch) ---------
  auto compute_dense_labels = [&](int level_check, float dense_thresh) -> std::vector<uint8_t> {
    // nodes at this level [start..end]
    auto level_start = [](int lvl) { return (1 << lvl) - 1; };
    auto level_end = [&](int lvl, int N) {
      int e = (1 << (lvl + 1)) - 2;
      return e < (N - 1) ? e : (N - 1);
    };

    int s = level_start(level_check);
    int e = level_end(level_check, nPts);
    s = std::max(0, s);
    e = std::max(s, e);
    int count = e - s + 1;

    // Pull that small slice of masks to host
    std::vector<Mask> slice(count);
    cudaMemcpy(slice.data(), d_masks + s, count * sizeof(Mask), cudaMemcpyDeviceToHost);

    std::vector<uint8_t> dense(NUM_LABELS, 0);
    for (int L = 0; L < NUM_LABELS; ++L) {
      int hits = 0;
      for (int i = 0; i < count; i++) {
#if CUKD_LABEL_MASK_WORDS == 1
        const auto word = slice[i].w[0];
        const auto bit = (cukd_label_word_t(1) << (L % CUKD_LABEL_BITS_PER_WORD));
        hits += (word & bit) ? 1 : 0;
#else
        const int wi = L / CUKD_LABEL_BITS_PER_WORD;
        const int bi = L % CUKD_LABEL_BITS_PER_WORD;
        const cukd_label_word_t bit = (cukd_label_word_t(1) << bi);
        hits += (slice[i].w[wi] & bit) ? 1 : 0;
#endif
      }
      float frac = (count > 0) ? float(hits) / float(count) : 1.f;
      dense[L] = (frac >= dense_thresh) ? 1 : 0;
    }
    return dense;
  };

  // --- build per-label offsets for sorted arrays (we'll reuse twice) ---
  std::vector<int> counts(NUM_LABELS, 0);
  for (int L : h_L) ++counts[L];
  std::vector<int> offsets(NUM_LABELS, 0);
  for (int L = 1; L < NUM_LABELS; ++L) offsets[L] = offsets[L - 1] + counts[L - 1];

  std::vector<float3> h_q_sorted(M);
  std::vector<int> label_of_pos(M);
  {
    auto tmp = offsets;
    for (int i = 0; i < M; ++i) {
      int L = h_L[i];
      int pos = tmp[L]++;
      h_q_sorted[pos] = h_q[i];
      label_of_pos[pos] = L;
    }
  }
  offsets[0] = 0;
  for (int L = 1; L < NUM_LABELS; ++L) offsets[L] = offsets[L - 1] + counts[L - 1];

  // device buffers for sorted queries
  float3* d_q_sorted = nullptr;
  int* d_idx_sorted = nullptr;
  float* d_d2_sorted = nullptr;
  int *d_counts = nullptr, *d_offsets = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q_sorted, M * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_idx_sorted, M * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_d2_sorted, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_counts, NUM_LABELS * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_offsets, NUM_LABELS * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_q_sorted, h_q_sorted.data(), M * sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_counts, counts.data(), NUM_LABELS * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), NUM_LABELS * sizeof(int), cudaMemcpyHostToDevice));

  // -------- 2D batched (same as before) ----------
  int maxCount = 0;
  for (int c : counts) maxCount = std::max(maxCount, c);
  dim3 BS2(256), GS2((maxCount + BS2.x - 1) / BS2.x, NUM_LABELS);
  auto launch_batched2D = [&](cudaEvent_t, cudaEvent_t stop) {
    kernel_fcp_label_batched2D_constL<<<GS2, BS2>>>(d_q_sorted, d_offsets, d_counts, NUM_LABELS, nodes, nPts, d_masks, d_idx_sorted, d_d2_sorted);
    CUDA_CHECK(cudaEventRecord(stop));
  };
  float t_batched_ms = time_kernel(launch_batched2D, REPEATS);

  // -------- HYBRID: dense -> unfiltered, rare -> pruned (batched2D) ----------
  const int LEVEL_CHECK = 10;         // try 7..10
  const float DENSE_THRESH = 0.70f;  // label present in >=70% of nodes at that level -> dense
  auto dense = compute_dense_labels(LEVEL_CHECK, DENSE_THRESH);

  // Build two index ranges for rare/dense within the sorted layout
  // We simply run *two* kernels over disjoint stripes via masks in counts.
  std::vector<int> counts_rare(NUM_LABELS), counts_dense(NUM_LABELS);
  for (int L = 0; L < NUM_LABELS; ++L) {
    counts_rare[L] = dense[L] ? 0 : counts[L];
    counts_dense[L] = dense[L] ? counts[L] : 0;
  }
  int *d_counts_rare = nullptr, *d_counts_dense = nullptr;
  CUDA_CHECK(cudaMalloc(&d_counts_rare, NUM_LABELS * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_counts_dense, NUM_LABELS * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_counts_rare, counts_rare.data(), NUM_LABELS * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_counts_dense, counts_dense.data(), NUM_LABELS * sizeof(int), cudaMemcpyHostToDevice));

  // A tiny kernel that treats a (offsets, counts) stripe as a flat array for unfiltered:
  auto launch_unfiltered_stripes = [&](const int* d_countsX, cudaEvent_t, cudaEvent_t stop) {
    // One pass: launch over each label's stripe using the baseline kernel
    for (int L = 0; L < NUM_LABELS; ++L) {
      if (counts_dense[L] == 0) continue;
      int begin = offsets[L], cnt = counts_dense[L];
      dim3 bs(256), gs((cnt + bs.x - 1) / bs.x);
      kernel_fcp_unfiltered<<<gs, bs>>>(d_q_sorted + begin, cnt, nodes, nPts, d_masks, d_idx_sorted + begin, d_d2_sorted + begin);
    }
    CUDA_CHECK(cudaEventRecord(stop));
  };

  // Rare labels -> pruned (2D batched) with zeroed counts for dense:
  auto launch_pruned_rare = [&](cudaEvent_t, cudaEvent_t stop) {
    kernel_fcp_label_batched2D_constL<<<GS2, BS2>>>(d_q_sorted, d_offsets, d_counts_rare, NUM_LABELS, nodes, nPts, d_masks, d_idx_sorted, d_d2_sorted);
    CUDA_CHECK(cudaEventRecord(stop));
  };

  auto time_combined = [&](int repeats) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // warmup
    for (int i = 0; i < 2; i++) {
      launch_unfiltered_stripes(d_counts_dense, start, stop);
      launch_pruned_rare(start, stop);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    float total_ms = 0.f;
    for (int r = 0; r < repeats; r++) {
      CUDA_CHECK(cudaEventRecord(start));
      launch_unfiltered_stripes(d_counts_dense, start, stop);
      launch_pruned_rare(start, stop);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      total_ms += ms;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / repeats;
  };

  float t_hybrid_ms = time_combined(REPEATS);

  // ----- print -----
  auto to_qps = [&](float ms) { return (double)M / (ms / 1000.0); };
  printf("\n=== Results ===\n");
  printf("Unfiltered:          %.3f ms  |  %.1f Mq/s\n", t_unf_ms, to_qps(t_unf_ms) / 1e6);
  printf("Filtered (as-is):    %.3f ms  |  %.1f Mq/s\n", t_lab_ms, to_qps(t_lab_ms) / 1e6);
  printf("Filtered (batched):  %.3f ms  |  %.1f Mq/s\n", t_batched_ms, to_qps(t_batched_ms) / 1e6);
  printf("Filtered (hybrid):   %.3f ms  |  %.1f Mq/s  [level=%d, dense>=%.0f%%]\n", t_hybrid_ms, to_qps(t_hybrid_ms) / 1e6, LEVEL_CHECK, DENSE_THRESH * 100.0f);

  // cleanup
  cudaFree(d_masks);
  cudaFree(d_q);
  cudaFree(d_L);
  cudaFree(d_idx);
  cudaFree(d_d2);
  cudaFree(d_q_sorted);
  cudaFree(d_idx_sorted);
  cudaFree(d_d2_sorted);
  cudaFree(d_counts);
  cudaFree(d_offsets);
  cudaFree(d_points);
  return 0;
}
