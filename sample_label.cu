#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cassert>

#include "cukd/builder.h"                 // from the cudaKDTree repo
#include "cukd/label_mask.h"              // from the label-pruning add-on
#include "cukd/traversal_label_pruned.h"  // from the label-pruning add-on

// ----------------------- data & traits -----------------------
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

// label accessor specialization
namespace cukd {
namespace labels {
template <>
struct default_label_traits<LPoint3f> {
  __host__ __device__ static inline int get_label(const LPoint3f& d) { return d.label; }
};
}  // namespace labels
}  // namespace cukd

// ----------------------- helpers -----------------------------
static inline float dist2(const float3& a, const float3& b) {
  float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

#define CUDA_CHECK(call)                                                                      \
  do {                                                                                        \
    cudaError_t err = (call);                                                                 \
    if (err != cudaSuccess) {                                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

// reference CPU nearest with same label; returns {-1,inf} if none
static std::pair<int, float> ref_nearest_same_label(const std::vector<LPoint3f>& pts, const float3& q, int label) {
  int best = -1;
  float bestD2 = INFINITY;
  for (int i = 0; i < (int)pts.size(); ++i) {
    if (pts[i].label != label) continue;
    float d2 = dist2(pts[i].p, q);
    if (d2 < bestD2 || (d2 == bestD2 && i < best)) {
      best = i;
      bestD2 = d2;
    }
  }
  return {best, bestD2};
}

// ----------------------- device query kernel -----------------
__global__ void
kernel_fcp_same_label(const float3* queries, const int* q_labels, int M, const LPoint3f* nodes, int N, const cukd::labels::Mask* masks, int* out_idx, float* out_d2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  cukd::labels::Mask desired;
  desired.clear();
  if (q_labels[i] >= 0) desired.setBit(q_labels[i]);
  float best;
  int idx = cukd::labels::fcp_label_filtered<LPoint3f, LPoint3f_traits, float3>(queries[i], nodes, N, masks, desired, best);
  out_idx[i] = idx;
  out_d2[i] = best;
}

int main(int argc, char** argv) {
  const int N = (argc > 1 ? std::max(1000, atoi(argv[1])) : 100000);   // points
  const int M = (argc > 2 ? std::max(100, atoi(argv[2])) : 2000);      // queries
  const int NUM_LABELS = (argc > 3 ? std::max(2, atoi(argv[3])) : 8);  // label classes
  printf("Building test with N=%d, M=%d, labels=%d\n", N, M, NUM_LABELS);

  // ----------------------- synth data ------------------------
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  std::uniform_int_distribution<int> lab(0, NUM_LABELS - 1);

  std::vector<LPoint3f> h_pts(N);
  std::vector<int> label_counts(NUM_LABELS, 0);
  for (int i = 0; i < N; i++) {
    h_pts[i].p = make_float3(uni(rng), uni(rng), uni(rng));
    h_pts[i].label = lab(rng);
    label_counts[h_pts[i].label]++;
  }
  // ensure every label appears at least once
  for (int L = 0; L < NUM_LABELS; L++)
    if (label_counts[L] == 0) {
      h_pts[L].label = L;
      label_counts[L]++;
    }

  // queries: random positions + random labels that exist
  std::vector<float3> h_q(M);
  std::vector<int> h_q_label(M);
  for (int i = 0; i < M; i++) {
    h_q[i] = make_float3(uni(rng) * 2.f, uni(rng) * 2.f, uni(rng) * 2.f);  // a bit wider
    int L;
    do {
      L = lab(rng);
    } while (label_counts[L] == 0);
    h_q_label[i] = L;
  }

  // ----------------------- device mem ------------------------
  LPoint3f* d_nodes = nullptr;
  CUDA_CHECK(cudaMalloc(&d_nodes, N * sizeof(LPoint3f)));
  CUDA_CHECK(cudaMemcpy(d_nodes, h_pts.data(), N * sizeof(LPoint3f), cudaMemcpyHostToDevice));

  // build kd-tree (level-order) in-place
  cukd::buildTree<LPoint3f, LPoint3f_traits>(d_nodes, N);

  // after buildTree(...) and build_label_masks(...)
  std::vector<LPoint3f> h_built(N);
  CUDA_CHECK(cudaMemcpy(h_built.data(), d_nodes, N * sizeof(LPoint3f), cudaMemcpyDeviceToHost));

  // build label masks
  cukd::labels::Mask* d_masks = nullptr;
  CUDA_CHECK(cudaMalloc(&d_masks, N * sizeof(cukd::labels::Mask)));
  cukd::labels::build_label_masks<LPoint3f>(d_nodes, N, d_masks);

  float3* d_q = nullptr;
  int* d_ql = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, M * sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_ql, M * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), M * sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ql, h_q_label.data(), M * sizeof(int), cudaMemcpyHostToDevice));

  int* d_idx = nullptr;
  float* d_d2 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_idx, M * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_d2, M * sizeof(float)));

  dim3 BS(128), GS((M + BS.x - 1) / BS.x);
  kernel_fcp_same_label<<<GS, BS>>>(d_q, d_ql, M, d_nodes, N, d_masks, d_idx, d_d2);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> got_idx(M);
  std::vector<float> got_d2(M);
  CUDA_CHECK(cudaMemcpy(got_idx.data(), d_idx, M * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_d2.data(), d_d2, M * sizeof(float), cudaMemcpyDeviceToHost));

  // ----------------------- check vs CPU ----------------------
  int failures = 0;
  double max_rel_err = 0.0;
  for (int i = 0; i < M; i++) {
    auto [ref_idx, ref_d2] = ref_nearest_same_label(h_pts, h_q[i], h_q_label[i]);
    if (ref_idx < 0) {
      if (got_idx[i] >= 0) {
        fprintf(stderr, "Query %d: expected no match, but GPU found %d\n", i, got_idx[i]);
        failures++;
      }
      continue;
    }
    if (got_idx[i] < 0) {
      fprintf(stderr, "Query %d: GPU found no match (expected %d)\n", i, ref_idx);
      failures++;
      continue;
    }
    if (h_built[got_idx[i]].label != h_q_label[i]) {
      fprintf(stderr, "Query %d: wrong label (%d != %d)\n", i, h_built[got_idx[i]].label, h_q_label[i]);
      failures++;
    }
    double denom = std::max(1e-6f, ref_d2);
    double rel = std::fabs(got_d2[i] - ref_d2) / denom;
    if (rel > max_rel_err) max_rel_err = rel;
    if (rel > 1e-5) {
      fprintf(stderr, "Query %d: distance mismatch gpu=%g ref=%g (rel=%g)\n", i, got_d2[i], ref_d2, rel);
      failures++;
    }
  }

  if (failures == 0)
    printf("PASS: all %d queries matched (max rel err %.3g).\n", M, max_rel_err);
  else
    printf("FAIL: %d/%d queries mismatched. Max rel err %.3g.\n", failures, M, max_rel_err);

  // cleanup
  cudaFree(d_nodes);
  cudaFree(d_masks);
  cudaFree(d_q);
  cudaFree(d_ql);
  cudaFree(d_idx);
  cudaFree(d_d2);
  return failures == 0 ? 0 : 1;
}
