#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "cukd/builder.h"
#include "cukd/label_mask.h"                // your fixed header
#include "cukd/traversal_label_pruned.h"    // your fixed header

#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ----------------------- data & traits -----------------------
struct LPoint3f { float3 p; int label; };

// IMPORTANT: keep this aligned with your environment
struct LPoint3f_traits : public cukd::default_data_traits<float3> {
  using point_t = float3;
  static constexpr int num_dims = 3;
  enum { has_explicit_dim = false };

  __host__ __device__ static inline const float3 &get_point(const LPoint3f &d){ return d.p; }
  __host__ __device__ static inline float3 &get_point(LPoint3f &d){ return d.p; }
  __host__ __device__ static inline float get_coord(const LPoint3f &d, int dim){ return cukd::get_coord(d.p,dim); }
};

namespace cukd { namespace labels {
  template<> struct default_label_traits<LPoint3f> {
    __host__ __device__ static inline int get_label(const LPoint3f &d){ return d.label; }
  };
}}

// ----------------------- kernels ------------------------------
using cukd::labels::Mask;

__global__ void kernel_fcp_label_same(const float3 *queries, const int *q_labels, int M,
                                      const LPoint3f *nodes, int N,
                                      const Mask *masks,
                                      int *out_idx, float *out_d2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i>=M) return;
  Mask desired; desired.clear(); desired.setBit(q_labels[i]); // single-label filter
  float best;
  int idx = cukd::labels::fcp_label_filtered<LPoint3f, LPoint3f_traits, float3>(
              queries[i], nodes, N, masks, desired, best);
  out_idx[i]=idx; out_d2[i]=best;
}

// Baseline: same traversal *without* pruning by passing an "all labels allowed" mask.
__device__ inline Mask make_all_mask() {
  Mask m;
  #pragma unroll
  for (int w=0; w<CUKD_LABEL_MASK_WORDS; ++w) m.w[w]=~0ull;
  return m;
}
__global__ void kernel_fcp_unfiltered(const float3 *queries, int M,
                                      const LPoint3f *nodes, int N,
                                      const Mask *masks, // still read masks, but they don't prune
                                      int *out_idx, float *out_d2)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i>=M) return;
  float best;
  int idx = cukd::labels::fcp_label_filtered<LPoint3f, LPoint3f_traits, float3>(
              queries[i], nodes, N, masks, make_all_mask(), best);
  out_idx[i]=idx; out_d2[i]=best;
}

// ----------------------- CPU ref (optional spot check) -------
static inline float d2(const float3&a,const float3&b){
  float dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z; return dx*dx+dy*dy+dz*dz;
}
static std::pair<int,float> ref_same_label(const std::vector<LPoint3f> &pts, const float3 &q, int L){
  int best=-1; float bd=INFINITY;
  for (int i=0;i<(int)pts.size();++i) if (pts[i].label==L){
    float dd=d2(pts[i].p,q); if (dd<bd){ bd=dd; best=i; }
  }
  return {best,bd};
}

// ----------------------- timing helper -----------------------
float time_kernel(std::function<void(cudaEvent_t, cudaEvent_t)> launch, int repeats=5){
  cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  // warmup
  for (int i=0;i<2;i++) launch(start,stop);
  CUDA_CHECK(cudaDeviceSynchronize());
  float total_ms=0.f;
  for (int i=0;i<repeats;i++){
    launch(start,stop);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
    total_ms+=ms;
  }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms/repeats;
}

// ----------------------- main -----------------------
int main(int argc, char**){
  // Tunables
  int N = 1<<20;       // points
  int M = 1<<20;       // queries
  int NUM_LABELS = 16; // classes
  int REPEATS = 5;

  printf("Benchmark: N=%d points, M=%d queries, labels=%d\n", N, M, NUM_LABELS);

  // Data
  std::mt19937 rng(0xBADC0DE);
  std::uniform_real_distribution<float> uni(-1.f,1.f);
  std::uniform_int_distribution<int> lab(0,NUM_LABELS-1);

  std::vector<LPoint3f> h_pts(N);
  for (int i=0;i<N;i++){
    h_pts[i].p = make_float3(uni(rng),uni(rng),uni(rng));
    h_pts[i].label = lab(rng);
  }

  // Device buffers
  LPoint3f *d_nodes; CUDA_CHECK(cudaMalloc(&d_nodes, N*sizeof(LPoint3f)));
  CUDA_CHECK(cudaMemcpy(d_nodes, h_pts.data(), N*sizeof(LPoint3f), cudaMemcpyHostToDevice));

  // Build kd-tree in-place and label masks
  cukd::buildTree<LPoint3f, LPoint3f_traits>(d_nodes, N);

  Mask *d_masks; CUDA_CHECK(cudaMalloc(&d_masks, N*sizeof(Mask)));
  cukd::labels::build_label_masks<LPoint3f>(d_nodes, N, d_masks);

  // Queries: random locations + random target label (same-label search)
  std::vector<float3> h_q(M); std::vector<int> h_L(M);
  for (int i=0;i<M;i++){ h_q[i]=make_float3(uni(rng),uni(rng),uni(rng)); h_L[i]=lab(rng); }

  float3 *d_q; int *d_L; CUDA_CHECK(cudaMalloc(&d_q, M*sizeof(float3)));
  CUDA_CHECK(cudaMalloc(&d_L, M*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), M*sizeof(float3), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_L, h_L.data(), M*sizeof(int), cudaMemcpyHostToDevice));

  int *d_idx; float *d_d2; CUDA_CHECK(cudaMalloc(&d_idx, M*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_d2, M*sizeof(float)));

  dim3 BS(256), GS((M+BS.x-1)/BS.x);

  // Baseline (unfiltered)
  auto launch_unfiltered = [&](cudaEvent_t start, cudaEvent_t stop){
    CUDA_CHECK(cudaEventRecord(start));
    kernel_fcp_unfiltered<<<GS,BS>>>(d_q, M, d_nodes, N, d_masks, d_idx, d_d2);
    CUDA_CHECK(cudaEventRecord(stop));
  };
  float t_unf_ms = time_kernel(launch_unfiltered, REPEATS);

  // Label-pruned (same-label)
  auto launch_label = [&](cudaEvent_t start, cudaEvent_t stop){
    CUDA_CHECK(cudaEventRecord(start));
    kernel_fcp_label_same<<<GS,BS>>>(d_q, d_L, M, d_nodes, N, d_masks, d_idx, d_d2);
    CUDA_CHECK(cudaEventRecord(stop));
  };
  float t_lab_ms = time_kernel(launch_label, REPEATS);

  // Throughput
  double qps_unf = (double)M / (t_unf_ms/1000.0);
  double qps_lab = (double)M / (t_lab_ms/1000.0);

  printf("\n=== Results ===\n");
  printf("Unfiltered:  %.3f ms  |  %.1f Mq/s\n", t_unf_ms, qps_unf/1e6);
  printf("Label-pruned:%.3f ms  |  %.1f Mq/s\n", t_lab_ms, qps_lab/1e6);
  printf("Speedup:     %.2fx\n", t_unf_ms / t_lab_ms);

  // Optional: spot-check correctness on 32 random queries
  std::vector<LPoint3f> h_built(N);
  CUDA_CHECK(cudaMemcpy(h_built.data(), d_nodes, N*sizeof(LPoint3f), cudaMemcpyDeviceToHost));
  std::vector<int> got_idx(32); std::vector<float> got_d2(32);
  CUDA_CHECK(cudaMemcpy(got_idx.data(), d_idx, 32*sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_d2.data(),  d_d2,  32*sizeof(float), cudaMemcpyDeviceToHost));
  for (int i=0;i<32;i++){
    auto [ri, rd] = ref_same_label(h_built, h_q[i], h_L[i]);
    if (ri>=0 && got_idx[i]>=0 && h_built[got_idx[i]].label==h_L[i]){
      double rel = std::fabs(got_d2[i]-rd)/std::max(1e-6f,rd);
      if (rel>1e-4) fprintf(stderr,"[check %d] rel err %.3g\n", i, rel);
    }
  }

  // Cleanup
  cudaFree(d_nodes); cudaFree(d_masks); cudaFree(d_q); cudaFree(d_L); cudaFree(d_idx); cudaFree(d_d2);
  return 0;
}
