#pragma once
// Label-aware subtree masks for cudaKDTree with selectable word size (64/32/16 bits).
// Default: 64-bit. Override with -DCUKD_LABEL_BITS_PER_WORD=32 or 16.
// If your total labels <= CUKD_LABEL_BITS_PER_WORD, also set -DCUKD_LABEL_MASK_WORDS=1
// to enable the fastest single-word path.

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CUKD_LABEL_BITS_PER_WORD
#define CUKD_LABEL_BITS_PER_WORD 16
#endif

#ifndef CUKD_LABEL_MASK_WORDS
#define CUKD_LABEL_MASK_WORDS 1
#endif

#if   (CUKD_LABEL_BITS_PER_WORD==64)
  using cukd_label_word_t = uint64_t;
  #define CUKD_LABEL_FFS(x) __ffsll((unsigned long long)(x))
#elif (CUKD_LABEL_BITS_PER_WORD==32)
  using cukd_label_word_t = uint32_t;
  #define CUKD_LABEL_FFS(x) __ffs((int)(x))
#elif (CUKD_LABEL_BITS_PER_WORD==16)
  using cukd_label_word_t = uint16_t;
  #define CUKD_LABEL_FFS(x) __ffs((int)(x))
#else
  #error "CUKD_LABEL_BITS_PER_WORD must be 16, 32, or 64"
#endif

namespace cukd {
namespace labels {

// Left-balanced heap topology (as in cudaKDTree)
struct LBHeapTopology {
  __host__ __device__ static inline int parent(int idx){ return idx?((idx-1)>>1) : -1; }
  __host__ __device__ static inline int left(int idx){ return (idx<<1)+1; }
  __host__ __device__ static inline int right(int idx){ return (idx<<1)+2; }
  __host__ __device__ static inline bool valid(int idx,int N){ return idx>=0 && idx<N; }
  __host__ __device__ static inline int level(int idx){
  #if defined(__CUDA_ARCH__)
    return 31 - __clz((unsigned)(idx+1));
  #elif defined(__GNUG__)
    return 31 - __builtin_clz((unsigned)(idx+1));
  #else
    int v = idx+1, l = -1; while (v){ v>>=1; ++l; } return (l<0)?0:l;
  #endif
  }
};

struct Mask {
  cukd_label_word_t w[CUKD_LABEL_MASK_WORDS];
  __host__ __device__ inline void clear(){
    #pragma unroll
    for (int i=0;i<CUKD_LABEL_MASK_WORDS;i++) w[i]=0;
  }
  __host__ __device__ inline void setBit(int bit){
    if (bit < 0) return;
    const int i = bit / CUKD_LABEL_BITS_PER_WORD;
    const int b = bit % CUKD_LABEL_BITS_PER_WORD;
    if (i < CUKD_LABEL_MASK_WORDS)
      w[i] |= (cukd_label_word_t(1) << b);
  }
  __host__ __device__ inline void orWith(const Mask &o){
    #pragma unroll
    for (int i=0;i<CUKD_LABEL_MASK_WORDS;i++) w[i] |= o.w[i];
  }
  __host__ __device__ inline bool intersects(const Mask &o) const{
    #pragma unroll
    for (int i=0;i<CUKD_LABEL_MASK_WORDS;i++) if (w[i] & o.w[i]) return true;
    return false;
  }
};

// Users can specialize this for their data type.
template<typename data_t>
struct default_label_traits {
  __host__ __device__ static inline int get_label(const data_t &d){ return d.label; }
};

// ---- kernels to build subtree masks bottom-up ----
template<typename data_t, typename label_traits=default_label_traits<data_t> >
static __global__ void init_node_masks(const data_t * __restrict__ nodes,
                                       int N, Mask * __restrict__ masks){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>=N) return;
  Mask m; m.clear();
  int L = label_traits::get_label(nodes[i]);
  if (L>=0) m.setBit(L);
  masks[i]=m;
}

static __global__ void reduce_level_masks_kernel(int start, int end, int N,
                                                 Mask * __restrict__ masks){
  int i = start + blockIdx.x*blockDim.x + threadIdx.x;
  if (i> end) return;
  int L = LBHeapTopology::left(i), R = LBHeapTopology::right(i);
  Mask m = masks[i];
  if (LBHeapTopology::valid(L,N)) m.orWith(masks[L]);
  if (LBHeapTopology::valid(R,N)) m.orWith(masks[R]);
  masks[i] = m;
}

__host__ __device__ inline int level_start(int lvl){ return (1<<lvl)-1; }
__host__ __device__ inline int level_end(int lvl, int N){
  int e = (1<<(lvl+1)) - 2; return e < (N-1) ? e : (N-1);
}

template<typename data_t, typename label_traits=default_label_traits<data_t> >
inline void build_label_masks(const data_t *d_nodes, int N, Mask *d_masks, cudaStream_t stream=0){
  if (N<=0) return;
  const int BS=256; const int GS=(N+BS-1)/BS;
  init_node_masks<data_t,label_traits><<<GS,BS,0,stream>>>(d_nodes,N,d_masks);

  int h;
#if defined(__GNUG__) || defined(__CUDA_ARCH__)
  h = 31 - __builtin_clz((unsigned)N);
#else
  h = 0; for (int t=N; t>1; t>>=1) ++h;
#endif

  for (int lvl = h-1; lvl>=0; --lvl){
    int s = level_start(lvl);
    int e = level_end(lvl,N);
    if (s>e) continue;
    int count = e - s + 1;
    int g = (count + BS - 1) / BS;
    reduce_level_masks_kernel<<<g,BS,0,stream>>>(s,e,N,d_masks);
  }
}

} // namespace labels
} // namespace cukd
