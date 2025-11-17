#pragma once
// Label-pruned stack-free kd-tree traversal (generic + single-bit fast paths)
// Optimizations applied:
//  - Lazy close-child mask check (avoid reading mask for the near child).
//  - Depth-gated prune: skip subtree-mask check for the top K levels (default 6).
//  - Maintain current depth explicitly to avoid repeated __clz() level() calls.
//  - Early-exit on first intersecting mask word to reduce memory traffic.
//  - Single-bit desired-label fast path works for any number of mask words.
//
// Tunables (compile-time):
//   -DCUKD_PRUNE_SKIP_LEVELS=6
//   -DCUKD_LABEL_BITS_PER_WORD={64|32|16}
//   -DCUKD_LABEL_MASK_WORDS={1|2|...}

#include <cuda_runtime.h>
#include <math.h>
#include "cukd/label_mask.h"

#ifndef CUKD_PRUNE_SKIP_LEVELS
#define CUKD_PRUNE_SKIP_LEVELS 6
#endif

namespace cukd {
namespace labels {

#ifndef CUKD_LABEL_BITS_PER_WORD
#define CUKD_LABEL_BITS_PER_WORD 16
#endif

#ifndef CUKD_LABEL_MASK_WORDS
#define CUKD_LABEL_MASK_WORDS 1
#endif

#if   (CUKD_LABEL_BITS_PER_WORD==64)
  using cukd_label_word_t = uint64_t;
#elif (CUKD_LABEL_BITS_PER_WORD==32)
  using cukd_label_word_t = uint32_t;
#elif (CUKD_LABEL_BITS_PER_WORD==16)
  using cukd_label_word_t = uint16_t;
#else
  #error "CUKD_LABEL_BITS_PER_WORD must be 16, 32, or 64"
#endif

template<typename P>
__device__ __forceinline__ float coord_of(const P& p, int dim){
  return reinterpret_cast<const float*>(&p)[dim];
}

// compute split dim using explicit dim if provided by data_traits, otherwise
// use current traversal depth modulo K (avoids LBHeapTopology::level(idx)).
template<typename data_traits, typename data_t>
__device__ __forceinline__ int get_split_dim(const data_t &node, int /*idx_unused*/, int depth){
  if constexpr (data_traits::has_explicit_dim) {
    return data_traits::get_dim(node);
  } else {
    const int K = data_traits::num_dims;
    return K ? (depth % K) : 0;
  }
}

template<typename data_traits>
struct split_dim_helper {
  template<typename data_t>
  __device__ __forceinline__ static int get(const data_t &node, int idx){
    if constexpr (data_traits::has_explicit_dim) {
      return data_traits::get_dim(node);
    } else {
      const int K = data_traits::num_dims;
      return K ? (LBHeapTopology::level(idx) % K) : 0;
    }
  }
};

namespace detail {
  __device__ __forceinline__ float dsq(const float2&a,const float2&b){ float dx=a.x-b.x, dy=a.y-b.y; return dx*dx+dy*dy; }
  __device__ __forceinline__ float dsq(const float3&a,const float3&b){ float dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z; return dx*dx+dy*dy+dz*dz; }
  __device__ __forceinline__ float dsq(const float4&a,const float4&b){ float dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z; return dx*dx+dy*dy+dz*dz; }
}

// ---------------- generic (multi-label) ----------------
template<typename data_t, typename data_traits, typename point_t>
__device__ __forceinline__
int fcp_label_filtered_generic(const point_t &query,
                               const data_t * __restrict__ nodes,
                               int N,
                               const Mask * __restrict__ masks,
                               const Mask desiredMask,
                               float &out_bestD2)
{
  if (N<=0) { out_bestD2 = CUDART_INF_F; return -1; }
  int curr=0, prev=-1, best=-1; float bestD2 = CUDART_INF_F;
  int depth = 0; // track depth explicitly to avoid level() calls

  while (curr != -1){
    const int parent = LBHeapTopology::parent(curr);
    const bool from_parent = (prev == parent);

    if (from_parent){
      // depth-gated: skip subtree-mask prune on top levels where masks are dense
      if (depth > CUKD_PRUNE_SKIP_LEVELS) {
        bool hit=false;
        #pragma unroll
        for (int w=0; w<CUKD_LABEL_MASK_WORDS; ++w) {
          if ((masks[curr].w[w] & desiredMask.w[w]) != 0) { hit = true; break; }
        }
        if (!hit) { prev=curr; curr=parent; depth -= 1; continue; }
      }

      int nodeLabel = default_label_traits<data_t>::get_label(nodes[curr]);
      if (nodeLabel >= 0){
        const int wi = nodeLabel / CUKD_LABEL_BITS_PER_WORD;
        const int bi = nodeLabel % CUKD_LABEL_BITS_PER_WORD;
        const cukd_label_word_t bit = (cukd_label_word_t(1) << bi);
        if (wi < CUKD_LABEL_MASK_WORDS && (desiredMask.w[wi] & bit)){
          const auto &p = data_traits::get_point(nodes[curr]);
          float dd = detail::dsq(p, query);
          if (dd < bestD2){ bestD2 = dd; best = curr; }
        }
      }
    }

    // choose child (lazy close-child mask check)
    const int sd = get_split_dim<data_traits>(nodes[curr], curr, depth);
    const float splitPos = data_traits::get_coord(nodes[curr], sd);
    const float signedDist = coord_of(query, sd) - splitPos;
    const float signedDist2 = signedDist * signedDist;
    const int closeSide  = (signedDist > 0.f);
    const int closeChild = LBHeapTopology::left(curr) + closeSide;
    const int farChild   = LBHeapTopology::right(curr) - closeSide;

    int next;
    if (from_parent){
      if (LBHeapTopology::valid(closeChild,N)) {
        next = closeChild; // do not read closeChild mask here; lazy check next iteration
      } else {
        bool goFar = false;
        if (LBHeapTopology::valid(farChild,N) && signedDist2 <= bestD2){
          // we *do* need a far-child mask test
          #pragma unroll
          for (int w=0; w<CUKD_LABEL_MASK_WORDS; ++w) { if ((masks[farChild].w[w] & desiredMask.w[w]) != 0) { goFar = true; break; } }
        }
        next = goFar ? farChild : parent;
      }
    } else if (prev == closeChild) {
      bool goFar = false;
      if (LBHeapTopology::valid(farChild,N) && signedDist2 <= bestD2){
        #pragma unroll
        for (int w=0; w<CUKD_LABEL_MASK_WORDS; ++w) { if ((masks[farChild].w[w] & desiredMask.w[w]) != 0) { goFar = true; break; } }
      }
      next = goFar ? farChild : parent;
    } else {
      next = parent;
    }
    // update depth for new current node
    if (next == parent) depth -= 1; else if (next == closeChild || next == farChild) depth += 1;
    prev=curr; curr=next;
  }
  out_bestD2 = bestD2; return best;
}

// ---------------- single-bit fast path (works for any number of words) ----------------
template<typename data_t, typename data_traits, typename point_t>
__device__ __forceinline__
int fcp_label_single_label(const point_t &query,
                          const data_t * __restrict__ nodes,
                          int N,
                          const Mask * __restrict__ masks,
                          int desiredLabel,
                          int desired_word,
                          cukd_label_word_t desired_bit,
                          float &out_bestD2)
{
  if (N<=0 || desiredLabel < 0){ out_bestD2 = CUDART_INF_F; return -1; }

  int curr=0, prev=-1, best=-1; float bestD2 = CUDART_INF_F;
  int depth = 0;
  while (curr!=-1){
    const int parent = LBHeapTopology::parent(curr);
    const bool from_parent = (prev==parent);

    if (from_parent){
      if (depth > CUKD_PRUNE_SKIP_LEVELS) {
        if ( (masks[curr].w[desired_word] & desired_bit) == 0 ) { prev=curr; curr=parent; depth -= 1; continue; }
      }

      const int nodeLabel = default_label_traits<data_t>::get_label(nodes[curr]);
      if (nodeLabel == desiredLabel) {
        const auto &p = data_traits::get_point(nodes[curr]);
        const float dd = detail::dsq(p, query);
        if (dd < bestD2){ bestD2=dd; best=curr; }
      }
    }

    // lazy close-child check
    const int sd = get_split_dim<data_traits>(nodes[curr], curr, depth);
    const float splitPos = data_traits::get_coord(nodes[curr], sd);
    const float signedDist = coord_of(query, sd) - splitPos;
    const float signedDist2 = signedDist * signedDist;
    const int closeSide  = (signedDist > 0.f);
    const int closeChild = LBHeapTopology::left(curr) + closeSide;
    const int farChild   = LBHeapTopology::right(curr) - closeSide;

    int next;
    if (from_parent){
      if (LBHeapTopology::valid(closeChild,N)) {
        next = closeChild;
      } else {
        const bool goFar = LBHeapTopology::valid(farChild,N) &&
                           (signedDist2 <= bestD2) &&
                           ( (depth > CUKD_PRUNE_SKIP_LEVELS) ? ((masks[farChild].w[desired_word] & desired_bit) != 0) : true );
        next = goFar ? farChild : parent;
      }
    } else if (prev == closeChild) {
      const bool goFar = LBHeapTopology::valid(farChild,N) &&
                         (signedDist2 <= bestD2) &&
                         ( (depth > CUKD_PRUNE_SKIP_LEVELS) ? ((masks[farChild].w[desired_word] & desired_bit) != 0) : true );
      next = goFar ? farChild : parent;
    } else {
      next = parent;
    }
    if (next == parent) depth -= 1; else if (next == closeChild || next == farChild) depth += 1;
    prev=curr; curr=next;
  }
  out_bestD2 = bestD2; return best;
}

// ---------------- wrapper ----------------
template<typename data_t, typename data_traits, typename point_t>
__device__ __forceinline__
int fcp_label_filtered(const point_t &query,
                       const data_t * __restrict__ nodes,
                       int N,
                       const Mask * __restrict__ masks,
                       const Mask desiredMask,
                       float &out_bestD2)
{
  // if desiredMask has exactly one bit set, use the optimized single-label path
  int desired_word = -1;
  int bit_index = -1;
  cukd_label_word_t desired_bit = 0;
  // find first set bit across words
  #pragma unroll
  for (int w=0; w<CUKD_LABEL_MASK_WORDS; ++w){
    cukd_label_word_t mw = desiredMask.w[w];
    if (mw){
      // check uniqueness: exactly one bit overall -> only one 'mw' nonzero and it has popcount==1
      const bool unique_word = ((mw & (mw-1)) == 0);
      bool others_zero = true;
      #pragma unroll
      for (int u=0; u<CUKD_LABEL_MASK_WORDS; ++u){ if (u!=w && desiredMask.w[u]) { others_zero=false; break; } }
      if (unique_word && others_zero){
        desired_word = w;
        // position of first bit
        int bi = 
#if   (CUKD_LABEL_BITS_PER_WORD==64)
          (__ffsll((unsigned long long)mw) - 1);
#else
          (__ffs((int)mw) - 1);
#endif
        bit_index = bi;
        desired_bit = (cukd_label_word_t(1) << bi);
      }
      break;
    }
  }
  if (desired_word >= 0 && bit_index >= 0){
    int desiredLabel = desired_word * CUKD_LABEL_BITS_PER_WORD + bit_index;
    return fcp_label_single_label<data_t, data_traits, point_t>(query, nodes, N, masks,
                                                                desiredLabel, desired_word, desired_bit,
                                                                out_bestD2);
  }
  return fcp_label_filtered_generic<data_t, data_traits, point_t>(query, nodes, N, masks, desiredMask, out_bestD2);
}

} // namespace labels
} // namespace cukd
