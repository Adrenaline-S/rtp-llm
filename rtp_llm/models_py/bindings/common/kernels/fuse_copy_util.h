#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>

namespace rtp_llm {

// Hard caps on copies fused into a single kernel launch. The structs below
// are passed by value as kernel parameters, so the arrays must be sized at
// compile time.
//
// Sizing rationale (worst-case callers as of 2026):
//   * cuda_graph_runner.cc::prepareInputs accumulates ~8 contiguous copies
//     plus 1 + group_count strided copies per launch (one launch per replay).
//   * PyWrappedModel.cc::forwardMicroBatched is the tightest path: it
//     accumulates across ALL micro-batches before a single flush. Per
//     micro-batch it adds ~8 contiguous copies (5 from buildPyAttentionInputs,
//     1 padding_offset, and up to 2 whole tagged block tables). With the current
//     planMicroBatches cap of 2 micro-batches that's 8 * 2 = 16 copies.
//
// 64 entries gives 4x headroom over today's worst case (16 contiguous, 5
// strided) and accommodates ~30 KV-cache groups before hitting the cap. Each
// FusedStridedCopyParams is 8 * 8 * 64 + 4 = 4100 bytes, well under the 32 KB
// kernel parameter buffer available on Volta and newer GPUs (all currently
// supported targets).
//
// If you need to raise these further: bump the constant, re-check the kernel
// parameter buffer size for the lowest supported compute capability, and
// extend the MaxFusedCopies / micro-batch unit tests accordingly. If the
// upper bound ever needs to be unbounded, prefer adding a chunked-launch
// helper (split into multiple param structs and launch each) over making the
// arrays dynamic — the kernel signature must stay POD for grid launch.
static constexpr int MAX_FUSED_D2D_COPIES     = 64;
static constexpr int MAX_FUSED_STRIDED_COPIES = 64;

inline void copyParamsAssert(bool value, const std::string& msg) {
    if (!value) {
        throw std::runtime_error(msg);
    }
}

struct FusedD2DCopyParams {
    const void* src[MAX_FUSED_D2D_COPIES];
    void*       dst[MAX_FUSED_D2D_COPIES];
    size_t      size[MAX_FUSED_D2D_COPIES];
    int         num_copies = 0;

    void add(const void* src_ptr, void* dst_ptr, size_t bytes) {
        copyParamsAssert(num_copies < MAX_FUSED_D2D_COPIES,
                         "FusedD2DCopyParams: num_copies (" + std::to_string(num_copies + 1)
                             + ") exceeds MAX_FUSED_D2D_COPIES (" + std::to_string(MAX_FUSED_D2D_COPIES)
                             + "). Bump the cap in fuse_copy_util.h after re-checking the sizing rationale.");
        src[num_copies]  = src_ptr;
        dst[num_copies]  = dst_ptr;
        size[num_copies] = bytes;
        ++num_copies;
    }

    void clear() {
        num_copies = 0;
    }
};

struct FusedStridedCopyParams {
    const void* src[MAX_FUSED_STRIDED_COPIES];
    void*       dst[MAX_FUSED_STRIDED_COPIES];
    size_t      num_rows[MAX_FUSED_STRIDED_COPIES];
    size_t      row_bytes[MAX_FUSED_STRIDED_COPIES];
    size_t      dst_num_rows[MAX_FUSED_STRIDED_COPIES];
    size_t      dst_row_bytes[MAX_FUSED_STRIDED_COPIES];
    size_t      src_row_stride[MAX_FUSED_STRIDED_COPIES];
    size_t      dst_row_stride[MAX_FUSED_STRIDED_COPIES];
    int         num_copies = 0;

    void add(const void* src_ptr,
             void*       dst_ptr,
             size_t      rows,
             size_t      row_b,
             size_t      src_stride,
             size_t      dst_stride,
             size_t      dst_rows = 0,
             size_t      dst_row_b = 0) {
        copyParamsAssert(num_copies < MAX_FUSED_STRIDED_COPIES,
                         "FusedStridedCopyParams: num_copies (" + std::to_string(num_copies + 1)
                             + ") exceeds MAX_FUSED_STRIDED_COPIES (" + std::to_string(MAX_FUSED_STRIDED_COPIES)
                             + "). Bump the cap in fuse_copy_util.h after re-checking the sizing rationale.");
        const size_t target_rows      = dst_rows == 0 ? rows : dst_rows;
        const size_t target_row_bytes = dst_row_b == 0 ? row_b : dst_row_b;
        copyParamsAssert(target_rows >= rows && target_row_bytes >= row_b,
                         "FusedStridedCopyParams: destination shape must contain source shape");
        src[num_copies]            = src_ptr;
        dst[num_copies]            = dst_ptr;
        num_rows[num_copies]       = rows;
        row_bytes[num_copies]      = row_b;
        dst_num_rows[num_copies]   = target_rows;
        dst_row_bytes[num_copies]  = target_row_bytes;
        src_row_stride[num_copies] = src_stride;
        dst_row_stride[num_copies] = dst_stride;
        ++num_copies;
    }

    void clear() {
        num_copies = 0;
    }
};
}  // namespace rtp_llm
