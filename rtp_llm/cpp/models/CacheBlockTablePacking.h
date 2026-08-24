#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <torch/extension.h>

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

struct PackedBlockTableRegion {
    size_t offset         = 0;
    size_t row_width      = 0;
    size_t batch_capacity = 0;

    size_t numel() const;
};

struct CacheGroupBlockTableRegion {
    std::string            tag;
    uint32_t               execution_ordinal = 0;
    PackedBlockTableRegion pool;
    PackedBlockTableRegion kernel;
};

class CacheBlockTablePackingPlan {
public:
    static CacheBlockTablePackingPlan create(const CacheConfig& config, size_t batch_capacity, size_t max_pool_blocks);
    static CacheBlockTablePackingPlan
    create(const CacheConfig& config, size_t batch_capacity, const std::vector<size_t>& pool_row_widths);

    static CacheBlockTablePackingPlan fromRegions(std::vector<CacheGroupBlockTableRegion> groups);

    const CacheGroupBlockTableRegion& group(uint32_t execution_ordinal) const;
    size_t                            groupCount() const noexcept;
    size_t                            batchCapacity() const noexcept;
    size_t                            poolNumel() const noexcept;
    size_t                            kernelNumel() const noexcept;

    torch::Tensor poolView(const torch::Tensor& backing, uint32_t execution_ordinal) const;
    torch::Tensor kernelView(const torch::Tensor& backing, uint32_t execution_ordinal) const;

private:
    static CacheBlockTablePackingPlan createValidated(std::vector<CacheGroupBlockTableRegion> groups);
    torch::Tensor view(const torch::Tensor& backing, const PackedBlockTableRegion& region, const char* name) const;

    std::vector<CacheGroupBlockTableRegion> groups_;
    size_t                                  batch_capacity_ = 0;
    size_t                                  pool_numel_     = 0;
    size_t                                  kernel_numel_   = 0;
};

// Lossless int64 wire representation used when TP ranks reconstruct the
// authoritative root packing plan. The wire includes tag identity and every
// region field that participates in CacheBlockTablePackingSignature.
std::vector<int64_t>       encodeCacheBlockTablePackingPlan(const CacheBlockTablePackingPlan& plan);
CacheBlockTablePackingPlan decodeCacheBlockTablePackingPlan(const std::vector<int64_t>& wire);
CacheBlockTablePackingPlan decodeCacheBlockTablePackingPlan(const int64_t* wire, size_t wire_size);

struct PackedBlockTableSnapshot {
    torch::Tensor              pool_host;
    torch::Tensor              pool_device;
    torch::Tensor              kernel_host;
    torch::Tensor              kernel_device;
    std::vector<torch::Tensor> pool_valid_lengths;
    std::vector<torch::Tensor> kernel_valid_lengths;
    CacheBlockTablePackingPlan plan;
};

torch::Tensor packKernelValidLengthsForTp(const std::vector<torch::Tensor>& valid_lengths,
                                          const CacheBlockTablePackingPlan& plan);
void          unpackKernelValidLengthsFromTp(const torch::Tensor&              packed,
                                             std::vector<torch::Tensor>&       valid_lengths,
                                             const CacheBlockTablePackingPlan& plan);

}  // namespace rtp_llm
