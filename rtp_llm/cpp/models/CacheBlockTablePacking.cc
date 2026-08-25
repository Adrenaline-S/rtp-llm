#include "rtp_llm/cpp/models/CacheBlockTablePacking.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "rtp_llm/cpp/cache/CacheGroupTagOrder.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {

constexpr int64_t kPackingPlanWireVersion = 1;

size_t checkedMultiply(size_t lhs, size_t rhs, const char* relation) {
    RTP_LLM_CHECK_WITH_INFO(
        lhs == 0 || rhs <= std::numeric_limits<size_t>::max() / lhs, "%s size overflow: %zu * %zu", relation, lhs, rhs);
    return lhs * rhs;
}

size_t checkedAdd(size_t lhs, size_t rhs, const char* relation) {
    RTP_LLM_CHECK_WITH_INFO(
        rhs <= std::numeric_limits<size_t>::max() - lhs, "%s offset overflow: %zu + %zu", relation, lhs, rhs);
    return lhs + rhs;
}

int64_t checkedWireValue(size_t value, const char* field) {
    RTP_LLM_CHECK_WITH_INFO(value <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                            "cache block-table %s exceeds int64 wire range",
                            field);
    return static_cast<int64_t>(value);
}

size_t checkedSizeValue(int64_t value, const char* field) {
    RTP_LLM_CHECK_WITH_INFO(value >= 0, "cache block-table %s is negative on TP wire", field);
    return static_cast<size_t>(value);
}

}  // namespace

size_t PackedBlockTableRegion::numel() const {
    return checkedMultiply(row_width, batch_capacity, "packed block-table region");
}

CacheBlockTablePackingPlan
CacheBlockTablePackingPlan::create(const CacheConfig& config, size_t batch_capacity, size_t max_pool_blocks) {
    std::vector<std::string> tags;
    tags.reserve(config.groups().size());
    for (const auto& group : config.groups()) {
        tags.push_back(group.tag);
    }
    tags = sortedCacheGroupTags(tags, "cache block-table packing plan");

    std::vector<size_t> pool_row_widths;
    pool_row_widths.reserve(tags.size());
    for (const auto& tag : tags) {
        const auto& group = config.group(tag);
        pool_row_widths.push_back(group.block_num == 0 ? max_pool_blocks :
                                                         std::min<size_t>(max_pool_blocks, group.block_num));
    }
    return create(config, batch_capacity, pool_row_widths);
}

CacheBlockTablePackingPlan CacheBlockTablePackingPlan::create(const CacheConfig&         config,
                                                              size_t                     batch_capacity,
                                                              const std::vector<size_t>& pool_row_widths) {
    std::vector<std::string> tags;
    tags.reserve(config.groups().size());
    for (const auto& group : config.groups()) {
        tags.push_back(group.tag);
    }
    tags = sortedCacheGroupTags(tags, "cache block-table packing plan");
    RTP_LLM_CHECK_WITH_INFO(pool_row_widths.size() == tags.size(),
                            "pool row-width count=%zu does not match cache group count=%zu",
                            pool_row_widths.size(),
                            tags.size());

    std::vector<CacheGroupBlockTableRegion> groups;
    groups.reserve(tags.size());
    size_t pool_offset   = 0;
    size_t kernel_offset = 0;
    for (size_t ordinal = 0; ordinal < tags.size(); ++ordinal) {
        const auto& group = config.group(tags[ordinal]);
        // A row expresses logical positions, not distinct allocations. Pool IDs
        // may therefore repeat (warm-up deliberately does this), so its width is
        // not bounded by the physical pool capacity in layout.block_num.
        const size_t pool_width             = pool_row_widths[ordinal];
        const size_t kernel_blocks_per_pool = group.kernelBlocksPerPoolBlock();
        const size_t kernel_width           = checkedMultiply(pool_width, kernel_blocks_per_pool, "kernel row width");

        CacheGroupBlockTableRegion region;
        region.tag               = tags[ordinal];
        region.execution_ordinal = static_cast<uint32_t>(ordinal);
        region.pool              = {pool_offset, pool_width, batch_capacity};
        region.kernel            = {kernel_offset, kernel_width, batch_capacity};
        pool_offset              = checkedAdd(pool_offset, region.pool.numel(), "pool packed table");
        kernel_offset            = checkedAdd(kernel_offset, region.kernel.numel(), "kernel packed table");
        groups.push_back(std::move(region));
    }
    return createValidated(std::move(groups));
}

CacheBlockTablePackingPlan CacheBlockTablePackingPlan::fromRegions(std::vector<CacheGroupBlockTableRegion> groups) {
    return createValidated(std::move(groups));
}

CacheBlockTablePackingPlan CacheBlockTablePackingPlan::createValidated(std::vector<CacheGroupBlockTableRegion> groups) {
    CacheBlockTablePackingPlan plan;
    plan.groups_ = std::move(groups);
    if (plan.groups_.empty()) {
        return plan;
    }
    plan.batch_capacity_          = plan.groups_.front().pool.batch_capacity;
    size_t expected_pool_offset   = 0;
    size_t expected_kernel_offset = 0;
    for (size_t ordinal = 0; ordinal < plan.groups_.size(); ++ordinal) {
        const auto& group = plan.groups_[ordinal];
        RTP_LLM_CHECK_WITH_INFO(group.execution_ordinal == ordinal,
                                "cache group execution ordinal mismatch: got %u expected %zu",
                                group.execution_ordinal,
                                ordinal);
        RTP_LLM_CHECK_WITH_INFO(group.pool.batch_capacity == plan.batch_capacity_
                                    && group.kernel.batch_capacity == plan.batch_capacity_,
                                "cache group batch capacities must be identical");
        RTP_LLM_CHECK_WITH_INFO(group.pool.offset == expected_pool_offset,
                                "pool regions must be dense: got offset=%zu expected=%zu",
                                group.pool.offset,
                                expected_pool_offset);
        RTP_LLM_CHECK_WITH_INFO(group.kernel.offset == expected_kernel_offset,
                                "kernel regions must be dense: got offset=%zu expected=%zu",
                                group.kernel.offset,
                                expected_kernel_offset);
        expected_pool_offset   = checkedAdd(expected_pool_offset, group.pool.numel(), "pool packed table");
        expected_kernel_offset = checkedAdd(expected_kernel_offset, group.kernel.numel(), "kernel packed table");
    }
    plan.pool_numel_   = expected_pool_offset;
    plan.kernel_numel_ = expected_kernel_offset;
    return plan;
}

const CacheGroupBlockTableRegion& CacheBlockTablePackingPlan::group(uint32_t execution_ordinal) const {
    RTP_LLM_CHECK_WITH_INFO(execution_ordinal < groups_.size(),
                            "cache group execution ordinal %u out of range [0, %zu)",
                            execution_ordinal,
                            groups_.size());
    return groups_[execution_ordinal];
}

size_t CacheBlockTablePackingPlan::groupCount() const noexcept {
    return groups_.size();
}

size_t CacheBlockTablePackingPlan::batchCapacity() const noexcept {
    return batch_capacity_;
}

size_t CacheBlockTablePackingPlan::poolNumel() const noexcept {
    return pool_numel_;
}

size_t CacheBlockTablePackingPlan::kernelNumel() const noexcept {
    return kernel_numel_;
}

std::vector<int64_t> encodeCacheBlockTablePackingPlan(const CacheBlockTablePackingPlan& plan) {
    std::vector<int64_t> wire;
    size_t               wire_size = 2;
    for (uint32_t ordinal = 0; ordinal < plan.groupCount(); ++ordinal) {
        wire_size = checkedAdd(
            wire_size, checkedAdd(8, plan.group(ordinal).tag.size(), "packing-plan wire"), "packing-plan wire");
    }
    wire.reserve(wire_size);
    wire.push_back(kPackingPlanWireVersion);
    wire.push_back(checkedWireValue(plan.groupCount(), "group count"));
    for (uint32_t ordinal = 0; ordinal < plan.groupCount(); ++ordinal) {
        const auto& group = plan.group(ordinal);
        wire.push_back(group.execution_ordinal);
        wire.push_back(checkedWireValue(group.pool.offset, "pool offset"));
        wire.push_back(checkedWireValue(group.pool.row_width, "pool row width"));
        wire.push_back(checkedWireValue(group.pool.batch_capacity, "pool batch capacity"));
        wire.push_back(checkedWireValue(group.kernel.offset, "kernel offset"));
        wire.push_back(checkedWireValue(group.kernel.row_width, "kernel row width"));
        wire.push_back(checkedWireValue(group.kernel.batch_capacity, "kernel batch capacity"));
        wire.push_back(checkedWireValue(group.tag.size(), "tag length"));
        for (const unsigned char byte : group.tag) {
            wire.push_back(byte);
        }
    }
    return wire;
}

CacheBlockTablePackingPlan decodeCacheBlockTablePackingPlan(const int64_t* wire, size_t wire_size) {
    RTP_LLM_CHECK_WITH_INFO(wire != nullptr && wire_size >= 2, "cache block-table TP wire is truncated");
    RTP_LLM_CHECK_WITH_INFO(wire[0] == kPackingPlanWireVersion,
                            "unsupported cache block-table TP wire version %lld",
                            static_cast<long long>(wire[0]));
    const size_t                            group_count = checkedSizeValue(wire[1], "group count");
    std::vector<CacheGroupBlockTableRegion> groups;
    groups.reserve(group_count);
    size_t cursor = 2;
    for (size_t ordinal = 0; ordinal < group_count; ++ordinal) {
        RTP_LLM_CHECK_WITH_INFO(cursor <= wire_size && wire_size - cursor >= 8,
                                "cache block-table TP wire is truncated at group %zu",
                                ordinal);
        CacheGroupBlockTableRegion group;
        group.execution_ordinal     = static_cast<uint32_t>(checkedSizeValue(wire[cursor++], "execution ordinal"));
        group.pool.offset           = checkedSizeValue(wire[cursor++], "pool offset");
        group.pool.row_width        = checkedSizeValue(wire[cursor++], "pool row width");
        group.pool.batch_capacity   = checkedSizeValue(wire[cursor++], "pool batch capacity");
        group.kernel.offset         = checkedSizeValue(wire[cursor++], "kernel offset");
        group.kernel.row_width      = checkedSizeValue(wire[cursor++], "kernel row width");
        group.kernel.batch_capacity = checkedSizeValue(wire[cursor++], "kernel batch capacity");
        const size_t tag_size       = checkedSizeValue(wire[cursor++], "tag length");
        RTP_LLM_CHECK_WITH_INFO(
            tag_size <= wire_size - cursor, "cache block-table TP wire tag is truncated at group %zu", ordinal);
        group.tag.reserve(tag_size);
        for (size_t index = 0; index < tag_size; ++index) {
            RTP_LLM_CHECK_WITH_INFO(wire[cursor] >= 0 && wire[cursor] <= std::numeric_limits<unsigned char>::max(),
                                    "cache block-table TP wire tag byte is invalid at group %zu",
                                    ordinal);
            group.tag.push_back(static_cast<char>(wire[cursor++]));
        }
        groups.push_back(std::move(group));
    }
    RTP_LLM_CHECK_WITH_INFO(
        cursor == wire_size, "cache block-table TP wire has %zu trailing values", wire_size - cursor);
    return CacheBlockTablePackingPlan::fromRegions(std::move(groups));
}

CacheBlockTablePackingPlan decodeCacheBlockTablePackingPlan(const std::vector<int64_t>& wire) {
    return decodeCacheBlockTablePackingPlan(wire.data(), wire.size());
}

torch::Tensor CacheBlockTablePackingPlan::view(const torch::Tensor&          backing,
                                               const PackedBlockTableRegion& region,
                                               const char*                   name) const {
    RTP_LLM_CHECK_WITH_INFO(backing.defined(), "%s packed backing must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(backing.dim() == 1, "%s packed backing must be flat", name);
    RTP_LLM_CHECK_WITH_INFO(
        backing.scalar_type() == torch::kInt32, "%s packed backing must use int32 block identities", name);
    RTP_LLM_CHECK_WITH_INFO(backing.is_contiguous(), "%s packed backing must be contiguous", name);
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(backing.numel()) >= checkedAdd(region.offset, region.numel(), name),
                            "%s packed backing is too small",
                            name);
    return backing.narrow(0, static_cast<int64_t>(region.offset), static_cast<int64_t>(region.numel()))
        .view({static_cast<int64_t>(region.batch_capacity), static_cast<int64_t>(region.row_width)});
}

torch::Tensor CacheBlockTablePackingPlan::poolView(const torch::Tensor& backing, uint32_t execution_ordinal) const {
    return view(backing, group(execution_ordinal).pool, "pool");
}

torch::Tensor CacheBlockTablePackingPlan::kernelView(const torch::Tensor& backing, uint32_t execution_ordinal) const {
    return view(backing, group(execution_ordinal).kernel, "kernel");
}

torch::Tensor packKernelValidLengthsForTp(const std::vector<torch::Tensor>& valid_lengths,
                                          const CacheBlockTablePackingPlan& plan) {
    RTP_LLM_CHECK_WITH_INFO(valid_lengths.size() == plan.groupCount(),
                            "kernel valid-length group count=%zu does not match packing plan=%zu",
                            valid_lengths.size(),
                            plan.groupCount());
    const auto batch_capacity = static_cast<int64_t>(plan.batchCapacity());
    auto       packed         = torch::empty({static_cast<int64_t>(plan.groupCount()), batch_capacity},
                               torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    for (uint32_t ordinal = 0; ordinal < plan.groupCount(); ++ordinal) {
        const auto& lengths = valid_lengths[ordinal];
        RTP_LLM_CHECK_WITH_INFO(lengths.defined() && lengths.scalar_type() == torch::kInt32 && !lengths.is_cuda()
                                    && lengths.is_contiguous() && lengths.numel() == batch_capacity,
                                "kernel valid lengths for execution ordinal %u must be contiguous host int32 [%ld]",
                                ordinal,
                                batch_capacity);
        packed[ordinal].copy_(lengths, /*non_blocking=*/false);
    }
    return packed;
}

void unpackKernelValidLengthsFromTp(const torch::Tensor&              packed,
                                    std::vector<torch::Tensor>&       valid_lengths,
                                    const CacheBlockTablePackingPlan& plan) {
    RTP_LLM_CHECK_WITH_INFO(packed.defined() && packed.scalar_type() == torch::kInt32 && !packed.is_cuda()
                                && packed.is_contiguous() && packed.dim() == 2
                                && static_cast<size_t>(packed.size(0)) == plan.groupCount()
                                && static_cast<size_t>(packed.size(1)) == plan.batchCapacity(),
                            "packed TP kernel valid lengths must have shape [%zu, %zu]",
                            plan.groupCount(),
                            plan.batchCapacity());
    RTP_LLM_CHECK_WITH_INFO(valid_lengths.size() == plan.groupCount(),
                            "destination kernel valid-length group count=%zu does not match packing plan=%zu",
                            valid_lengths.size(),
                            plan.groupCount());
    for (uint32_t ordinal = 0; ordinal < plan.groupCount(); ++ordinal) {
        auto& lengths = valid_lengths[ordinal];
        RTP_LLM_CHECK_WITH_INFO(lengths.defined() && lengths.scalar_type() == torch::kInt32 && !lengths.is_cuda()
                                    && lengths.is_contiguous()
                                    && static_cast<size_t>(lengths.numel()) == plan.batchCapacity(),
                                "destination kernel valid lengths for execution ordinal %u have invalid geometry",
                                ordinal);
        lengths.copy_(packed[ordinal], /*non_blocking=*/false);
    }
}

}  // namespace rtp_llm
