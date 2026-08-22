#include "rtp_llm/cpp/cache/BlockExpression.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

CacheKeyToGroupBlockMapping::CacheKeyToGroupBlockMapping(size_t cache_key_tokens, size_t group_block_tokens) {
    RTP_LLM_CHECK_WITH_INFO(cache_key_tokens > 0, "cache_key_tokens must be greater than zero");
    RTP_LLM_CHECK_WITH_INFO(group_block_tokens > 0, "group_block_tokens must be greater than zero");
    RTP_LLM_CHECK_WITH_INFO(group_block_tokens % cache_key_tokens == 0,
                            "group_block_tokens(%zu) must be divisible by cache_key_tokens(%zu)",
                            group_block_tokens,
                            cache_key_tokens);
    cache_keys_per_group_block_ = group_block_tokens / cache_key_tokens;
}

GroupBlockPosition CacheKeyToGroupBlockMapping::toGroupBlock(CacheKeyPosition key) const {
    return GroupBlockPosition{key.value / cache_keys_per_group_block_};
}

CacheKeyRange CacheKeyToGroupBlockMapping::toCacheKeyRange(GroupBlockPosition block, size_t key_count) const {
    RTP_LLM_CHECK_WITH_INFO(block.value <= std::numeric_limits<size_t>::max() / cache_keys_per_group_block_,
                            "group block position overflow: position=%zu ratio=%zu",
                            block.value,
                            cache_keys_per_group_block_);
    const size_t raw_begin = block.value * cache_keys_per_group_block_;
    const size_t begin     = std::min(raw_begin, key_count);
    const size_t remaining = key_count - begin;
    const size_t width     = std::min(cache_keys_per_group_block_, remaining);
    return CacheKeyRange{CacheKeyPosition{begin}, CacheKeyPosition{begin + width}};
}

CacheKeyPosition CacheKeyToGroupBlockMapping::toCacheKeyAnchor(GroupBlockPosition block, size_t key_count) const {
    const auto range = toCacheKeyRange(block, key_count);
    RTP_LLM_CHECK_WITH_INFO(range.begin.value < range.end.value,
                            "group block position %zu has no cache-key anchor for key_count=%zu",
                            block.value,
                            key_count);
    return CacheKeyPosition{range.end.value - 1};
}

size_t CacheKeyToGroupBlockMapping::completeGroupBlockCount(size_t key_count) const noexcept {
    return key_count / cache_keys_per_group_block_;
}

size_t CacheKeyToGroupBlockMapping::toCacheKeyPrefixLength(size_t group_block_count) const {
    RTP_LLM_CHECK_WITH_INFO(group_block_count <= std::numeric_limits<size_t>::max() / cache_keys_per_group_block_,
                            "cache-key prefix length overflow: group_blocks=%zu ratio=%zu",
                            group_block_count,
                            cache_keys_per_group_block_);
    return group_block_count * cache_keys_per_group_block_;
}

size_t PoolBlockToKernelBlockProjection::resolveKernelBlocksPerPoolBlock(size_t pool_block_tokens,
                                                                         size_t kernel_block_tokens) {
    RTP_LLM_CHECK_WITH_INFO(pool_block_tokens > 0, "pool_block_tokens must be greater than zero");
    RTP_LLM_CHECK_WITH_INFO(kernel_block_tokens > 0, "kernel_block_tokens must be greater than zero");
    RTP_LLM_CHECK_WITH_INFO(pool_block_tokens % kernel_block_tokens == 0,
                            "pool_block_tokens(%zu) must be divisible by kernel_block_tokens(%zu)",
                            pool_block_tokens,
                            kernel_block_tokens);
    return pool_block_tokens / kernel_block_tokens;
}

PoolBlockToKernelBlockProjection::PoolBlockToKernelBlockProjection(size_t kernel_blocks_per_pool_block):
    kernel_blocks_per_pool_block_(kernel_blocks_per_pool_block) {
    RTP_LLM_CHECK_WITH_INFO(kernel_blocks_per_pool_block_ > 0,
                            "kernel_blocks_per_pool_block must be greater than zero");
}

PoolBlockToKernelBlockProjection::PoolBlockToKernelBlockProjection(size_t pool_block_tokens,
                                                                   size_t kernel_block_tokens):
    PoolBlockToKernelBlockProjection(resolveKernelBlocksPerPoolBlock(pool_block_tokens, kernel_block_tokens)) {}

size_t PoolBlockToKernelBlockProjection::projectedSize(size_t pool_block_count) const {
    RTP_LLM_CHECK_WITH_INFO(pool_block_count <= std::numeric_limits<size_t>::max() / kernel_blocks_per_pool_block_,
                            "kernel projection size overflow: pool_blocks=%zu factor=%zu",
                            pool_block_count,
                            kernel_blocks_per_pool_block_);
    return pool_block_count * kernel_blocks_per_pool_block_;
}

BlockIdxType PoolBlockToKernelBlockProjection::projectedBase(PoolBlockId source) const {
    constexpr auto max_block_id = static_cast<uint64_t>(std::numeric_limits<BlockIdxType>::max());
    RTP_LLM_CHECK_WITH_INFO(source.value >= 0, "PoolBlockId must be nonnegative, got %d", source.value);
    RTP_LLM_CHECK_WITH_INFO(kernel_blocks_per_pool_block_ - 1 <= max_block_id,
                            "kernel block id overflow: pool_block_id=%d factor=%zu",
                            source.value,
                            kernel_blocks_per_pool_block_);
    const uint64_t last_offset = static_cast<uint64_t>(kernel_blocks_per_pool_block_ - 1);
    const uint64_t factor      = static_cast<uint64_t>(kernel_blocks_per_pool_block_);
    RTP_LLM_CHECK_WITH_INFO(static_cast<uint64_t>(source.value) <= (max_block_id - last_offset) / factor,
                            "kernel block id overflow: pool_block_id=%d factor=%zu",
                            source.value,
                            kernel_blocks_per_pool_block_);
    return static_cast<BlockIdxType>(static_cast<uint64_t>(source.value) * factor);
}

void PoolBlockToKernelBlockProjection::append(PoolBlockId source, BlockIndicesType& destination) const {
    const auto base = projectedBase(source);
    for (size_t kernel_offset = 0; kernel_offset < kernel_blocks_per_pool_block_; ++kernel_offset) {
        destination.push_back(base + static_cast<BlockIdxType>(kernel_offset));
    }
}

void PoolBlockToKernelBlockProjection::project(const std::vector<PoolBlockId>& source,
                                               BlockIndicesType&               destination) const {
    destination.clear();
    for (const auto block_id : source) {
        (void)projectedBase(block_id);
    }
    destination.reserve(projectedSize(source.size()));
    for (const auto block_id : source) {
        append(block_id, destination);
    }
}

void GroupBlockToPoolBlockBinding::resize(size_t group_block_count) {
    pool_block_ids_.resize(group_block_count);
}

void GroupBlockToPoolBlockBinding::bind(GroupBlockPosition position, PoolBlockId block_id) {
    RTP_LLM_CHECK_WITH_INFO(position.value < pool_block_ids_.size(),
                            "group block position %zu out of range, size=%zu",
                            position.value,
                            pool_block_ids_.size());
    RTP_LLM_CHECK_WITH_INFO(block_id.value >= 0, "PoolBlockId must be nonnegative, got %d", block_id.value);
    pool_block_ids_[position.value] = block_id;
}

void GroupBlockToPoolBlockBinding::unbind(GroupBlockPosition position) {
    RTP_LLM_CHECK_WITH_INFO(position.value < pool_block_ids_.size(),
                            "group block position %zu out of range, size=%zu",
                            position.value,
                            pool_block_ids_.size());
    pool_block_ids_[position.value] = std::nullopt;
}

std::optional<PoolBlockId> GroupBlockToPoolBlockBinding::lookup(GroupBlockPosition position) const {
    RTP_LLM_CHECK_WITH_INFO(position.value < pool_block_ids_.size(),
                            "group block position %zu out of range, size=%zu",
                            position.value,
                            pool_block_ids_.size());
    return pool_block_ids_[position.value];
}

size_t GroupBlockToPoolBlockBinding::size() const noexcept {
    return pool_block_ids_.size();
}

void GroupBlockToPoolBlockBinding::append(const Snapshot& block_ids) {
    pool_block_ids_.reserve(pool_block_ids_.size() + block_ids.size());
    for (const auto block_id : block_ids) {
        if (block_id.has_value()) {
            RTP_LLM_CHECK_WITH_INFO(block_id->value >= 0, "PoolBlockId must be nonnegative, got %d", block_id->value);
        }
        pool_block_ids_.push_back(block_id);
    }
}

void GroupBlockToPoolBlockBinding::append(const std::vector<PoolBlockId>& block_ids) {
    pool_block_ids_.reserve(pool_block_ids_.size() + block_ids.size());
    for (const auto block_id : block_ids) {
        append(block_id);
    }
}

void GroupBlockToPoolBlockBinding::append(PoolBlockId block_id) {
    RTP_LLM_CHECK_WITH_INFO(block_id.value >= 0, "PoolBlockId must be nonnegative, got %d", block_id.value);
    pool_block_ids_.push_back(block_id);
}

std::optional<PoolBlockId> GroupBlockToPoolBlockBinding::popBack() {
    RTP_LLM_CHECK(!pool_block_ids_.empty());
    const auto block_id = pool_block_ids_.back();
    pool_block_ids_.pop_back();
    return block_id;
}

void GroupBlockToPoolBlockBinding::remove(const std::vector<GroupBlockPosition>& positions) {
    for (const auto position : positions) {
        unbind(position);
    }
}

void GroupBlockToPoolBlockBinding::swap(GroupBlockPosition lhs, GroupBlockPosition rhs) {
    RTP_LLM_CHECK_WITH_INFO(lhs.value < pool_block_ids_.size() && rhs.value < pool_block_ids_.size(),
                            "group block swap out of range: lhs=%zu rhs=%zu size=%zu",
                            lhs.value,
                            rhs.value,
                            pool_block_ids_.size());
    std::swap(pool_block_ids_[lhs.value], pool_block_ids_[rhs.value]);
}

void GroupBlockToPoolBlockBinding::assign(const Snapshot& block_ids) {
    pool_block_ids_.clear();
    append(block_ids);
}

void GroupBlockToPoolBlockBinding::assign(const std::vector<PoolBlockId>& block_ids) {
    pool_block_ids_.clear();
    append(block_ids);
}

GroupBlockToPoolBlockBinding::Snapshot GroupBlockToPoolBlockBinding::snapshot() const {
    return pool_block_ids_;
}

}  // namespace rtp_llm
