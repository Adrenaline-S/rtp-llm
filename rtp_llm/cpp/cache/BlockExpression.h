#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace rtp_llm {

using CacheKeyType = int64_t;
using BlockIdxType = int32_t;

constexpr BlockIdxType NULL_BLOCK_IDX = static_cast<BlockIdxType>(-1);

inline bool isNullBlockIdx(BlockIdxType block_idx) {
    return block_idx == NULL_BLOCK_IDX;
}

using CacheKeysType    = std::vector<CacheKeyType>;
using BlockIndicesType = std::vector<BlockIdxType>;

struct CacheKeyPosition {
    size_t value;
};

struct GroupBlockPosition {
    size_t value;
};

struct PoolBlockId {
    BlockIdxType value;
};

struct CacheKeyRange {
    CacheKeyPosition begin;
    CacheKeyPosition end;
};

constexpr bool operator==(CacheKeyPosition lhs, CacheKeyPosition rhs) noexcept {
    return lhs.value == rhs.value;
}

constexpr bool operator==(GroupBlockPosition lhs, GroupBlockPosition rhs) noexcept {
    return lhs.value == rhs.value;
}

constexpr bool operator==(PoolBlockId lhs, PoolBlockId rhs) noexcept {
    return lhs.value == rhs.value;
}

constexpr bool operator==(CacheKeyRange lhs, CacheKeyRange rhs) noexcept {
    return lhs.begin == rhs.begin && lhs.end == rhs.end;
}

class CacheKeyToGroupBlockMapping {
public:
    CacheKeyToGroupBlockMapping(size_t cache_key_tokens, size_t group_block_tokens);

    GroupBlockPosition toGroupBlock(CacheKeyPosition key) const;
    CacheKeyRange      toCacheKeyRange(GroupBlockPosition block, size_t key_count) const;
    CacheKeyPosition   toCacheKeyAnchor(GroupBlockPosition block, size_t key_count) const;
    size_t             completeGroupBlockCount(size_t key_count) const noexcept;
    size_t             toCacheKeyPrefixLength(size_t group_block_count) const;

private:
    size_t cache_keys_per_group_block_;
};

using GroupBlockToCacheKeyRange  = CacheKeyToGroupBlockMapping;
using GroupBlockToCacheKeyAnchor = CacheKeyToGroupBlockMapping;

class PoolBlockToKernelBlockProjection {
public:
    explicit PoolBlockToKernelBlockProjection(size_t kernel_blocks_per_pool_block);
    PoolBlockToKernelBlockProjection(size_t pool_block_tokens, size_t kernel_block_tokens);

    size_t projectedSize(size_t pool_block_count) const;
    void   append(PoolBlockId source, BlockIndicesType& destination) const;
    void   project(const std::vector<PoolBlockId>& source, BlockIndicesType& destination) const;

private:
    static size_t resolveKernelBlocksPerPoolBlock(size_t pool_block_tokens, size_t kernel_block_tokens);
    BlockIdxType  projectedBase(PoolBlockId source) const;

    size_t kernel_blocks_per_pool_block_;
};

class GroupBlockToPoolBlockBinding {
public:
    using Snapshot = std::vector<std::optional<PoolBlockId>>;

    void                       resize(size_t group_block_count);
    void                       bind(GroupBlockPosition position, PoolBlockId block_id);
    void                       unbind(GroupBlockPosition position);
    std::optional<PoolBlockId> lookup(GroupBlockPosition position) const;
    size_t                     size() const noexcept;

    void                       append(const Snapshot& block_ids);
    void                       append(const std::vector<PoolBlockId>& block_ids);
    void                       append(PoolBlockId block_id);
    std::optional<PoolBlockId> popBack();
    void                       remove(const std::vector<GroupBlockPosition>& positions);
    void                       swap(GroupBlockPosition lhs, GroupBlockPosition rhs);
    void                       assign(const Snapshot& block_ids);
    void                       assign(const std::vector<PoolBlockId>& block_ids);
    Snapshot                   snapshot() const;

private:
    Snapshot pool_block_ids_;
};

}  // namespace rtp_llm
