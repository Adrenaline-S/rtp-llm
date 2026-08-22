#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/BlockExpression.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct CacheConfig;

struct BlockDependency {
    // Dependency metadata belongs to the request's global cache-key timeline. Filtered resource views preserve the
    // original ordinal and may retain a parent_key that is absent from the view so prefix-tree caches can attach it
    // when the parent becomes available.
    bool         has_parent{false};
    CacheKeyType parent_key{0};
    uint32_t     ordinal{0};
};

using BlockDependenciesType = std::vector<BlockDependency>;

class KVCacheResource {
public:
    void initGroups(const CacheConfig& config);
    void resizeBlocks(int reserved_blocks, std::optional<PoolBlockId> initial_block = std::nullopt);

    int blocksNum(std::string_view tag) const;

    GroupBlockToPoolBlockBinding& mutableBlockBinding(std::string_view tag) const;
    GroupBlockToPoolBlockBinding& mutableBlockBindingForLayer(int layer_id, std::string_view tag) const;

    const GroupBlockToPoolBlockBinding& blockBinding(std::string_view tag) const;
    const GroupBlockToPoolBlockBinding& blockBindingForLayer(int layer_id, std::string_view tag) const;

    const std::vector<std::string>& groupTagsForLayer(int layer_id) const;
    const std::string&              soleGroupTagForLayer(int layer_id) const;

    int layerNum() const;
    int groupNums() const;

    const std::map<std::string, GroupBlockToPoolBlockBinding>& blocksByTag() const;

    bool layerOwnsTag(int layer_id, std::string_view tag) const;

    const CacheKeysType& cacheKeys() const;
    void                 setCacheKeysAndBlockDependencies(CacheKeysType keys, BlockDependenciesType dependencies);
    void                 setCacheKeys(CacheKeysType keys);
    bool                 cacheKeysAreCpCanonical() const;
    void                 setCacheKeysAreCpCanonical(bool cache_keys_are_cp_canonical);
    void                 appendCacheKey(CacheKeyType key);
    void                 popBackCacheKey();
    void                 clearCacheKeys();

    const BlockDependenciesType& blockDependencies() const;

    // Return rank-local cache keys: every cp_size-th key starting from cp_rank.
    // localCacheKeys(r, s)[i] == cacheKeys()[i * s + r]
    // Note: when cacheKeys().size() % cp_size != 0 (e.g. 1 real block, cp_size=2),
    // localCacheKeys may return fewer entries than blockBinding().size().  This is
    // intentional — padding blocks carry no real data and must NOT participate in
    // device cache insert, PD transfer, or connector operations.  Downstream code
    // (e.g. insertIntoCache) already uses min(keys, blocks) to handle this.
    CacheKeysType localCacheKeys(int cp_rank, int cp_size) const {
        CacheKeysType local;
        for (int i = cp_rank; i < static_cast<int>(cache_keys.size()); i += cp_size) {
            local.push_back(cache_keys[i]);
        }
        return local;
    }

    size_t reuseBlockNum() const;

    size_t deviceReuseBlockNum() const;
    void   setDeviceReuseBlockNum(size_t device_reuse_blocks_num);

    size_t memoryReuseBlockNum() const;
    void   setMemoryReuseBlockNum(size_t memory_reuse_blocks_num);

    size_t remoteReuseBlockNum() const;
    void   setRemoteReuseBlockNum(size_t remote_reuse_blocks_num);

    bool lastBlockAligned() const;
    void setLastBlockAligned(bool last_block_aligned);

    size_t remoteReuseBlocksNum() const;
    void   setRemoteReuseBlocksNum(size_t remote_reuse_blocks_num);

    void swapBlocks(std::string_view tag, size_t rhs, size_t lhs);

    std::string debugString() const;

private:
    bool layerContainsTag(int layer_id, std::string_view tag) const;

    std::vector<std::vector<std::string>>                       layer_group_tags_;
    mutable std::map<std::string, GroupBlockToPoolBlockBinding> blocks_by_tag_;
    CacheKeysType                                               cache_keys;
    BlockDependenciesType                                       block_dependencies;
    bool                                                        cache_keys_are_cp_canonical_{false};

    size_t device_reuse_block_num_{0};
    size_t memory_reuse_block_num_{0};
    size_t remote_reuse_block_num_{0};
    bool   last_block_aligned_{false};
};

using KVCacheResourcePtr = std::shared_ptr<KVCacheResource>;

}  // namespace rtp_llm
