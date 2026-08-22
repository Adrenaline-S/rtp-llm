#include "rtp_llm/cpp/cache/KVCacheResource.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

void KVCacheResource::initGroups(const CacheConfig& config) {

    layer_group_tags_.clear();
    blocks_by_tag_.clear();

    const auto& groups = config.groups();

    for (const auto& group : groups) {
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "KVCacheResource requires a non-empty cache group tag");

        RTP_LLM_CHECK_WITH_INFO(blocks_by_tag_.emplace(group.tag, GroupBlockToPoolBlockBinding{}).second,
                                "KVCacheResource has duplicate tag=%s",
                                group.tag.c_str());
    }

    const auto& layers = config.layerMemberships();
    layer_group_tags_.reserve(layers.size());
    for (const auto& layer : layers) {
        for (const auto& tag : layer.group_tags) {
            config.groupForLayer(layer.layer_id, tag);
        }
        layer_group_tags_.push_back(layer.group_tags);
    }
}

void KVCacheResource::resizeBlocks(int reserved_blocks, std::optional<PoolBlockId> initial_block) {
    RTP_LLM_CHECK_WITH_INFO(reserved_blocks >= 0, "reserved_blocks must be nonnegative, got %d", reserved_blocks);
    for (auto& [tag, binding] : blocks_by_tag_) {
        (void)tag;
        const size_t old_size = binding.size();
        binding.resize(static_cast<size_t>(reserved_blocks));
        if (initial_block.has_value()) {
            for (size_t group_block_position = old_size; group_block_position < binding.size();
                 ++group_block_position) {
                binding.bind(GroupBlockPosition{group_block_position}, *initial_block);
            }
        }
    }
}

int KVCacheResource::blocksNum(std::string_view tag) const {
    return static_cast<int>(blockBinding(tag).size());
}

GroupBlockToPoolBlockBinding& KVCacheResource::mutableBlockBinding(std::string_view tag) const {
    const auto value = std::string(tag);
    const auto it    = blocks_by_tag_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != blocks_by_tag_.end(), "KVCacheResource missing tag=%s", value.c_str());
    return it->second;
}

GroupBlockToPoolBlockBinding& KVCacheResource::mutableBlockBindingForLayer(int layer_id, std::string_view tag) const {
    RTP_LLM_CHECK_WITH_INFO(layerContainsTag(layer_id, tag),
                            "KVCacheResource layer=%d does not own tag=%s",
                            layer_id,
                            std::string(tag).c_str());
    return mutableBlockBinding(tag);
}

const GroupBlockToPoolBlockBinding& KVCacheResource::blockBinding(std::string_view tag) const {
    return mutableBlockBinding(tag);
}

const GroupBlockToPoolBlockBinding& KVCacheResource::blockBindingForLayer(int layer_id, std::string_view tag) const {
    return mutableBlockBindingForLayer(layer_id, tag);
}

bool KVCacheResource::layerContainsTag(int layer_id, std::string_view tag) const {
    const auto& tags  = groupTagsForLayer(layer_id);
    const auto  value = std::string(tag);
    return std::find(tags.begin(), tags.end(), value) != tags.end();
}

const std::vector<std::string>& KVCacheResource::groupTagsForLayer(int layer_id) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layer_group_tags_.size(),
                            "KVCacheResource invalid layer_id=%d size=%zu",
                            layer_id,
                            layer_group_tags_.size());
    return layer_group_tags_[static_cast<size_t>(layer_id)];
}

const std::string& KVCacheResource::soleGroupTagForLayer(int layer_id) const {
    const auto& tags = groupTagsForLayer(layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        tags.size() == 1, "KVCacheResource layer=%d requires exactly one group, got %zu", layer_id, tags.size());
    return tags.front();
}

int KVCacheResource::layerNum() const {
    return static_cast<int>(layer_group_tags_.size());
}

int KVCacheResource::groupNums() const {
    return static_cast<int>(blocks_by_tag_.size());
}

const std::map<std::string, GroupBlockToPoolBlockBinding>& KVCacheResource::blocksByTag() const {
    return blocks_by_tag_;
}

bool KVCacheResource::layerOwnsTag(int layer_id, std::string_view tag) const {
    if (tag.empty() || blocks_by_tag_.find(std::string(tag)) == blocks_by_tag_.end()) {
        return false;
    }
    return layerContainsTag(layer_id, tag);
}

const CacheKeysType& KVCacheResource::cacheKeys() const {
    return cache_keys;
}

void KVCacheResource::setCacheKeysAndBlockDependencies(CacheKeysType keys, BlockDependenciesType dependencies) {
    RTP_LLM_CHECK_WITH_INFO(keys.size() == dependencies.size(),
                            "cache timeline size mismatch: keys=%zu dependencies=%zu",
                            keys.size(),
                            dependencies.size());
    cache_keys                   = std::move(keys);
    block_dependencies           = std::move(dependencies);
    cache_keys_are_cp_canonical_ = false;
}

void KVCacheResource::setCacheKeys(CacheKeysType keys) {
    BlockDependenciesType dependencies;
    dependencies.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        BlockDependency dependency;
        dependency.ordinal = static_cast<uint32_t>(i);
        if (i > 0) {
            dependency.has_parent = true;
            dependency.parent_key = keys[i - 1];
        }
        dependencies.push_back(dependency);
    }
    setCacheKeysAndBlockDependencies(std::move(keys), std::move(dependencies));
}

bool KVCacheResource::cacheKeysAreCpCanonical() const {
    return cache_keys_are_cp_canonical_;
}

void KVCacheResource::setCacheKeysAreCpCanonical(bool cache_keys_are_cp_canonical) {
    cache_keys_are_cp_canonical_ = cache_keys_are_cp_canonical;
}

void KVCacheResource::appendCacheKey(CacheKeyType key) {
    RTP_LLM_CHECK_WITH_INFO(block_dependencies.size() == cache_keys.size(),
                            "cache key/dependency timeline diverged before append: keys=%zu dependencies=%zu",
                            cache_keys.size(),
                            block_dependencies.size());
    BlockDependency dependency;
    dependency.ordinal = static_cast<uint32_t>(cache_keys.size());
    if (!cache_keys.empty()) {
        dependency.has_parent = true;
        dependency.parent_key = cache_keys.back();
    }
    const size_t new_size = cache_keys.size() + 1;
    if (cache_keys.capacity() < new_size || block_dependencies.capacity() < new_size) {
        CacheKeysType         new_keys         = cache_keys;
        BlockDependenciesType new_dependencies = block_dependencies;
        new_keys.push_back(key);
        new_dependencies.push_back(dependency);
        cache_keys.swap(new_keys);
        block_dependencies.swap(new_dependencies);
        return;
    }
    cache_keys.push_back(key);
    block_dependencies.push_back(dependency);
}

void KVCacheResource::popBackCacheKey() {
    if (cache_keys.empty()) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(block_dependencies.size() == cache_keys.size(),
                            "cache key/dependency timeline diverged before pop: keys=%zu dependencies=%zu",
                            cache_keys.size(),
                            block_dependencies.size());
    cache_keys.pop_back();
    block_dependencies.pop_back();
}

void KVCacheResource::clearCacheKeys() {
    cache_keys.clear();
    block_dependencies.clear();
}

const BlockDependenciesType& KVCacheResource::blockDependencies() const {
    return block_dependencies;
}

size_t KVCacheResource::reuseBlockNum() const {
    return device_reuse_block_num_ + memory_reuse_block_num_ + remote_reuse_block_num_;
}

size_t KVCacheResource::deviceReuseBlockNum() const {
    return device_reuse_block_num_;
}

void KVCacheResource::setDeviceReuseBlockNum(size_t device_reuse_blocks_num) {
    device_reuse_block_num_ = device_reuse_blocks_num;
}

size_t KVCacheResource::memoryReuseBlockNum() const {
    return memory_reuse_block_num_;
}

void KVCacheResource::setMemoryReuseBlockNum(size_t memory_reuse_blocks_num) {
    memory_reuse_block_num_ = memory_reuse_blocks_num;
}

size_t KVCacheResource::remoteReuseBlockNum() const {
    return remote_reuse_block_num_;
}

void KVCacheResource::setRemoteReuseBlockNum(size_t remote_reuse_blocks_num) {
    remote_reuse_block_num_ = remote_reuse_blocks_num;
}

bool KVCacheResource::lastBlockAligned() const {
    return last_block_aligned_;
}

void KVCacheResource::setLastBlockAligned(bool last_block_aligned) {
    last_block_aligned_ = last_block_aligned;
}

std::string KVCacheResource::debugString() const {
    std::stringstream debug_string;
    for (const auto& [tag, binding] : blocks_by_tag_) {
        debug_string << "group:[" << tag << "], block:[";
        for (const auto& block : binding.snapshot()) {
            if (block.has_value()) {
                debug_string << block->value << ", ";
            } else {
                debug_string << "missing, ";
            }
        }
        debug_string << "], ";
    }

    return debug_string.str();
}

void KVCacheResource::swapBlocks(std::string_view tag, size_t rhs, size_t lhs) {
    mutableBlockBinding(tag).swap(GroupBlockPosition{rhs}, GroupBlockPosition{lhs});
}

}  // namespace rtp_llm
