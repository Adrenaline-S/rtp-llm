#pragma once

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct BlockBufferPtrInfo {
    torch::Tensor kv_addr;
    torch::Tensor kv_scale_addr;
};

// Dense, immutable all-layer view for one cache group. A group that does not
// own a layer stores an undefined kv_addr at that layer. Scale storage is
// optional even for active layers.
class CacheLayerLayout {
public:
    CacheLayerLayout() = default;

    explicit CacheLayerLayout(std::vector<BlockBufferPtrInfo> layers): layers_(std::move(layers)) {
        for (const auto& layer : layers_) {
            active_layer_count_ += layer.kv_addr.defined() ? 1 : 0;
        }
    }

    bool empty() const noexcept {
        return active_layer_count_ == 0;
    }

    size_t activeLayerCount() const noexcept {
        return active_layer_count_;
    }

    size_t size() const noexcept {
        return layers_.size();
    }

    bool hasLayer(size_t layer_id) const {
        RTP_LLM_CHECK_WITH_INFO(
            layer_id < layers_.size(), "CacheLayerLayout invalid layer_id=%zu size=%zu", layer_id, layers_.size());
        return layers_[layer_id].kv_addr.defined();
    }

    const BlockBufferPtrInfo& at(size_t layer_id) const {
        RTP_LLM_CHECK_WITH_INFO(
            layer_id < layers_.size(), "CacheLayerLayout invalid layer_id=%zu size=%zu", layer_id, layers_.size());
        return layers_[layer_id];
    }

    const std::vector<BlockBufferPtrInfo>& layers() const noexcept {
        return layers_;
    }

private:
    std::vector<BlockBufferPtrInfo> layers_;
    size_t                          active_layer_count_ = 0;
};

// Immutable memory-layout projection built once from a resolved CacheConfig.
class GroupedCacheLayerLayout {
public:
    using GroupLayouts = std::map<std::string, CacheLayerLayout>;

    GroupedCacheLayerLayout() = default;

    GroupedCacheLayerLayout(const CacheConfig& config, GroupLayouts groups): groups_(std::move(groups)) {
        RTP_LLM_CHECK_WITH_INFO(groups_.size() == config.groups().size(),
                                "GroupedCacheLayerLayout group count=%zu config count=%zu",
                                groups_.size(),
                                config.groups().size());
        layer_group_tags_.reserve(config.layerMemberships().size());
        for (const auto& layer : config.layerMemberships()) {
            layer_group_tags_.push_back(layer.group_tags);
        }
        for (const auto& group_config : config.groups()) {
            const auto it = groups_.find(group_config.tag);
            RTP_LLM_CHECK_WITH_INFO(
                it != groups_.end(), "GroupedCacheLayerLayout missing config tag=%s", group_config.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(it->second.size() == layer_group_tags_.size(),
                                    "GroupedCacheLayerLayout tag=%s layer count=%zu config count=%zu",
                                    group_config.tag.c_str(),
                                    it->second.size(),
                                    layer_group_tags_.size());
            group_layouts_.emplace(group_config.tag, group_config.layout);
            group_types_.emplace(group_config.tag, group_config.policy.group_type);
        }
    }

    const CacheLayerLayout& group(std::string_view tag) const {
        const std::string value(tag);
        const auto        it = groups_.find(value);
        RTP_LLM_CHECK_WITH_INFO(it != groups_.end(), "GroupedCacheLayerLayout missing tag=%s", value.c_str());
        return it->second;
    }

    const BlockBufferPtrInfo& at(std::string_view tag, size_t layer_id) const {
        return group(tag).at(layer_id);
    }

    // Layer-only access is valid only when exactly one group has data for the
    // requested layer.
    const BlockBufferPtrInfo& at(size_t layer_id) const {
        const BlockBufferPtrInfo* result = nullptr;
        size_t                    count  = 0;
        for (const auto& [tag, layout] : groups_) {
            (void)tag;
            if (layout.hasLayer(layer_id)) {
                result = &layout.at(layer_id);
                ++count;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(count == 1,
                                "GroupedCacheLayerLayout layer=%zu requires exactly one active group, got %zu",
                                layer_id,
                                count);
        return *result;
    }

    const GroupLayouts& groups() const noexcept {
        return groups_;
    }

    bool hasGroupData(std::string_view tag) const {
        return !group(tag).empty();
    }

    const std::vector<std::string>& groupTagsForLayer(int layer_id) const {
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layer_group_tags_.size(),
                                "GroupedCacheLayerLayout invalid layer=%d size=%zu",
                                layer_id,
                                layer_group_tags_.size());
        return layer_group_tags_[static_cast<size_t>(layer_id)];
    }

    size_t layerCount() const noexcept {
        return layer_group_tags_.size();
    }

    const CacheGroupLayout& groupLayout(std::string_view tag) const {
        const std::string value(tag);
        const auto        it = group_layouts_.find(value);
        RTP_LLM_CHECK_WITH_INFO(it != group_layouts_.end(),
                                "GroupedCacheLayerLayout missing group layout tag=%s",
                                value.c_str());
        return it->second;
    }

    CacheGroupType groupType(std::string_view tag) const {
        const std::string value(tag);
        const auto        it = group_types_.find(value);
        RTP_LLM_CHECK_WITH_INFO(
            it != group_types_.end(), "GroupedCacheLayerLayout missing group type tag=%s", value.c_str());
        return it->second;
    }

private:
    GroupLayouts                            groups_;
    std::map<std::string, CacheGroupLayout> group_layouts_;
    std::map<std::string, CacheGroupType>   group_types_;
    std::vector<std::vector<std::string>>    layer_group_tags_;
};

struct KVCacheBuffer {
    torch::Tensor kv_blocks;
    torch::Tensor kv_scale_blocks;
};

}  // namespace rtp_llm
