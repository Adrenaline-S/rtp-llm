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
        layer_group_tags_.reserve(config.layers().size());
        for (const auto& layer : config.layers()) {
            layer_group_tags_.push_back(layer);
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
            group_configs_.emplace(group_config.tag, group_config);
            group_types_.emplace(group_config.tag, group_config.policy.group_type);
        }
    }

    // Builds a local-layer execution view over buffers owned by a sealed root
    // configuration. The target config supplies local group membership while
    // global_layer_ids selects the corresponding backing buffers.
    static GroupedCacheLayerLayout project(const GroupedCacheLayerLayout& source,
                                           const CacheConfig&             target_config,
                                           const std::vector<size_t>&     global_layer_ids) {
        RTP_LLM_CHECK_WITH_INFO(target_config.mainLayerCount() == global_layer_ids.size(),
                                "cache layout projection config layers=%u mapping size=%zu",
                                target_config.mainLayerCount(),
                                global_layer_ids.size());

        GroupedCacheLayerLayout result;
        result.layer_group_tags_.reserve(global_layer_ids.size());
        for (size_t local = 0; local < global_layer_ids.size(); ++local) {
            result.layer_group_tags_.push_back(target_config.groupTagsForLayer(static_cast<int>(local)));
        }

        for (const auto& target_group : target_config.groups()) {
            CacheGroup local_group = target_group;
            local_group.layer_ids.clear();
            std::vector<BlockBufferPtrInfo> layers(global_layer_ids.size());
            const auto&                     source_group = source.group(target_group.tag);
            for (size_t local = 0; local < global_layer_ids.size(); ++local) {
                const auto& tags = result.layer_group_tags_[local];
                if (std::find(tags.begin(), tags.end(), target_group.tag) == tags.end()) {
                    continue;
                }
                local_group.layer_ids.push_back(static_cast<int>(local));
                const auto global = global_layer_ids[local];
                if (source_group.hasLayer(global)) {
                    layers[local] = source_group.at(global);
                }
            }
            result.groups_.emplace(target_group.tag, CacheLayerLayout(std::move(layers)));
            result.group_configs_.emplace(target_group.tag, std::move(local_group));
            result.group_types_.emplace(target_group.tag, target_group.policy.group_type);
        }
        return result;
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

    const CacheGroup& groupConfig(std::string_view tag) const {
        const std::string value(tag);
        const auto        it = group_configs_.find(value);
        RTP_LLM_CHECK_WITH_INFO(
            it != group_configs_.end(), "GroupedCacheLayerLayout missing group config tag=%s", value.c_str());
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
    GroupLayouts                          groups_;
    std::map<std::string, CacheGroup>     group_configs_;
    std::map<std::string, CacheGroupType> group_types_;
    std::vector<std::vector<std::string>> layer_group_tags_;
};

struct KVCacheBuffer {
    torch::Tensor kv_blocks;
    torch::Tensor kv_scale_blocks;
};

}  // namespace rtp_llm
