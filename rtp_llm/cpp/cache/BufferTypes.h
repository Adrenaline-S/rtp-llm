#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <torch/extension.h>
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct BlockBufferPtrInfo {
    torch::Tensor kv_addr;
    torch::Tensor kv_scale_addr;
};

struct CacheLayerLayout {
    std::vector<std::vector<int>> layer_to_group_ids;
    std::vector<CacheGroupType>   group_types;
    std::vector<size_t>           group_seq_block_sizes;
    std::vector<size_t>           group_kernel_seq_block_sizes;
    std::vector<size_t>           group_kernel_blocks_per_kv_block;
    std::vector<std::string>        group_tags;
    std::vector<std::map<std::string, int>> layer_tag_to_group_id;
    std::vector<CacheGroupType>   layer_attn_types;
    std::vector<torch::Tensor>              layers_to_kv_buffer_ptrs;
    std::vector<torch::Tensor>              layers_to_scale_buffer_ptrs;
    std::vector<std::vector<torch::Tensor>> layers_to_kv_buffer_ptrs_by_group;
    std::vector<std::vector<torch::Tensor>> layers_to_scale_buffer_ptrs_by_group;
};

struct CacheGroupLayerLayout {
    int            group_id           = -1;
    std::string    group_tag;
    CacheGroupType group_type         = CacheGroupType::FULL;
    size_t         seq_size_per_block = 0;
    size_t         kernel_seq_size_per_block = 0;
    size_t         kernel_blocks_per_kv_block = 1;
    CacheLayerLayout layout;
};

struct GroupedCacheLayerLayout {
    std::vector<CacheGroupLayerLayout> group_layouts;

    // Compatibility snapshot used by existing call sites while the grouped API
    // is rolled through model/runtime boundaries.
    std::vector<std::vector<int>> layer_to_group_ids;
    std::vector<CacheGroupType>   group_types;
    std::vector<size_t>           group_seq_block_sizes;
    std::vector<size_t>           group_kernel_seq_block_sizes;
    std::vector<size_t>           group_kernel_blocks_per_kv_block;
    std::vector<std::string>      group_tags;
    std::vector<std::map<std::string, int>> layer_tag_to_group_id;
    std::vector<CacheGroupType>             layer_attn_types;
    std::vector<torch::Tensor>              layers_to_kv_buffer_ptrs;
    std::vector<torch::Tensor>              layers_to_scale_buffer_ptrs;
    std::vector<std::vector<torch::Tensor>> layers_to_kv_buffer_ptrs_by_group;
    std::vector<std::vector<torch::Tensor>> layers_to_scale_buffer_ptrs_by_group;

    const CacheLayerLayout& defaultLayout() const {
        RTP_LLM_CHECK_WITH_INFO(group_layouts.size() == 1,
                                "GroupedCacheLayerLayout::defaultLayout requires exactly one group, got %zu",
                                group_layouts.size());
        return group_layouts.front().layout;
    }

    CacheLayerLayout& defaultLayout() {
        RTP_LLM_CHECK_WITH_INFO(group_layouts.size() == 1,
                                "GroupedCacheLayerLayout::defaultLayout requires exactly one group, got %zu",
                                group_layouts.size());
        return group_layouts.front().layout;
    }

    const CacheGroupLayerLayout& groupLayout(int group_id) const {
        for (const auto& group_layout : group_layouts) {
            if (group_layout.group_id == group_id) {
                return group_layout;
            }
        }
        RTP_LLM_FAIL("GroupedCacheLayerLayout missing group_id=%d", group_id);
    }

    CacheGroupLayerLayout& groupLayout(int group_id) {
        for (auto& group_layout : group_layouts) {
            if (group_layout.group_id == group_id) {
                return group_layout;
            }
        }
        RTP_LLM_FAIL("GroupedCacheLayerLayout missing group_id=%d", group_id);
    }

    static GroupedCacheLayerLayout fromFlat(const CacheLayerLayout& flat) {
        GroupedCacheLayerLayout grouped;
        grouped.layer_to_group_ids = flat.layer_to_group_ids;
        grouped.group_types = flat.group_types;
        grouped.group_seq_block_sizes = flat.group_seq_block_sizes;
        grouped.group_kernel_seq_block_sizes = flat.group_kernel_seq_block_sizes;
        grouped.group_kernel_blocks_per_kv_block = flat.group_kernel_blocks_per_kv_block;
        grouped.group_tags = flat.group_tags;
        grouped.layer_tag_to_group_id = flat.layer_tag_to_group_id;
        grouped.layer_attn_types = flat.layer_attn_types;
        grouped.layers_to_kv_buffer_ptrs = flat.layers_to_kv_buffer_ptrs;
        grouped.layers_to_scale_buffer_ptrs = flat.layers_to_scale_buffer_ptrs;
        grouped.layers_to_kv_buffer_ptrs_by_group = flat.layers_to_kv_buffer_ptrs_by_group;
        grouped.layers_to_scale_buffer_ptrs_by_group = flat.layers_to_scale_buffer_ptrs_by_group;

        const size_t group_count = std::max(flat.group_types.size(), flat.group_tags.size());
        grouped.group_layouts.reserve(group_count);
        for (size_t gid = 0; gid < group_count; ++gid) {
            CacheGroupLayerLayout group_layout;
            group_layout.group_id = static_cast<int>(gid);
            if (gid < flat.group_tags.size()) {
                group_layout.group_tag = flat.group_tags[gid];
            }
            if (gid < flat.group_types.size()) {
                group_layout.group_type = flat.group_types[gid];
            }
            if (gid < flat.group_seq_block_sizes.size()) {
                group_layout.seq_size_per_block = flat.group_seq_block_sizes[gid];
            }
            if (gid < flat.group_kernel_seq_block_sizes.size()) {
                group_layout.kernel_seq_size_per_block = flat.group_kernel_seq_block_sizes[gid];
            }
            if (gid < flat.group_kernel_blocks_per_kv_block.size()) {
                group_layout.kernel_blocks_per_kv_block = flat.group_kernel_blocks_per_kv_block[gid];
            }
            group_layout.layout.group_types = {group_layout.group_type};
            group_layout.layout.group_tags = {group_layout.group_tag};
            group_layout.layout.group_seq_block_sizes = {group_layout.seq_size_per_block};
            group_layout.layout.group_kernel_seq_block_sizes = {group_layout.kernel_seq_size_per_block};
            group_layout.layout.group_kernel_blocks_per_kv_block = {group_layout.kernel_blocks_per_kv_block};
            group_layout.layout.layer_to_group_ids.resize(flat.layer_to_group_ids.size());
            group_layout.layout.layer_tag_to_group_id.resize(flat.layer_tag_to_group_id.size());
            group_layout.layout.layer_attn_types.resize(flat.layer_attn_types.size(), group_layout.group_type);
            group_layout.layout.layers_to_kv_buffer_ptrs.resize(flat.layers_to_kv_buffer_ptrs.size());
            group_layout.layout.layers_to_scale_buffer_ptrs.resize(flat.layers_to_scale_buffer_ptrs.size());

            for (size_t layer_id = 0; layer_id < flat.layer_to_group_ids.size(); ++layer_id) {
                const auto& gids = flat.layer_to_group_ids[layer_id];
                if (std::find(gids.begin(), gids.end(), static_cast<int>(gid)) == gids.end()) {
                    continue;
                }
                group_layout.layout.layer_to_group_ids[layer_id] = {static_cast<int>(gid)};
                if (layer_id < flat.layer_tag_to_group_id.size()) {
                    group_layout.layout.layer_tag_to_group_id[layer_id] = flat.layer_tag_to_group_id[layer_id];
                }
                if (layer_id < flat.layers_to_kv_buffer_ptrs_by_group.size()
                    && gid < flat.layers_to_kv_buffer_ptrs_by_group[layer_id].size()) {
                    group_layout.layout.layers_to_kv_buffer_ptrs[layer_id] =
                        flat.layers_to_kv_buffer_ptrs_by_group[layer_id][gid];
                } else if (gids.size() == 1 && layer_id < flat.layers_to_kv_buffer_ptrs.size()) {
                    group_layout.layout.layers_to_kv_buffer_ptrs[layer_id] = flat.layers_to_kv_buffer_ptrs[layer_id];
                }
                if (layer_id < flat.layers_to_scale_buffer_ptrs_by_group.size()
                    && gid < flat.layers_to_scale_buffer_ptrs_by_group[layer_id].size()) {
                    group_layout.layout.layers_to_scale_buffer_ptrs[layer_id] =
                        flat.layers_to_scale_buffer_ptrs_by_group[layer_id][gid];
                } else if (gids.size() == 1 && layer_id < flat.layers_to_scale_buffer_ptrs.size()) {
                    group_layout.layout.layers_to_scale_buffer_ptrs[layer_id] = flat.layers_to_scale_buffer_ptrs[layer_id];
                }
            }
            grouped.group_layouts.push_back(std::move(group_layout));
        }
        if (grouped.group_layouts.empty() && !flat.layers_to_kv_buffer_ptrs.empty()) {
            CacheGroupLayerLayout group_layout;
            group_layout.group_id = 0;
            group_layout.layout = flat;
            grouped.group_layouts.push_back(std::move(group_layout));
        }
        return grouped;
    }
};

struct KVCacheBuffer {
    torch::Tensor kv_blocks;
    torch::Tensor kv_scale_blocks;
};

}  // namespace rtp_llm
