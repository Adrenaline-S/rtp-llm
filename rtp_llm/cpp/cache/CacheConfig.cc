#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace rtp_llm {

namespace {

CacheGroupType groupTypeForSpec(const KVCacheSpec& spec) {
    return spec.type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
}

std::vector<GroupBase> copyGroups(const CacheTopology& topology) {
    const auto groups = topology.groups();
    return {groups.begin(), groups.end()};
}

size_t checkedAddSize(size_t lhs, size_t rhs, const char* what) {
    RTP_LLM_CHECK_WITH_INFO(lhs <= std::numeric_limits<size_t>::max() - rhs, "%s overflow", what);
    return lhs + rhs;
}

size_t checkedMulSize(size_t lhs, size_t rhs, const char* what) {
    RTP_LLM_CHECK_WITH_INFO(lhs == 0 || rhs <= std::numeric_limits<size_t>::max() / lhs, "%s overflow", what);
    return lhs * rhs;
}

}  // namespace

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.group_type == rhs.group_type && lhs.enable_prefix_reuse == rhs.enable_prefix_reuse
           && lhs.evict_policy == rhs.evict_policy && lhs.reservable == rhs.reservable
           && lhs.fixed_block_num == rhs.fixed_block_num && lhs.charge_to_paged_budget == rhs.charge_to_paged_budget
           && lhs.active_tail_blocks == rhs.active_tail_blocks && lhs.validate_tail_blocks == rhs.validate_tail_blocks
           && lhs.cp_mapping == rhs.cp_mapping && lhs.cp_slice == rhs.cp_slice;
}

void CacheConfig::setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
    RTP_LLM_CHECK_WITH_INFO(!new_groups.empty(), "CacheConfig::setTopology requires at least one cache group");
    RTP_LLM_CHECK_WITH_INFO(!new_layers.empty(), "CacheConfig::setTopology requires at least one cache layer");
    const auto expected_layers = layer_all_num > 0 ? layer_all_num : layer_num;
    RTP_LLM_CHECK_WITH_INFO(expected_layers == 0 || new_layers.size() == static_cast<size_t>(expected_layers),
                            "CacheConfig::setTopology layer count %zu != expected %u",
                            new_layers.size(),
                            expected_layers);

    for (size_t gid = 0; gid < new_groups.size(); ++gid) {
        auto& group = new_groups[gid];
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheConfig::setTopology got null spec at group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheConfig::setTopology requires tag for group %zu", gid);
        RTP_LLM_CHECK_WITH_INFO(group.spec->tag == group.tag,
                                "CacheConfig::setTopology tag=%s does not match spec tag=%s",
                                group.tag.c_str(),
                                group.spec->tag.c_str());

        const auto expected_group_type = groupTypeForSpec(*group.spec);
        RTP_LLM_CHECK_WITH_INFO(expected_group_type != CacheGroupType::LINEAR
                                    || group.policy.group_type == CacheGroupType::LINEAR,
                                "CacheConfig::setTopology group %zu tag=%s policy type %s does not match spec type %d",
                                gid,
                                group.tag.c_str(),
                                cacheGroupTypeName(group.policy.group_type),
                                static_cast<int>(group.spec->type));

        group.spec = group.spec->clone();
    }

    cache_topology = CacheTopology::create(std::move(new_groups), std::move(new_layers));
}

void CacheConfig::setGroupPolicies(const std::unordered_map<std::string, CacheGroupPolicy>& policies) {
    RTP_LLM_CHECK_WITH_INFO(policies.size() == topology().groups().size(),
                            "CacheConfig::setGroupPolicies size %zu != group size %zu",
                            policies.size(),
                            topology().groups().size());
    auto groups = copyGroups(topology());
    for (auto& group : groups) {
        const auto it = policies.find(group.tag);
        RTP_LLM_CHECK_WITH_INFO(
            it != policies.end(), "CacheConfig::setGroupPolicies missing tag=%s", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == it->second.group_type,
                                "CacheConfig::setGroupPolicies cannot change group type for tag=%s",
                                group.tag.c_str());
        group.policy = it->second;
    }
    setTopology(std::move(groups), topology().layers());
}

std::shared_ptr<CacheConfig>
CacheConfig::mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(groupNums() > 0, "CacheConfig::mergeMTPModule requires destination topology");
    RTP_LLM_CHECK_WITH_INFO(propose_config.groupNums() > 0, "CacheConfig::mergeMTPModule requires propose topology");
    RTP_LLM_CHECK_WITH_INFO(module_index >= 0, "CacheConfig::mergeMTPModule invalid module_index=%d", module_index);

    auto sub_cfg           = std::make_shared<CacheConfig>(propose_config);
    sub_cfg->layer_all_num = sub_cfg->layer_num;

    const auto mtp_layer_num = propose_config.layer_num;
    const auto total_layers =
        static_cast<size_t>(main_layer_num) + static_cast<size_t>(module_index + 1) * mtp_layer_num;
    auto target_groups = copyGroups(topology());
    auto target_layers = topology().layers();
    target_layers.resize(total_layers);
    for (size_t layer_id = 0; layer_id < target_layers.size(); ++layer_id) {
        target_layers[layer_id].layer_id = static_cast<int>(layer_id);
    }
    const auto target_group_num = target_groups.size();

    std::vector<GroupBase> sub_groups;
    std::vector<LayerBase> sub_layers(static_cast<size_t>(mtp_layer_num));
    sub_groups.reserve(target_group_num);

    for (auto& target_group : target_groups) {
        const auto& tag               = target_group.tag;
        const bool  has_propose_group = propose_config.topology().groups().size() > 0
                                       && std::any_of(propose_config.topology().groups().begin(),
                                                      propose_config.topology().groups().end(),
                                                      [&tag](const GroupBase& group) { return group.tag == tag; });
        const auto& source_group = has_propose_group ? propose_config.group(tag) : target_group;

        if (has_propose_group) {
            RTP_LLM_CHECK_WITH_INFO(source_group.seq_size_per_block == target_group.seq_size_per_block,
                                    "CacheConfig::mergeMTPModule tag=%s physical span mismatch: main=%zu propose=%zu",
                                    tag.c_str(),
                                    target_group.seq_size_per_block,
                                    source_group.seq_size_per_block);
            RTP_LLM_CHECK_WITH_INFO(
                samePolicy(source_group.policy, target_group.policy),
                "CacheConfig::mergeMTPModule tag=%s policy mismatch between main and propose cache groups",
                tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(
                source_group.layer_ids.size() == static_cast<size_t>(mtp_layer_num),
                "CacheConfig::mergeMTPModule tag=%s must cover every module layer, got=%zu expected=%u",
                tag.c_str(),
                source_group.layer_ids.size(),
                mtp_layer_num);
            for (size_t local_layer_id = 0; local_layer_id < source_group.layer_ids.size(); ++local_layer_id) {
                RTP_LLM_CHECK_WITH_INFO(
                    source_group.layer_ids[local_layer_id] == static_cast<int>(local_layer_id),
                    "CacheConfig::mergeMTPModule tag=%s source layers must be ordered 0..%u, index=%zu value=%d",
                    tag.c_str(),
                    mtp_layer_num - 1,
                    local_layer_id,
                    source_group.layer_ids[local_layer_id]);
            }

            const auto   main_group_layers = static_cast<size_t>(std::count_if(
                target_group.layer_ids.begin(), target_group.layer_ids.end(), [main_layer_num](int layer_id) {
                    return layer_id >= 0 && static_cast<uint32_t>(layer_id) < main_layer_num;
                }));
            const size_t expected_existing_layers =
                main_group_layers + static_cast<size_t>(module_index) * mtp_layer_num;
            RTP_LLM_CHECK_WITH_INFO(target_group.layer_ids.size() == expected_existing_layers,
                                    "CacheConfig::mergeMTPModule tag=%s physical group alignment mismatch: "
                                    "existing_layers=%zu expected=%zu module=%d main_group_layers=%zu module_layers=%u",
                                    tag.c_str(),
                                    target_group.layer_ids.size(),
                                    expected_existing_layers,
                                    module_index,
                                    main_group_layers,
                                    mtp_layer_num);
        }

        GroupBase sub_group = source_group;
        sub_group.layer_ids.clear();

        if (!has_propose_group) {
            sub_groups.push_back(std::move(sub_group));
            continue;
        }

        for (int local_layer_id : propose_config.layerIdsForGroup(tag)) {
            if (local_layer_id < 0 || local_layer_id >= static_cast<int>(mtp_layer_num)) {
                continue;
            }
            const auto global_layer_id = mtpGlobalLayerId(main_layer_num, module_index, mtp_layer_num, local_layer_id);
            RTP_LLM_CHECK_WITH_INFO(global_layer_id != std::numeric_limits<uint32_t>::max(),
                                    "CacheConfig::mergeMTPModule invalid global layer: main=%u module=%d "
                                    "module_layers=%u local=%d",
                                    main_layer_num,
                                    module_index,
                                    mtp_layer_num,
                                    local_layer_id);
            const auto global_layer = static_cast<size_t>(global_layer_id);

            sub_group.layer_ids.push_back(local_layer_id);
            auto& sub_layer    = sub_layers[static_cast<size_t>(local_layer_id)];
            sub_layer.layer_id = local_layer_id;
            sub_layer.group_tags.push_back(tag);

            target_group.layer_ids.push_back(static_cast<int>(global_layer_id));
            target_layers[global_layer].group_tags.push_back(tag);
        }

        sub_groups.push_back(std::move(sub_group));
    }

    RTP_LLM_CHECK_WITH_INFO(sub_groups.size() == target_group_num,
                            "CacheConfig::mergeMTPModule sub group count %zu != target group count %zu",
                            sub_groups.size(),
                            target_group_num);
    for (size_t layer_id = 0; layer_id < sub_layers.size(); ++layer_id) {
        RTP_LLM_CHECK_WITH_INFO(!sub_layers[layer_id].group_tags.empty(),
                                "CacheConfig::mergeMTPModule missing group mapping for sub layer %zu",
                                layer_id);
    }

    sub_cfg->setTopology(std::move(sub_groups), std::move(sub_layers));
    layer_all_num = static_cast<uint32_t>(total_layers);
    setTopology(std::move(target_groups), std::move(target_layers));
    return sub_cfg;
}

size_t CacheConfig::blockSizeBytesForGroup(std::string_view tag) const {
    const auto&  main_group = group(tag);
    const size_t main_layers =
        static_cast<size_t>(std::count_if(main_group.layer_ids.begin(), main_group.layer_ids.end(), [&](int layer_id) {
            return layer_id >= 0 && static_cast<uint32_t>(layer_id) < layer_num;
        }));
    const auto layoutBytes = [&](const GroupBase& source_group, size_t layer_count) {
        const size_t layer_stride =
            checkedAddSize(source_group.kv_block_stride_bytes, source_group.kv_scale_stride_bytes, "group stride");
        return checkedMulSize(layer_count, layer_stride, "group layout block bytes");
    };

    size_t bytes = layoutBytes(main_group, main_layers);
    for (const auto& sub_config : mtp_sub_configs) {
        if (!sub_config || !sub_config->topology().contains(tag)) {
            continue;
        }
        const auto& sub_group = sub_config->group(tag);
        bytes = checkedAddSize(bytes, layoutBytes(sub_group, sub_group.layer_ids.size()), "group combined block bytes");
    }
    return bytes;
}

const GroupBase& CacheConfig::physicalGroupForLayer(int layer_id, std::string_view tag) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0, "invalid cache layer id=%d", layer_id);
    if (static_cast<uint32_t>(layer_id) < layer_num) {
        return groupForLayer(layer_id, tag);
    }
    size_t layer_begin = layer_num;
    for (const auto& sub_config : mtp_sub_configs) {
        if (!sub_config) {
            continue;
        }
        const size_t layer_end = checkedAddSize(layer_begin, sub_config->layer_num, "MTP layer range");
        if (static_cast<size_t>(layer_id) < layer_end) {
            RTP_LLM_CHECK_WITH_INFO(sub_config->topology().contains(tag),
                                    "MTP cache layer=%d has no group tag=%s",
                                    layer_id,
                                    std::string(tag).c_str());
            return sub_config->groupForLayer(static_cast<int>(static_cast<size_t>(layer_id) - layer_begin), tag);
        }
        layer_begin = layer_end;
    }
    RTP_LLM_FAIL("cache layer=%d tag=%s is outside main/MTP layouts", layer_id, std::string(tag).c_str());
}

size_t CacheConfig::layerBlockStrideBytes(int layer_id) const {
    size_t stride = 0;
    for (const auto& tag : topology().layer(layer_id).group_tags) {
        const auto& group = physicalGroupForLayer(layer_id, tag);
        stride            = std::max(
            stride, checkedAddSize(group.kv_block_stride_bytes, group.kv_scale_stride_bytes, "layer group stride"));
    }
    return stride;
}

void CacheConfig::applyTokenCapacity(uint64_t capacity_tokens) {
    RTP_LLM_CHECK_WITH_INFO(capacity_tokens > 0, "CacheConfig::applyTokenCapacity requires positive tokens");
    auto groups = copyGroups(topology());
    for (auto& group : groups) {
        RTP_LLM_CHECK_WITH_INFO(group.seq_size_per_block > 0,
                                "CacheConfig::applyTokenCapacity tag=%s has zero physical span",
                                group.tag.c_str());
        uint64_t blocks = 0;
        if (group.policy.fixed_block_num > 0) {
            blocks = group.policy.fixed_block_num;
        } else {
            blocks =
                capacity_tokens / group.seq_size_per_block + (capacity_tokens % group.seq_size_per_block != 0 ? 1 : 0);
        }
        RTP_LLM_CHECK_WITH_INFO(blocks > 0 && blocks <= std::numeric_limits<uint32_t>::max(),
                                "CacheConfig::applyTokenCapacity tag=%s block count overflow: %lu",
                                group.tag.c_str(),
                                blocks);
        group.block_num = static_cast<uint32_t>(blocks);
    }
    setTopology(std::move(groups), topology().layers());
    for (auto& sub_cfg : mtp_sub_configs) {
        if (sub_cfg != nullptr) {
            sub_cfg->applyTokenCapacity(capacity_tokens);
        }
    }
}

uint64_t CacheConfig::tokenCapacity() const {
    RTP_LLM_CHECK_WITH_INFO(groupNums() > 0, "CacheConfig::tokenCapacity requires topology");
    uint64_t capacity = std::numeric_limits<uint64_t>::max();
    for (const auto& group : topology().groups()) {
        if (group.policy.fixed_block_num > 0) {
            continue;
        }
        capacity = std::min(capacity,
                            static_cast<uint64_t>(group.block_num) * static_cast<uint64_t>(group.seq_size_per_block));
    }
    return capacity != std::numeric_limits<uint64_t>::max() ? capacity : 0;
}

std::string CacheConfig::debugString(size_t indent) const {
    const std::string indent_str = std::string(indent, ' ');
    const std::string indent1    = indent_str + "  ";

    std::ostringstream os;
    os << indent_str << "CacheConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

    os << indent1 << "# Model Configuration:\n";
    OUTPUT_FIELD_EXPR("dtype", static_cast<int>(dtype));
    OUTPUT_FIELD(layer_num);
    OUTPUT_FIELD(layer_all_num);
    OUTPUT_FIELD_EXPR("use_mla", (use_mla ? "true" : "false"));
    os << "\n";

    const auto topology_groups = topology().groups();

    os << indent1 << "# Attention Configuration:\n";
    OUTPUT_FIELD_EXPR("full_group_num",
                      std::count_if(topology_groups.begin(), topology_groups.end(), [](const GroupBase& group) {
                          return group.policy.group_type == CacheGroupType::FULL;
                      }));
    OUTPUT_FIELD_EXPR("linear_group_num",
                      std::count_if(topology_groups.begin(), topology_groups.end(), [](const GroupBase& group) {
                          return group.policy.group_type == CacheGroupType::LINEAR;
                      }));
    os << indent1 << "group_block_layout=[";
    bool first_group = true;
    for (const auto& group : topology_groups) {
        if (!first_group) {
            os << ",";
        }
        first_group = false;
        os << "{" << group.tag << ":" << group.block_num << "}";
    }
    os << "]\n";
    os << "\n";

    os << indent1 << "# Cache Specifications:\n";
    OUTPUT_FIELD_EXPR("groups.size()", topology_groups.size());
    for (const auto& group : topology_groups) {
        const auto& spec = group.spec;
        if (!spec) {
            os << indent1 << "group[" << group.tag << "].spec=null\n";
            continue;
        }

        os << indent1 << "group[" << group.tag << "] {\n";
        os << spec->debugString(indent + 2);
        os << indent1 << "}\n";
    }
    os << "\n";

    os << indent1 << "# Layer Mapping:\n";
    for (const auto& group : topology_groups) {
        os << indent1 << "group[" << group.tag << "].layers=" << rtp_llm::vectorToString(group.layer_ids) << "\n";
        os << indent1 << "group[" << group.tag << "].type=" << static_cast<int>(group.policy.group_type) << "\n";
    }
    os << "\n";

    os << indent1 << "# MTP Configurations:\n";
    OUTPUT_FIELD_EXPR("mtp_sub_configs.size()", mtp_sub_configs.size());
    for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
        const auto& sub = mtp_sub_configs[i];
        if (!sub) {
            os << indent1 << "mtp_sub_configs[" << i << "]=null\n";
            continue;
        }
        os << indent1 << "mtp_sub_configs[" << i << "]:\n";
        os << sub->debugString(indent + 4);
    }
    os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

    os << indent_str << "}\n";
    return os.str();
}

}  // namespace rtp_llm
