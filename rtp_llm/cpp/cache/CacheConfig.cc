#include "rtp_llm/cpp/cache/CacheConfig.h"

#include <algorithm>
#include <sstream>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.policyFingerprint() == rhs.policyFingerprint();
}

void CacheConfig::setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers) {
    main_config.setTopology(std::move(new_groups), std::move(new_layers));
}

void CacheConfig::setGroupBlockLayout(const std::vector<std::string>& group_tags,
                                      const std::vector<uint32_t>&    block_nums,
                                      const std::vector<size_t>&      kv_block_stride_bytes,
                                      const std::vector<size_t>&      kv_scale_stride_bytes) {
    main_config.setGroupBlockLayout(group_tags, block_nums, kv_block_stride_bytes, kv_scale_stride_bytes);
}

std::vector<GroupIdentity> CacheConfig::allGroupIdentities() const {
    std::vector<GroupIdentity> identities;
    for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
        for (const auto& tag : moduleConfig(module_id).groupTags()) {
            identities.push_back({tag, module_id});
        }
    }
    return identities;
}

void CacheConfig::validateCrossModuleInvariants() const {
    RTP_LLM_CHECK_WITH_INFO(moduleCount() > 0, "cache config requires at least one module");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "cache-key block span must be positive");
    std::vector<std::string> seen_tags;
    for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
        const auto& module = moduleConfig(module_id);
        RTP_LLM_CHECK_WITH_INFO(module.seq_size_per_block == seq_size_per_block,
                                "module %zu cache-key block span %zu differs from shared span %zu",
                                module_id,
                                module.seq_size_per_block,
                                seq_size_per_block);
        for (const auto& tag : module.groupTags()) {
            RTP_LLM_CHECK_WITH_INFO(
                std::find(seen_tags.begin(), seen_tags.end(), tag) == seen_tags.end(),
                "cache tag '%s' (module %zu) collides with the tag of another module; the Python model "
                "description must declare module-scoped tags (e.g. mtp_<tag>)",
                tag.c_str(),
                module_id);
            seen_tags.push_back(tag);
        }
        for (size_t earlier_id = 0; earlier_id < module_id; ++earlier_id) {
            for (const auto& group : module.groups()) {
                if (!group.policy.enable_prefix_reuse) {
                    continue;
                }
                for (const auto& earlier : moduleConfig(earlier_id).groups()) {
                    if (earlier.policy.enable_prefix_reuse && earlier.policy.group_type == group.policy.group_type) {
                        RTP_LLM_CHECK_WITH_INFO(
                            earlier.policy.explicit_block_num == group.policy.explicit_block_num,
                            "jointly reusable groups %s and %s must use the same explicit block capacity",
                            earlier.tag.c_str(),
                            group.tag.c_str());
                    }
                }
            }
        }
    }
}

void CacheConfig::fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                   const std::vector<std::vector<int>>& layers_by_group,
                                   const std::vector<CacheGroupType>&   types,
                                   const std::vector<std::string>&      tags,
                                   const std::vector<CacheGroupPolicy>& policies) {
    main_config.fromGroupedSpecs(specs, layers_by_group, types, tags, policies);
}

uint32_t CacheConfig::blockWeight(const std::string& tag) const {
    const auto it = pool_block_weights.find(tag);
    return it == pool_block_weights.end() ? 1u : it->second;
}

uint32_t CacheConfig::blockNumForGroup(const GroupBase& group, uint32_t baseline) const {
    const auto weight = blockWeight(group.tag);
    RTP_LLM_CHECK_WITH_INFO(weight > 0, "cache pool '%s' block weight must be positive", group.tag.c_str());
    // Main's #1574 rule: a group either pins its own explicit block count or
    // follows the global baseline; pool weights scale the baseline share.
    const uint64_t blocks =
        group.policy.explicit_block_num > 0 ?
            group.policy.explicit_block_num :
            static_cast<uint64_t>(baseline) * weight;
    RTP_LLM_CHECK_WITH_INFO(
        blocks <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
        "cache pool '%s' block count exceeds int32 block ID range: baseline=%u weight=%u blocks=%lu",
        group.tag.c_str(),
        baseline,
        weight,
        blocks);
    return static_cast<uint32_t>(blocks);
}

uint32_t CacheConfig::maxBaselineBlockNum() const {
    uint64_t maximum = std::numeric_limits<uint32_t>::max();
    for (const auto& identity : allGroupIdentities()) {
        if (isAllocationOnly(identity.module_id)) {
            continue;
        }
        const auto& group = this->group(identity.tag, identity.module_id);
        if (group.policy.explicit_block_num > 0) {
            (void)blockNumForGroup(group, 1);
            continue;
        }
        const auto weight = blockWeight(group.tag);
        RTP_LLM_CHECK_WITH_INFO(weight > 0, "cache pool '%s' block weight must be positive", group.tag.c_str());
        // Baseline * weight must stay inside the int32 block-ID range.
        const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / weight;
        maximum = std::min(maximum, limit);
    }
    return static_cast<uint32_t>(maximum);
}

void CacheConfig::finalizeBlockNums(uint32_t                   global_block_num,
                                    const RuntimeConfig&       runtime_config,
                                    const PPBlockNumOverrides* pp_overrides) {
    RTP_LLM_CHECK_WITH_INFO(global_block_num > 0, "cache configuration requires positive baseline block count");
    (void)runtime_config;
    const auto block_num_for_group = [&](const GroupBase& group) {
        if (pp_overrides == nullptr) {
            return blockNumForGroup(group, global_block_num);
        }
        const auto it = pp_overrides->find(group.tag);
        RTP_LLM_CHECK_WITH_INFO(
            it != pp_overrides->end(), "cache group [%s] is missing from PP block overrides", group.tag.c_str());
        return it->second;
    };
    main_config.finalizeBlockNums(block_num_for_group);
    for (auto& sub_cfg : mtp_sub_configs) {
        // The draft allocation row on stage 0 has no stage-local capacity.
        // An actual count is assigned by PP negotiation from the owning stage.
        if (!sub_cfg.allocation_only || pp_overrides != nullptr) {
            sub_cfg.finalizeBlockNums(block_num_for_group);
        }
    }
}

std::string CacheConfig::debugString(size_t indent) const {
    const std::string indent_str = std::string(indent, ' ');
    const std::string indent1    = indent_str + "  ";

    std::ostringstream os;
    os << indent_str << "CacheConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

    os << indent1 << "# Shared Configuration:\n";
    OUTPUT_FIELD(cache_min_replay_tokens);
    OUTPUT_FIELD(cache_key_hash_seed);
    OUTPUT_FIELD(seq_size_per_block);
    OUTPUT_FIELD(global_layer_begin);
    os << "\n";

#undef OUTPUT_FIELD
#undef OUTPUT_FIELD_EXPR

    os << indent1 << "main_config:\n";
    os << main_config.debugString(indent + 2);
    os << "\n";

    os << indent1 << "mtp_sub_configs.size()=" << mtp_sub_configs.size() << "\n";
    for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
        os << indent1 << "mtp_sub_configs[" << i << "]:\n";
        os << mtp_sub_configs[i].debugString(indent + 4);
    }
    os << "\n";

    os << indent_str << "}\n";
    return os.str();
}

}  // namespace rtp_llm
