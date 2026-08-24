#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockExpression.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

CacheGroupType groupTypeForSpec(const KVCacheSpec& spec) {
    return spec.type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR : CacheGroupType::FULL;
}

bool isFullAttentionSpec(KVCacheSpecType type) {
    return type == KVCacheSpecType::MultiHeadAttention || type == KVCacheSpecType::MultiHeadLatentAttention;
}

std::string cacheGroupPolicySummary(const CacheGroupPolicy& policy) {
    std::ostringstream os;
    os << "{group_type=" << cacheGroupTypeName(policy.group_type) << ", prefix_reuse=" << policy.enable_prefix_reuse
       << ", evict=" << static_cast<int>(policy.evict_policy) << ", reservable=" << policy.reservable
       << ", explicit_block_num=" << policy.explicit_block_num << ", active_tail_blocks=" << policy.active_tail_blocks
       << ", validate_tail_blocks=" << policy.validate_tail_blocks
       << ", cp_mapping=" << static_cast<int>(policy.cp_mapping) << ", cp_slice=" << static_cast<int>(policy.cp_slice)
       << '}';
    return os.str();
}

std::string targetGroupSummary(const CacheConfig& target_config) {
    std::ostringstream os;
    os << '[';
    bool first = true;
    for (const auto& group : target_config.groups()) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << "{tag=" << group.tag << ", group_type=" << cacheGroupTypeName(group.policy.group_type);
        if (group.spec != nullptr) {
            os << ", spec_type=" << static_cast<int>(group.spec->type)
               << ", dtype=" << static_cast<int>(group.spec->memoryLayoutDType())
               << ", block_size_bytes=" << group.spec->block_size_bytes()
               << ", scale_block_size_bytes=" << group.spec->scale_block_size_bytes();
        } else {
            os << ", spec=null";
        }
        os << ", seq_size_per_block=" << group.seq_size_per_block
           << ", kv_block_stride_bytes=" << group.kv_block_stride_bytes
           << ", kv_scale_stride_bytes=" << group.kv_scale_stride_bytes
           << ", policy=" << cacheGroupPolicySummary(group.policy) << '}';
    }
    os << ']';
    return os.str();
}

std::optional<std::string> resolveDefaultMTPGroupAlias(const CacheConfig& target_config,
                                                       const CacheConfig& propose_config) {
    const auto& propose_groups = propose_config.groups();
    if (propose_groups.size() != 1 || propose_groups.front().tag != "default") {
        return std::nullopt;
    }

    for (const auto& target_group : target_config.groups()) {
        if (target_group.tag == "default") {
            return std::nullopt;  // Exact tag matching remains authoritative.
        }
    }

    const auto& source_group = propose_groups.front();
    if (source_group.policy.group_type != CacheGroupType::FULL || source_group.spec == nullptr
        || !isFullAttentionSpec(source_group.spec->type)) {
        return std::nullopt;
    }

    std::vector<std::string> candidates;
    for (const auto& target_group : target_config.groups()) {
        // The aliased draft layer uses its own MTP memory layout. Group-tag APIs still expose it as the target
        // group, however, so both logical block granularity and physical block shape must remain compatible.
        if (CacheConfig::samePolicy(target_group.policy, source_group.policy) && target_group.spec != nullptr
            && target_group.spec->type == source_group.spec->type
            && target_group.spec->memoryLayoutDType() == source_group.spec->memoryLayoutDType()
            && target_group.spec->block_size_bytes() == source_group.spec->block_size_bytes()
            && target_group.spec->scale_block_size_bytes() == source_group.spec->scale_block_size_bytes()
            && target_group.seq_size_per_block == source_group.seq_size_per_block
            && target_group.kv_block_stride_bytes == source_group.kv_block_stride_bytes
            && target_group.kv_scale_stride_bytes == source_group.kv_scale_stride_bytes) {
            candidates.push_back(target_group.tag);
        }
    }

    if (candidates.empty()) {
        const auto target_summary = targetGroupSummary(target_config);
        const auto source_policy  = cacheGroupPolicySummary(source_group.policy);
        RTP_LLM_FAIL("CacheConfig::composeMTPModule no compatible target group for sole propose tag=default: "
                     "source_spec_type=%d source_dtype=%d source_seq_size_per_block=%zu "
                     "source_block_size_bytes=%zu source_scale_block_size_bytes=%zu "
                     "source_kv_block_stride_bytes=%zu source_kv_scale_stride_bytes=%zu "
                     "source_policy=%s target_groups=%s",
                     static_cast<int>(source_group.spec->type),
                     static_cast<int>(source_group.spec->memoryLayoutDType()),
                     source_group.seq_size_per_block,
                     source_group.spec->block_size_bytes(),
                     source_group.spec->scale_block_size_bytes(),
                     source_group.kv_block_stride_bytes,
                     source_group.kv_scale_stride_bytes,
                     source_policy.c_str(),
                     target_summary.c_str());
    }
    if (candidates.size() > 1) {
        const auto target_summary = targetGroupSummary(target_config);
        RTP_LLM_FAIL("CacheConfig::composeMTPModule ambiguous default FULL group mapping: "
                     "compatible target groups=%zu spec_type=%d target_groups=%s",
                     candidates.size(),
                     static_cast<int>(source_group.spec->type),
                     target_summary.c_str());
    }
    return candidates.front();
}

}  // namespace

bool CacheConfig::samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs) {
    return lhs.group_type == rhs.group_type && lhs.enable_prefix_reuse == rhs.enable_prefix_reuse
           && lhs.evict_policy == rhs.evict_policy && lhs.reservable == rhs.reservable
           && lhs.explicit_block_num == rhs.explicit_block_num && lhs.active_tail_blocks == rhs.active_tail_blocks
           && lhs.validate_tail_blocks == rhs.validate_tail_blocks && lhs.cp_mapping == rhs.cp_mapping
           && lhs.cp_slice == rhs.cp_slice;
}

size_t CacheGroup::kernelBlocksPerPoolBlock() const {
    return PoolBlockToKernelBlockProjection(seq_size_per_block, kernel_seq_size_per_block).projectedSize(1);
}

void CacheConfig::replaceAssemblyTopology(std::vector<CacheGroup> groups, std::vector<CacheLayer> layers) {
    groups_ = std::move(groups);
    layers_ = std::move(layers);
    validateAndBuildIndex();
}

void CacheConfig::validateAndBuildIndex() {
    RTP_LLM_CHECK_WITH_INFO(!groups_.empty(), "CacheConfig requires at least one cache group");
    RTP_LLM_CHECK_WITH_INFO(!layers_.empty(), "CacheConfig requires at least one cache layer");
    const auto expected_layers = main_layer_count_;
    RTP_LLM_CHECK_WITH_INFO(expected_layers == 0 || layers_.size() >= static_cast<size_t>(expected_layers),
                            "CacheConfig layer count %zu is smaller than main layer count %u",
                            layers_.size(),
                            expected_layers);

    tag_to_idx_.clear();
    tag_to_idx_.reserve(groups_.size());
    std::vector<std::unordered_set<int>> group_layers(groups_.size());
    for (size_t idx = 0; idx < groups_.size(); ++idx) {
        auto& group = groups_[idx];
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheConfig got null spec at group %zu", idx);
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheConfig requires tag for group %zu", idx);
        RTP_LLM_CHECK_WITH_INFO(
            tag_to_idx_.emplace(group.tag, idx).second, "CacheConfig has duplicate tag=%s", group.tag.c_str());

        const auto expected_group_type = groupTypeForSpec(*group.spec);
        RTP_LLM_CHECK_WITH_INFO(expected_group_type != CacheGroupType::LINEAR
                                    || group.policy.group_type == CacheGroupType::LINEAR,
                                "CacheConfig group %zu tag=%s policy type %s does not match spec type %d",
                                idx,
                                group.tag.c_str(),
                                cacheGroupTypeName(group.policy.group_type),
                                static_cast<int>(group.spec->type));

        group.spec = group.spec->clone();
        if (group.block_num == 0) {
            group.block_num = block_count_basis_;
        }
        if (group.seq_size_per_block == 0) {
            group.seq_size_per_block = group.spec->seq_size_per_block > 0 ?
                                           group.spec->seq_size_per_block :
                                           std::max<size_t>(1, cache_key_block_tokens_);
        }
        if (group.kernel_seq_size_per_block == 0) {
            group.kernel_seq_size_per_block =
                group.policy.group_type == CacheGroupType::FULL && kernel_block_tokens_ > 0 ?
                    std::min(kernel_block_tokens_, group.seq_size_per_block) :
                    group.seq_size_per_block;
        }
        if (group.kv_block_stride_bytes == 0) {
            group.kv_block_stride_bytes = group.spec->block_size_bytes();
        }
        if (group.kv_scale_stride_bytes == 0) {
            group.kv_scale_stride_bytes = group.spec->scale_block_size_bytes();
        }
        RTP_LLM_CHECK_WITH_INFO(group.kv_block_stride_bytes == group.spec->block_size_bytes(),
                                "CacheConfig tag=%s kv stride=%zu differs from spec block bytes=%zu",
                                group.tag.c_str(),
                                group.kv_block_stride_bytes,
                                group.spec->block_size_bytes());
        RTP_LLM_CHECK_WITH_INFO(group.kv_scale_stride_bytes == group.spec->scale_block_size_bytes(),
                                "CacheConfig tag=%s scale stride=%zu differs from spec scale block bytes=%zu",
                                group.tag.c_str(),
                                group.kv_scale_stride_bytes,
                                group.spec->scale_block_size_bytes());
        group.kernelBlocksPerPoolBlock();
        for (int layer_id : group.layer_ids) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers_.size(),
                                    "CacheConfig tag=%s has invalid layer_id=%d",
                                    group.tag.c_str(),
                                    layer_id);
            RTP_LLM_CHECK_WITH_INFO(group_layers[idx].emplace(layer_id).second,
                                    "CacheConfig tag=%s has duplicate layer_id=%d",
                                    group.tag.c_str(),
                                    layer_id);
        }
    }

    for (size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        const auto& layer = layers_[layer_index];
        RTP_LLM_CHECK_WITH_INFO(!layer.empty(), "CacheConfig layer=%zu has no cache group", layer_index);
        std::unordered_set<std::string> seen_tags;
        for (const auto& tag : layer) {
            const auto idx_it = tag_to_idx_.find(tag);
            RTP_LLM_CHECK_WITH_INFO(idx_it != tag_to_idx_.end(),
                                    "CacheConfig layer=%zu references unknown tag=%s",
                                    layer_index,
                                    tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(
                seen_tags.emplace(tag).second, "CacheConfig layer=%zu has duplicate tag=%s", layer_index, tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(group_layers[idx_it->second].count(static_cast<int>(layer_index)) != 0,
                                    "CacheConfig layer=%zu tag=%s is missing reverse group membership",
                                    layer_index,
                                    tag.c_str());
        }
    }
    for (const auto& group : groups_) {
        for (int layer_id : group.layer_ids) {
            const auto& tags = layers_[static_cast<size_t>(layer_id)];
            RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), group.tag) != tags.end(),
                                    "CacheConfig tag=%s layer=%d is missing reverse layer membership",
                                    group.tag.c_str(),
                                    layer_id);
        }
    }
}

const CacheGroup& CacheConfig::group(std::string_view tag) const {
    const std::string value(tag);
    const auto        it = tag_to_idx_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_idx_.end(), "CacheConfig missing tag=%s", value.c_str());
    return groups_[it->second];
}

const std::vector<std::string>& CacheConfig::groupTagsForLayer(int layer_id) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers_.size(),
                            "CacheConfig invalid layer_id=%d size=%zu",
                            layer_id,
                            layers_.size());
    return layers_[static_cast<size_t>(layer_id)];
}

const CacheGroup& CacheConfig::groupForLayer(int layer_id, std::string_view tag) const {
    const auto&       tags = groupTagsForLayer(layer_id);
    const std::string value(tag);
    RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), value) != tags.end(),
                            "CacheConfig layer=%d does not own tag=%s",
                            layer_id,
                            value.c_str());
    return group(tag);
}

const CacheGroup& CacheConfig::soleGroupForLayer(int layer_id) const {
    const auto& tags = groupTagsForLayer(layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        tags.size() == 1, "CacheConfig layer=%d requires exactly one group, got %zu", layer_id, tags.size());
    return group(tags.front());
}

bool CacheConfig::hasSingleGlobalGroup() const noexcept {
    return groups_.size() == 1;
}

bool CacheConfig::hasOneGroupPerLayer() const noexcept {
    return std::all_of(layers_.begin(), layers_.end(), [](const CacheLayer& layer) { return layer.size() == 1; });
}

size_t CacheConfig::mtpModuleCount() const noexcept {
    return mtp_sub_configs_.size();
}

const CacheConfig& CacheConfig::mtpModule(size_t module_index) const {
    RTP_LLM_CHECK_WITH_INFO(module_index < mtp_sub_configs_.size(),
                            "CacheConfig invalid MTP module index=%zu count=%zu",
                            module_index,
                            mtp_sub_configs_.size());
    const auto& module = mtp_sub_configs_[module_index];
    RTP_LLM_CHECK_WITH_INFO(module != nullptr, "CacheConfig MTP module index=%zu is null", module_index);
    return *module;
}

void CacheConfig::replaceAssemblyGroupBlockLayout(const std::vector<uint32_t>& block_nums,
                                                  const std::vector<size_t>&   kv_block_stride_bytes,
                                                  const std::vector<size_t>&   kv_scale_stride_bytes) {
    const size_t group_num = groups_.size();
    RTP_LLM_CHECK_WITH_INFO(block_nums.size() == group_num,
                            "CacheConfig::replaceGroupBlockLayout block_nums size %zu != group size %zu",
                            block_nums.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(kv_block_stride_bytes.size() == group_num,
                            "CacheConfig::replaceGroupBlockLayout kv stride size %zu != group size %zu",
                            kv_block_stride_bytes.size(),
                            group_num);
    RTP_LLM_CHECK_WITH_INFO(kv_scale_stride_bytes.size() == group_num,
                            "CacheConfig::replaceGroupBlockLayout scale stride size %zu != group size %zu",
                            kv_scale_stride_bytes.size(),
                            group_num);
    auto groups = groups_;
    for (size_t idx = 0; idx < group_num; ++idx) {
        groups[idx].block_num             = block_nums[idx];
        groups[idx].kv_block_stride_bytes = kv_block_stride_bytes[idx];
        groups[idx].kv_scale_stride_bytes = kv_scale_stride_bytes[idx];
    }
    group_block_layout_initialized_ = true;
    replaceAssemblyTopology(std::move(groups), layers_);
}

std::shared_ptr<CacheConfig>
CacheConfig::composeAssemblyMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(groupNums() > 0, "CacheConfig::composeMTPModule requires destination topology");
    RTP_LLM_CHECK_WITH_INFO(propose_config.groupNums() > 0, "CacheConfig::composeMTPModule requires propose topology");
    RTP_LLM_CHECK_WITH_INFO(module_index >= 0, "CacheConfig::composeMTPModule invalid module_index=%d", module_index);

    auto sub_cfg                = std::make_shared<CacheConfig>(propose_config);
    sub_cfg->block_count_basis_ = block_count_basis_;

    const auto mtp_layer_num = propose_config.main_layer_count_;
    const auto total_layers =
        static_cast<size_t>(main_layer_num) + static_cast<size_t>(module_index + 1) * mtp_layer_num;
    auto target_groups = groups_;
    auto target_layers = layers_;
    target_layers.resize(total_layers);

    const auto target_group_num = target_groups.size();
    // Tag is the only identity used to pair a propose group with a target group.
    std::unordered_map<std::string, const CacheGroup*> propose_group_by_tag;
    for (const auto& propose_group : propose_config.groups()) {
        propose_group_by_tag.emplace(propose_group.tag, &propose_group);
    }
    const auto default_alias_target_tag = resolveDefaultMTPGroupAlias(*this, propose_config);

    std::vector<CacheGroup> sub_groups;
    std::vector<CacheLayer> sub_layers(static_cast<size_t>(mtp_layer_num));
    sub_groups.reserve(target_group_num);

    // target_idx only walks the local target_groups copy being mutated here; the
    // tag read from it is the identity everything else is keyed by.
    for (size_t target_idx = 0; target_idx < target_group_num; ++target_idx) {
        const std::string tag             = target_groups[target_idx].tag;
        const auto        propose_it      = propose_group_by_tag.find(tag);
        const bool        has_exact_group = propose_it != propose_group_by_tag.end();
        const bool        uses_default_alias =
            !has_exact_group && default_alias_target_tag.has_value() && tag == *default_alias_target_tag;

        const CacheGroup* source_group_ptr = &group(tag);
        if (has_exact_group) {
            source_group_ptr = propose_it->second;
        } else if (uses_default_alias) {
            // The alias holds only when propose has exactly one group tagged "default".
            source_group_ptr = &propose_config.group("default");
        }
        const bool  has_propose_group = has_exact_group || uses_default_alias;
        const auto& source_group      = *source_group_ptr;

        if (has_propose_group) {
            RTP_LLM_CHECK_WITH_INFO(
                source_group.layer_ids.size() == static_cast<size_t>(mtp_layer_num),
                "CacheConfig::composeMTPModule source_tag=%s target_tag=%s must cover every module layer, "
                "got=%zu expected=%u",
                source_group.tag.c_str(),
                tag.c_str(),
                source_group.layer_ids.size(),
                mtp_layer_num);
            for (size_t local_layer_id = 0; local_layer_id < source_group.layer_ids.size(); ++local_layer_id) {
                RTP_LLM_CHECK_WITH_INFO(
                    source_group.layer_ids[local_layer_id] == static_cast<int>(local_layer_id),
                    "CacheConfig::composeMTPModule source_tag=%s target_tag=%s source layers must be ordered 0..%u, "
                    "index=%zu value=%d",
                    source_group.tag.c_str(),
                    tag.c_str(),
                    mtp_layer_num - 1,
                    local_layer_id,
                    source_group.layer_ids[local_layer_id]);
            }

            const size_t expected_existing_layers = static_cast<size_t>(shared_pool_layout_layer_count_)
                                                    + static_cast<size_t>(module_index) * mtp_layer_num;
            RTP_LLM_CHECK_WITH_INFO(target_groups[target_idx].layer_ids.size() == expected_existing_layers,
                                    "CacheConfig::composeMTPModule source_tag=%s target_tag=%s "
                                    "physical group alignment mismatch: "
                                    "existing_layers=%zu expected=%zu module=%d group_layer_num=%d module_layers=%u",
                                    source_group.tag.c_str(),
                                    tag.c_str(),
                                    target_groups[target_idx].layer_ids.size(),
                                    expected_existing_layers,
                                    module_index,
                                    shared_pool_layout_layer_count_,
                                    mtp_layer_num);
        }

        CacheGroup sub_group = source_group;
        sub_group.layer_ids.clear();
        if (uses_default_alias) {
            RTP_LLM_LOG_INFO("CacheConfig::composeMTPModule aliases propose tag=default to target tag=%s: "
                             "module=%d spec_type=%d dtype=%d seq_size_per_block=%zu "
                             "kv_block_stride_bytes=%zu kv_scale_stride_bytes=%zu",
                             tag.c_str(),
                             module_index,
                             static_cast<int>(source_group.spec->type),
                             static_cast<int>(source_group.spec->memoryLayoutDType()),
                             source_group.seq_size_per_block,
                             source_group.kv_block_stride_bytes,
                             source_group.kv_scale_stride_bytes);
            auto aliased_spec = source_group.spec->clone();
            sub_group.tag     = tag;
            sub_group.spec    = std::move(aliased_spec);
        }

        if (!has_propose_group) {
            sub_groups.push_back(std::move(sub_group));
            continue;
        }

        for (int local_layer_id : source_group.layer_ids) {
            if (local_layer_id < 0 || local_layer_id >= static_cast<int>(mtp_layer_num)) {
                continue;
            }
            const auto global_layer_id = mtpGlobalLayerId(main_layer_num, module_index, mtp_layer_num, local_layer_id);
            RTP_LLM_CHECK_WITH_INFO(global_layer_id != std::numeric_limits<uint32_t>::max(),
                                    "CacheConfig::composeMTPModule invalid global layer: main=%u module=%d "
                                    "module_layers=%u local=%d",
                                    main_layer_num,
                                    module_index,
                                    mtp_layer_num,
                                    local_layer_id);
            const auto global_layer = static_cast<size_t>(global_layer_id);

            sub_group.layer_ids.push_back(local_layer_id);
            auto& sub_layer = sub_layers[static_cast<size_t>(local_layer_id)];
            sub_layer.push_back(tag);

            target_groups[target_idx].layer_ids.push_back(static_cast<int>(global_layer_id));
            target_layers[global_layer].push_back(tag);
        }

        sub_groups.push_back(std::move(sub_group));
    }

    RTP_LLM_CHECK_WITH_INFO(sub_groups.size() == target_group_num,
                            "CacheConfig::composeMTPModule sub group count %zu != target group count %zu",
                            sub_groups.size(),
                            target_group_num);
    for (size_t layer_id = 0; layer_id < sub_layers.size(); ++layer_id) {
        RTP_LLM_CHECK_WITH_INFO(!sub_layers[layer_id].empty(),
                                "CacheConfig::composeMTPModule missing group mapping for sub layer %zu",
                                layer_id);
    }

    sub_cfg->group_block_layout_initialized_ = group_block_layout_initialized_;
    sub_cfg->replaceAssemblyTopology(std::move(sub_groups), std::move(sub_layers));
    replaceAssemblyTopology(std::move(target_groups), std::move(target_layers));
    return sub_cfg;
}
void CacheConfig::projectAssemblyBlockCounts(uint32_t block_count_basis) {
    if (block_count_basis > 0) {
        block_count_basis_ = block_count_basis;
    }

    if (!uses_independent_block_pools_ || !group_block_layout_initialized_ || groupNums() == 0) {
        explicit_pool_reserve_bytes_ = 0;
        if (groupNums() > 0) {
            auto groups = groups_;
            for (auto& group : groups) {
                group.block_num = block_count_basis;
            }
            replaceAssemblyTopology(std::move(groups), layers_);
        }
        return;
    }

    size_t     reserve = 0;
    const auto step    = static_cast<uint32_t>(std::max(1, linear_step_));
    auto       groups  = groups_;
    for (auto& group_config : groups) {
        const auto explicit_independent_blocks = group_config.policy.explicit_block_num;
        uint32_t   rule_blocks                 = block_count_basis;
        if (explicit_independent_blocks > 0) {
            rule_blocks = explicit_independent_blocks;
        } else if (group_config.policy.group_type == CacheGroupType::SWA) {
            rule_blocks = block_count_basis / step + (block_count_basis % step != 0 ? 1u : 0u);
        }
        group_config.block_num = rule_blocks;

        if (explicit_independent_blocks > 0) {
            reserve += static_cast<size_t>(rule_blocks) * group_config.layer_ids.size()
                       * (group_config.kv_block_stride_bytes + group_config.kv_scale_stride_bytes);
        }
    }
    explicit_pool_reserve_bytes_ = reserve;
    replaceAssemblyTopology(std::move(groups), layers_);
}

std::string CacheConfig::debugString(size_t indent) const {
    const std::string indent_str = std::string(indent, ' ');
    const std::string indent1    = indent_str + "  ";

    std::ostringstream os;
    os << indent_str << "CacheConfig{\n";

#define OUTPUT_FIELD(field) os << indent1 << #field << "=" << field << "\n"
#define OUTPUT_FIELD_EXPR(name, expr) os << indent1 << name << "=" << expr << "\n"

    os << indent1 << "# Model Configuration:\n";
    OUTPUT_FIELD_EXPR("dtype", static_cast<int>(dtype_));
    OUTPUT_FIELD_EXPR("layer_num", main_layer_count_);
    OUTPUT_FIELD_EXPR("layer_all_num", layers_.size());
    OUTPUT_FIELD_EXPR("use_mla", (uses_mla_ ? "true" : "false"));
    os << "\n";

    os << indent1 << "# Block Configuration:\n";
    OUTPUT_FIELD_EXPR("block_num", block_count_basis_);
    OUTPUT_FIELD_EXPR("seq_size_per_block", cache_key_block_tokens_);
    OUTPUT_FIELD_EXPR("kernel_seq_size_per_block", kernel_block_tokens_);
    os << "\n";

    os << indent1 << "# Block Sizing Information:\n";
    OUTPUT_FIELD_EXPR("block_size_bytes", paged_block_budget_bytes_);
    OUTPUT_FIELD_EXPR("kv_block_stride_bytes", shared_pool_kv_block_stride_bytes_);
    OUTPUT_FIELD_EXPR("kv_scale_stride_bytes", shared_pool_kv_scale_stride_bytes_);
    os << "\n";

    // Debug rendering walks the tagged group records directly; the printed order
    // is local storage order and carries no identity.
    const auto&                   resolved_groups = groups();
    std::vector<CacheGroupPolicy> group_policies;
    std::vector<uint32_t>         group_block_nums;
    std::vector<std::string>      group_tags;
    std::vector<std::vector<int>> layers_by_group;
    group_policies.reserve(resolved_groups.size());
    group_tags.reserve(resolved_groups.size());
    layers_by_group.reserve(resolved_groups.size());
    for (const auto& group : resolved_groups) {
        group_policies.push_back(group.policy);
        group_tags.push_back(group.tag);
        layers_by_group.push_back(group.layer_ids);
        if (group_block_layout_initialized_) {
            group_block_nums.push_back(group.block_num);
        }
    }

    std::vector<std::vector<std::string>> layer_to_group_tags;
    layer_to_group_tags.reserve(layers().size());
    for (const auto& layer : layers()) {
        layer_to_group_tags.push_back(layer);
    }

    os << indent1 << "# Attention Configuration:\n";
    OUTPUT_FIELD_EXPR("linear_step", linear_step_);
    OUTPUT_FIELD_EXPR("group_layer_num", shared_pool_layout_layer_count_);
    OUTPUT_FIELD_EXPR("full_group_num",
                      std::count_if(group_policies.begin(), group_policies.end(), [](const CacheGroupPolicy& p) {
                          return p.group_type == CacheGroupType::FULL;
                      }));
    OUTPUT_FIELD_EXPR("linear_group_num",
                      std::count_if(group_policies.begin(), group_policies.end(), [](const CacheGroupPolicy& p) {
                          return p.group_type == CacheGroupType::LINEAR;
                      }));
    os << indent1 << "group_block_nums=" << rtp_llm::vectorToString(group_block_nums) << "\n";
    os << "\n";

    os << indent1 << "# Cache Specifications:\n";
    OUTPUT_FIELD_EXPR("groups.size()", resolved_groups.size());
    for (size_t i = 0; i < resolved_groups.size(); ++i) {
        const auto& spec = resolved_groups[i].spec;
        if (!spec) {
            os << indent1 << "groups[" << i << "].spec=null\n";
            continue;
        }

        os << indent1 << "groups[" << i << "] {\n";
        os << spec->debugString(indent + 2);
        os << indent1 << "}\n";
    }
    os << "\n";

    os << indent1 << "# Layer Mapping:\n";
    OUTPUT_FIELD_EXPR("layers_by_group.size()", layers_by_group.size());
    os << indent1 << "layers_by_group=" << rtp_llm::vectorsToString(layers_by_group) << "\n";
    OUTPUT_FIELD_EXPR("group_policies.size()", group_policies.size());
    os << indent1 << "group_types=[";
    for (size_t i = 0; i < group_policies.size(); ++i) {
        os << static_cast<int>(group_policies[i].group_type);
        if (i + 1 < group_policies.size()) {
            os << ",";
        }
    }
    os << "]\n";
    OUTPUT_FIELD_EXPR("group_tags.size()", group_tags.size());
    os << indent1 << "group_tags=[";
    for (size_t i = 0; i < group_tags.size(); ++i) {
        os << group_tags[i];
        if (i + 1 < group_tags.size()) {
            os << ",";
        }
    }
    os << "]\n";
    OUTPUT_FIELD_EXPR("layer_to_group_tags.size()", layer_to_group_tags.size());
    os << indent1 << "layer_to_group_tags=[";
    for (size_t layer_index = 0; layer_index < layer_to_group_tags.size(); ++layer_index) {
        if (layer_index > 0) {
            os << ",";
        }
        os << "[";
        for (size_t tag_index = 0; tag_index < layer_to_group_tags[layer_index].size(); ++tag_index) {
            if (tag_index > 0) {
                os << ",";
            }
            os << layer_to_group_tags[layer_index][tag_index];
        }
        os << "]";
    }
    os << "]\n";
    os << "\n";

    os << indent1 << "# MTP Configurations:\n";
    OUTPUT_FIELD_EXPR("mtp_sub_configs.size()", mtp_sub_configs_.size());
    for (size_t i = 0; i < mtp_sub_configs_.size(); ++i) {
        const auto& sub = mtp_sub_configs_[i];
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
