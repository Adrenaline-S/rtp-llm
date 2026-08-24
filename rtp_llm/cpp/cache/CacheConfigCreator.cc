#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

#include "rtp_llm/cpp/cache/BlockExpression.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool blockNumFitsBudget(uint32_t block_num, size_t total_budget_bytes, const KVCacheBlockBudget& budget, int step) {
    if (budget.explicit_pool_reserve_bytes > total_budget_bytes) {
        return false;
    }

    size_t remaining = total_budget_bytes - budget.explicit_pool_reserve_bytes;
    if (budget.paged_block_bytes > 0) {
        if (static_cast<size_t>(block_num) > remaining / budget.paged_block_bytes) {
            return false;
        }
        remaining -= static_cast<size_t>(block_num) * budget.paged_block_bytes;
    }

    const auto safe_step  = static_cast<uint32_t>(std::max(1, step));
    const auto swa_blocks = block_num / safe_step + (block_num % safe_step != 0 ? 1u : 0u);
    return budget.swa_block_bytes == 0 || static_cast<size_t>(swa_blocks) <= remaining / budget.swa_block_bytes;
}

KVCacheBlockBudget blockBudgetForConfig(const CacheConfig& config) {
    KVCacheBlockBudget budget;
    if (!config.usesIndependentBlockPools()) {
        budget.paged_block_bytes = config.pagedBlockBudgetBytes();
        return budget;
    }

    budget.explicit_pool_reserve_bytes = config.explicitPoolReserveBytes();
    for (const auto& group : config.groups()) {
        if (config.usesExplicitIndependentBlocks(group.tag)) {
            continue;
        }
        const auto group_bytes = config.blockSizeBytes(group.tag);
        switch (group.policy.group_type) {
            case CacheGroupType::FULL:
            case CacheGroupType::LINEAR:
                budget.paged_block_bytes += group_bytes;
                break;
            case CacheGroupType::SWA:
                budget.swa_block_bytes += group_bytes;
                break;
        }
    }
    return budget;
}

void addBlockBudget(KVCacheBlockBudget& total, const KVCacheBlockBudget& addition, size_t multiplier = 1) {
    const auto add = [multiplier](size_t& dst, size_t value, const char* name) {
        RTP_LLM_CHECK_WITH_INFO(multiplier == 0 || value <= (std::numeric_limits<size_t>::max() - dst) / multiplier,
                                "kv cache %s budget overflow: current=%zu addition=%zu multiplier=%zu",
                                name,
                                dst,
                                value,
                                multiplier);
        dst += value * multiplier;
    };
    add(total.explicit_pool_reserve_bytes, addition.explicit_pool_reserve_bytes, "explicit reserve");
    add(total.paged_block_bytes, addition.paged_block_bytes, "paged block bytes");
    add(total.swa_block_bytes, addition.swa_block_bytes, "SWA block bytes");
}

}  // namespace

void CacheConfigCreator::setupKernelSeqSize(CacheConfig&         config,
                                            const KVCacheConfig& kv_cache_config,
                                            const char*          config_name) {
    auto groups = config.groups();
    if (kv_cache_config.kernel_seq_size_per_block > 0) {
        const auto requested_kernel_seq_size_per_block = static_cast<size_t>(kv_cache_config.kernel_seq_size_per_block);
        (void)config_name;
        (void)PoolBlockToKernelBlockProjection(config.cacheKeyBlockTokens(), requested_kernel_seq_size_per_block);
        for (auto& group : groups) {
            group.kernel_seq_size_per_block =
                group.policy.group_type == CacheGroupType::FULL ?
                    std::min(requested_kernel_seq_size_per_block, group.seq_size_per_block) :
                    group.seq_size_per_block;
        }
    }

    size_t compatibility_seq_size    = std::numeric_limits<size_t>::max();
    size_t compatibility_kernel_size = std::numeric_limits<size_t>::max();
    for (const auto& group : groups) {
        compatibility_seq_size    = std::min(compatibility_seq_size, group.seq_size_per_block);
        compatibility_kernel_size = std::min(compatibility_kernel_size, group.kernel_seq_size_per_block);
    }
    config.cache_key_block_tokens_ = compatibility_seq_size;
    config.kernel_block_tokens_    = compatibility_kernel_size;
    config.replaceAssemblyTopology(std::move(groups), config.layers());
}

uint32_t CacheConfigCreator::computeBlockNum(CacheConfig&                                     config,
                                             const ModelConfig&                               model_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        config.projectAssemblyBlockCounts(kv_cache_config.test_block_num);
        return static_cast<uint32_t>(kv_cache_config.test_block_num);
    }

    const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
        runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
    config.projectAssemblyBlockCounts(0);

    const auto block_budget = blockBudgetForConfig(config);
    if (block_budget.explicit_pool_reserve_bytes > 0) {
        RTP_LLM_CHECK_WITH_INFO(kv_cache_mem_size > block_budget.explicit_pool_reserve_bytes,
                                "kv cache budget %zu MiB is smaller than explicitly-sized pool reservation %zu MiB "
                                "(reduce explicitly sized pool blocks if needed)",
                                kv_cache_mem_size / 1024 / 1024,
                                block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
        RTP_LLM_LOG_INFO("kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB",
                         kv_cache_mem_size / 1024 / 1024,
                         block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
    }
    return maxKVCacheBlockNumForBudget(kv_cache_mem_size, block_budget, config.linearStep());
}

namespace {

uint32_t mhaLocalKvHeadNum(const ModelConfig& model_config, const ParallelismConfig& parallelism_config) {
    const auto     attn_tp = std::max<int64_t>(1, parallelism_config.get_attn_tp_size());
    const uint32_t tp      = static_cast<uint32_t>(attn_tp);
    const uint32_t kv      = static_cast<uint32_t>(model_config.attn_config.kv_head_num);
    RTP_LLM_CHECK_WITH_INFO(kv > 0, "local kv head num requires positive kv_head_num");
    return (kv % tp == 0) ? kv / tp : kv / std::gcd(kv, tp);
}

uint32_t linearLocalKvHeadNum(const ModelConfig& model_config, const ParallelismConfig& parallelism_config) {
    const auto     attn_tp     = std::max<int64_t>(1, parallelism_config.get_attn_tp_size());
    const uint32_t tp          = static_cast<uint32_t>(attn_tp);
    const uint32_t value_heads = static_cast<uint32_t>(model_config.linear_attention_config.linear_num_value_heads);
    RTP_LLM_CHECK_WITH_INFO(value_heads > 0, "local kv head num requires positive linear_num_value_heads");
    RTP_LLM_CHECK_WITH_INFO(value_heads % tp == 0,
                            "linear_num_value_heads must be divisible by attention TP, global=%u tp=%u",
                            value_heads,
                            tp);
    const uint32_t local_value_heads = value_heads / tp;
    RTP_LLM_CHECK_WITH_INFO(
        local_value_heads > 0, "invalid local linear value heads: global=%u tp=%u", value_heads, tp);
    return local_value_heads;
}

uint32_t localKvHeadNumForType(KVCacheSpecType          type,
                               const ModelConfig&       model_config,
                               const ParallelismConfig& parallelism_config) {
    switch (type) {
        case KVCacheSpecType::MultiHeadAttention:
            return mhaLocalKvHeadNum(model_config, parallelism_config);
        case KVCacheSpecType::LinearAttention:
            return linearLocalKvHeadNum(model_config, parallelism_config);
        case KVCacheSpecType::MultiHeadLatentAttention:
        case KVCacheSpecType::OpaqueKV:
        case KVCacheSpecType::OpaqueState:
            return 1;
        default:
            RTP_LLM_FAIL("unknown KVCacheSpecType=%d", static_cast<int>(type));
    }
    return 1;
}

int hybridGroupLayerNum(const ModelConfig& model_config) {
    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            model_config.num_layers);
    const int linear_count = static_cast<int>(std::count(types.begin(), types.end(), HybridAttentionType::LINEAR));
    const int full_count   = static_cast<int>(types.size()) - linear_count;
    if (linear_count > 0 && full_count > 0) {
        return std::max(std::gcd(linear_count, full_count), full_count);
    }
    return std::max({linear_count, full_count, 1});
}

void validateIndependentDescs(const ModelConfig& model_config,
                              uint32_t           kernel_tokens_per_block,
                              int                gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(
        model_config.kv_cache_spec_descs.size() == static_cast<size_t>(model_config.num_layers),
        "hybrid-pool desc config requires layer-wise kv_cache_spec_descs for every layer, got %zu/%ld",
        model_config.kv_cache_spec_descs.size(),
        model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(gen_num_per_cycle >= 0,
                            "hybrid-pool desc config requires non-negative gen_num_per_cycle, got %d",
                            gen_num_per_cycle);
    for (int64_t layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        const auto& descs = model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "hybrid-pool desc config layer %ld has no descs", layer_id);
        for (const auto& desc : descs) {
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
                RTP_LLM_CHECK_WITH_INFO(
                    desc.compression_ratio > 0,
                    "desc tag=%s derives entries from kernel block but has invalid compression_ratio=%u",
                    desc.tag.c_str(),
                    desc.compression_ratio);
                RTP_LLM_CHECK_WITH_INFO(
                    kernel_tokens_per_block > 0,
                    "desc tag=%s derives entries from kernel block but kernel_tokens_per_block is 0",
                    desc.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block % desc.compression_ratio == 0,
                                        "desc tag=%s compression_ratio=%u must divide kernel block %u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio,
                                        kernel_tokens_per_block);
                RTP_LLM_CHECK_WITH_INFO(desc.kernel_tokens_per_block_alignment > 0,
                                        "desc tag=%s has invalid kernel_tokens_per_block_alignment=0",
                                        desc.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(
                    kernel_tokens_per_block >= desc.kernel_tokens_per_block_alignment
                        && kernel_tokens_per_block % desc.kernel_tokens_per_block_alignment == 0,
                    "desc tag=%s derives entries from kernel block, so kernel_seq_size_per_block(%u) "
                    "must be >= %u and a multiple of %u",
                    desc.tag.c_str(),
                    kernel_tokens_per_block,
                    desc.kernel_tokens_per_block_alignment,
                    desc.kernel_tokens_per_block_alignment);
            }
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
                RTP_LLM_CHECK_WITH_INFO(desc.compression_ratio > 0,
                                        "state ring desc tag=%s requires positive compression_ratio",
                                        desc.tag.c_str());
            }
        }
    }
}

void validateSingleLayerSpecs(const ModelConfig& model_config, const LayerBuiltSpecs& layer_specs) {
    RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == static_cast<size_t>(model_config.num_layers),
                            "single cache config requires layer-wise runtime specs for every layer, got %zu/%ld",
                            layer_specs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(!layer_specs.empty(), "single cache config requires at least one runtime spec");
    RTP_LLM_CHECK_WITH_INFO(layer_specs[0].size() == 1,
                            "single cache config requires exactly one spec for layer 0, got %zu",
                            layer_specs[0].size());
    const auto& first = layer_specs[0][0];
    RTP_LLM_CHECK_WITH_INFO(first.spec != nullptr, "single cache config got null runtime spec for layer 0");
    const auto fingerprint = first.spec->fingerprint();
    for (int64_t layer_id = 1; layer_id < model_config.num_layers; ++layer_id) {
        const auto layer = static_cast<size_t>(layer_id);
        RTP_LLM_CHECK_WITH_INFO(layer_specs[layer].size() == 1,
                                "single cache config requires exactly one spec for layer %ld, got %zu",
                                layer_id,
                                layer_specs[layer].size());
        const auto& built = layer_specs[layer][0];
        RTP_LLM_CHECK_WITH_INFO(
            built.spec != nullptr, "single cache config got null runtime spec for layer %ld", layer_id);
        RTP_LLM_CHECK_WITH_INFO(
            built.tag == first.tag,
            "single cache config requires consistent tag across layers, layer %ld has tag=%s but layer 0 has tag=%s",
            layer_id,
            built.tag.c_str(),
            first.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            built.spec->fingerprint() == fingerprint, "single cache config spec differs at layer %ld", layer_id);
    }
}

void validateHybridLayerSpecs(const ModelConfig& model_config, const LayerBuiltSpecs& layer_specs) {
    const auto& types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(
        model_config.num_layers > 0, "invalid model_config.num_layers=%ld", model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == static_cast<size_t>(model_config.num_layers),
                            "runtime kv_cache specs size %zu != num_layers %ld",
                            layer_specs.size(),
                            model_config.num_layers);
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            model_config.num_layers);
}

}  // namespace

void CacheConfigCreator::populateGroupsFromLayerSpecs(CacheConfig&             config,
                                                      const LayerBuiltSpecs&   layer_specs,
                                                      const ModelConfig&       model_config,
                                                      const ParallelismConfig& parallelism_config) {
    RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == static_cast<size_t>(config.mainLayerCount()),
                            "cache layer spec count %zu != layer_num %u",
                            layer_specs.size(),
                            config.mainLayerCount());

    const bool legacy_hybrid = model_config.hybrid_attention_config.enable_hybrid_attention
                               && !model_config.hybrid_attention_config.enable_independent_kv_cache_pools;
    std::map<std::string, CacheGroup> groups_by_tag;
    std::vector<std::string>          ordered_tags;
    std::vector<CacheLayer>           layers(static_cast<size_t>(config.mainLayerCount()));
    for (uint32_t layer_id = 0; layer_id < config.mainLayerCount(); ++layer_id) {
        const auto& specs = layer_specs[layer_id];
        RTP_LLM_CHECK_WITH_INFO(!specs.empty(), "cache layer %u has no specs", layer_id);
        if (legacy_hybrid) {
            RTP_LLM_CHECK_WITH_INFO(specs.size() == 1,
                                    "hybrid layer %u must have exactly one runtime kv_cache spec, got %zu",
                                    layer_id,
                                    specs.size());
        }
        std::set<std::string> layer_tags;
        for (const auto& built : specs) {
            RTP_LLM_CHECK_WITH_INFO(!built.tag.empty(), "cache layer %u has empty spec tag", layer_id);
            RTP_LLM_CHECK_WITH_INFO(
                built.spec != nullptr, "cache layer %u tag=%s has null spec", layer_id, built.tag.c_str());
            if (legacy_hybrid) {
                const auto group_type = built.spec->type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR :
                                                                                               CacheGroupType::FULL;
                const auto expected_type = model_config.hybrid_attention_config.hybrid_attention_types[layer_id]
                                                   == HybridAttentionType::LINEAR ?
                                               CacheGroupType::LINEAR :
                                               CacheGroupType::FULL;
                RTP_LLM_CHECK_WITH_INFO(group_type == expected_type,
                                        "hybrid layer %u desc tag=%s cache type %d does not match attention type %d",
                                        layer_id,
                                        built.tag.c_str(),
                                        static_cast<int>(group_type),
                                        static_cast<int>(expected_type));
            }
            RTP_LLM_CHECK_WITH_INFO(layer_tags.insert(built.tag).second,
                                    "hybrid-pool layer %u has duplicate tag=%s",
                                    layer_id,
                                    built.tag.c_str());

            const auto local_heads = localKvHeadNumForType(built.spec->type, model_config, parallelism_config);
            auto [it, inserted]    = groups_by_tag.emplace(built.tag, CacheGroup{});
            auto& group            = it->second;
            if (inserted) {
                group.tag                       = built.tag;
                group.spec                      = built.spec;
                group.policy                    = built.policy;
                group.local_kv_head_num         = local_heads;
                group.seq_size_per_block        = built.spec->seq_size_per_block;
                group.kernel_seq_size_per_block = group.seq_size_per_block;
                ordered_tags.push_back(built.tag);
            } else {
                RTP_LLM_CHECK_WITH_INFO(group.spec->fingerprint() == built.spec->fingerprint(),
                                        "hybrid-pool tag=%s has multiple physical prototypes",
                                        built.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == built.policy.group_type,
                                        "hybrid-pool tag=%s has inconsistent group type",
                                        built.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(CacheConfig::samePolicy(group.policy, built.policy),
                                        "hybrid-pool tag=%s has inconsistent policy",
                                        built.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(group.local_kv_head_num == local_heads,
                                        "hybrid-pool tag=%s has inconsistent local_kv_head_num",
                                        built.tag.c_str());
            }
            group.layer_ids.push_back(static_cast<int>(layer_id));
            layers[layer_id].push_back(built.tag);
        }
    }

    if (legacy_hybrid) {
        std::stable_partition(ordered_tags.begin(), ordered_tags.end(), [&](const std::string& tag) {
            return groups_by_tag.at(tag).policy.group_type == CacheGroupType::FULL;
        });
    }
    std::vector<CacheGroup> groups;
    groups.reserve(groups_by_tag.size());
    for (const auto& tag : ordered_tags) {
        groups.push_back(std::move(groups_by_tag.at(tag)));
    }
    config.replaceAssemblyTopology(std::move(groups), std::move(layers));
}

void CacheConfigCreator::finalizeGroupStorage(CacheConfig& config) {
    auto                  groups = config.groups();
    std::vector<uint32_t> group_block_nums(groups.size(), 0);
    std::vector<size_t>   group_kv_strides(groups.size(), 0);
    std::vector<size_t>   group_scale_strides(groups.size(), 0);
    size_t                max_kv_stride           = 0;
    size_t                max_scale_stride        = 0;
    size_t                total_kv_block_bytes    = 0;
    size_t                total_scale_block_bytes = 0;
    uint32_t              max_group_layers        = 0;
    config.uses_typed_cache_regions_              = false;
    config.uses_opaque_kv_cache_store_            = false;

    for (size_t idx = 0; idx < groups.size(); ++idx) {
        auto& group = groups[idx];
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "cache group tag=%s has null spec", group.tag.c_str());
        (void)group.kernelBlocksPerPoolBlock();
        group.kv_block_stride_bytes = group.spec->block_size_bytes();
        group.kv_scale_stride_bytes = group.spec->scale_block_size_bytes();
        group.block_num             = 0;
        group_kv_strides[idx]       = group.kv_block_stride_bytes;
        group_scale_strides[idx]    = group.kv_scale_stride_bytes;

        const auto layer_count = static_cast<uint32_t>(group.layer_ids.size());
        const bool contributes_to_paged_budget =
            !config.usesIndependentBlockPools()
            || ((group.policy.group_type == CacheGroupType::FULL || group.policy.group_type == CacheGroupType::LINEAR)
                && group.policy.explicit_block_num == 0);
        if (contributes_to_paged_budget) {
            total_kv_block_bytes += static_cast<size_t>(layer_count) * group.kv_block_stride_bytes;
            total_scale_block_bytes += static_cast<size_t>(layer_count) * group.kv_scale_stride_bytes;
        }
        max_kv_stride    = std::max(max_kv_stride, group.kv_block_stride_bytes);
        max_scale_stride = std::max(max_scale_stride, group.kv_scale_stride_bytes);
        max_group_layers = std::max(max_group_layers, layer_count);
        const bool opaque =
            group.spec->type == KVCacheSpecType::OpaqueKV || group.spec->type == KVCacheSpecType::OpaqueState;
        config.uses_typed_cache_regions_   = config.usesTypedCacheRegions() || opaque;
        config.uses_opaque_kv_cache_store_ = config.usesOpaqueKVCacheStore() || opaque;
        config.is_sparse_                  = config.isSparse() || group.spec->type == KVCacheSpecType::OpaqueKV;
    }

    config.shared_pool_layout_layer_count_    = static_cast<int>(std::max<uint32_t>(1, max_group_layers));
    config.shared_pool_kv_block_stride_bytes_ = max_kv_stride;
    config.shared_pool_kv_scale_stride_bytes_ = max_scale_stride;
    const size_t paged_block_bytes            = total_kv_block_bytes + total_scale_block_bytes;
    if (paged_block_bytes == 0) {
        RTP_LLM_CHECK_WITH_INFO(config.usesTypedCacheRegions(), "cache paged groups produced zero block bytes");
        config.paged_block_budget_bytes_ = 1;
    } else {
        config.paged_block_budget_bytes_ = paged_block_bytes;
    }
    config.explicit_pool_reserve_bytes_ = 0;
    config.disables_decode_first_malloc_device_reuse_ =
        config.disablesDecodeFirstMallocDeviceReuse() || config.usesOpaqueKVCacheStore();
    config.replaceAssemblyTopology(std::move(groups), config.layers());
    const auto& finalized_groups   = config.groups();
    config.cache_key_block_tokens_ = std::numeric_limits<size_t>::max();
    config.kernel_block_tokens_    = std::numeric_limits<size_t>::max();
    for (const auto& group : finalized_groups) {
        config.cache_key_block_tokens_ = std::min(config.cacheKeyBlockTokens(), group.seq_size_per_block);
        config.kernel_block_tokens_    = std::min(config.kernelBlockTokens(), group.kernel_seq_size_per_block);
    }
    if (config.usesIndependentBlockPools()) {
        config.replaceAssemblyGroupBlockLayout(group_block_nums, group_kv_strides, group_scale_strides);
    }
}

static LayerBuiltSpecs buildLayerSpecsFromDescs(const LayerKVCacheSpecDescs& layer_descs,
                                                const SpecBuildContext&      ctx,
                                                int64_t                      expected_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(layer_descs.size() == static_cast<size_t>(expected_layer_num),
                            "kv_cache_spec_descs size %zu != num_layers %ld",
                            layer_descs.size(),
                            expected_layer_num);
    LayerBuiltSpecs layer_specs(layer_descs.size());
    for (size_t layer_id = 0; layer_id < layer_descs.size(); ++layer_id) {
        const auto& descs = layer_descs[layer_id];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "kv_cache_spec_descs layer %zu has no descs", layer_id);
        auto& specs = layer_specs[layer_id];
        specs.reserve(descs.size());
        for (const auto& desc : descs) {
            specs.push_back(SpecBuilder::build(desc, ctx));
        }
    }
    return layer_specs;
}

CacheConfig CacheConfigCreator::createConfigFromDescs(const ModelConfig&       model_config,
                                                      const ParallelismConfig& parallelism_config,
                                                      const KVCacheConfig&     kv_cache_config,
                                                      int                      gen_num_per_cycle) {
    const bool    independent            = model_config.hybrid_attention_config.enable_independent_kv_cache_pools;
    const bool    hybrid                 = model_config.hybrid_attention_config.enable_hybrid_attention;
    const auto    dtype                  = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    constexpr int kDefaultKvCacheSeqSize = 64;
    const bool    has_seq_override       = independent && kv_cache_config.seq_size_per_block > 0
                                  && kv_cache_config.seq_size_per_block != kDefaultKvCacheSeqSize;
    const auto physical_tokens_per_block = has_seq_override ?
                                               static_cast<uint32_t>(kv_cache_config.seq_size_per_block) :
                                               static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    const auto kernel_tokens_per_block   = independent && kv_cache_config.kernel_seq_size_per_block > 0 ?
                                               static_cast<uint32_t>(kv_cache_config.kernel_seq_size_per_block) :
                                               physical_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block > 0, "cache seq_size_per_block must be > 0");
    if (independent) {
        RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block > 0, "hybrid-pool kernel_seq_size_per_block must be > 0");
        (void)PoolBlockToKernelBlockProjection(physical_tokens_per_block, kernel_tokens_per_block);
        validateIndependentDescs(model_config, kernel_tokens_per_block, gen_num_per_cycle);
    }

    CacheConfig config;
    config.main_layer_count_             = static_cast<uint32_t>(model_config.num_layers);
    config.cache_key_block_tokens_       = 0;
    config.kernel_block_tokens_          = 0;
    config.uses_mla_                     = model_config.attn_config.use_mla;
    config.dtype_                        = dtype;
    config.linear_step_                  = 1;
    config.is_sparse_                    = model_config.attn_config.is_sparse;
    config.uses_independent_block_pools_ = independent || hybrid;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = physical_tokens_per_block;
    ctx.kernel_tokens_per_block = independent ? kernel_tokens_per_block : 0;
    ctx.attn_config             = &model_config.attn_config;
    ctx.linear_attention_config = &model_config.linear_attention_config;
    ctx.parallelism_config      = &parallelism_config;
    ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
    auto layer_specs = buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

    if (!independent && hybrid) {
        validateHybridLayerSpecs(model_config, layer_specs);
        for (auto& specs : layer_specs) {
            for (auto& built : specs) {
                const auto type = built.spec->type == KVCacheSpecType::LinearAttention ? CacheGroupType::LINEAR :
                                                                                         CacheGroupType::FULL;
                built.policy    = defaultCacheGroupPolicy(type);
            }
        }
    } else if (!independent) {
        validateSingleLayerSpecs(model_config, layer_specs);
        for (auto& specs : layer_specs) {
            specs[0].policy = defaultCacheGroupPolicy(specs[0].spec->type == KVCacheSpecType::LinearAttention ?
                                                          CacheGroupType::LINEAR :
                                                          CacheGroupType::FULL);
        }
    }

    populateGroupsFromLayerSpecs(config, layer_specs, model_config, parallelism_config);
    RTP_LLM_CHECK_WITH_INFO(config.groupNums() > 0, "cache config produced no cache specs");
    {
        auto groups = config.groups();
        for (auto& group : groups) {
            group.kernel_seq_size_per_block =
                group.policy.group_type == CacheGroupType::FULL ?
                    std::min(static_cast<size_t>(kernel_tokens_per_block), group.seq_size_per_block) :
                    group.seq_size_per_block;
        }
        config.replaceAssemblyTopology(std::move(groups), config.layers());
    }
    if (!independent && hybrid) {
        const auto full_group_num =
            std::count_if(config.groups().begin(), config.groups().end(), [](const CacheGroup& group) {
                return group.policy.group_type == CacheGroupType::FULL;
            });
        RTP_LLM_CHECK_WITH_INFO(
            full_group_num <= 1,
            "multiple full attention cache groups (%zu) are not supported: FMHA parameters bind one block table before "
            "the layer loop",
            static_cast<size_t>(full_group_num));
        if (full_group_num != 0) {
            const auto& full_group =
                *std::find_if(config.groups().begin(), config.groups().end(), [](const CacheGroup& group) {
                    return group.policy.group_type == CacheGroupType::FULL;
                });
            if (full_group.spec == nullptr || full_group.tag != "full") {
                RTP_LLM_LOG_WARNING("hybrid full cache group is expected to be tag=full, got tag=%s type=%d",
                                    full_group.spec == nullptr ? "<null>" : full_group.tag.c_str(),
                                    static_cast<int>(full_group.policy.group_type));
            }
        }
    }
    finalizeGroupStorage(config);
    if (!independent && hybrid) {
        config.shared_pool_layout_layer_count_ = hybridGroupLayerNum(model_config);
    } else if (!independent) {
        config.shared_pool_layout_layer_count_ = static_cast<int>(model_config.num_layers);
    }

    if (!independent) {
        const auto full_group_num =
            std::count_if(config.groups().begin(), config.groups().end(), [](const CacheGroup& group) {
                return group.policy.group_type == CacheGroupType::FULL && group.spec
                       && (group.spec->type == KVCacheSpecType::MultiHeadAttention
                           || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention);
            });
        RTP_LLM_CHECK_WITH_INFO(full_group_num == 1,
                                "cache config requires exactly one FULL MHA/MLA cache group, got %zu",
                                static_cast<size_t>(full_group_num));
    }
    return config;
}

uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget, int linear_step) {
    RTP_LLM_CHECK_WITH_INFO(budget.paged_block_bytes > 0 || budget.swa_block_bytes > 0,
                            "kv cache block budget has zero marginal block bytes");

    uint32_t low  = 0;
    uint32_t high = std::numeric_limits<uint32_t>::max();
    while (low < high) {
        const uint32_t mid = low + static_cast<uint32_t>((static_cast<uint64_t>(high) - low + 1) / 2);
        if (blockNumFitsBudget(mid, total_budget_bytes, budget, linear_step)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

CacheConfig CacheConfigCreator::createRankLocalConfig(const ModelConfig&                 model_config,
                                                      const ParallelismConfig&           parallelism_config,
                                                      const RuntimeConfig&               runtime_config,
                                                      const KVCacheConfig&               kv_cache_config,
                                                      const std::optional<WarmUpResult>& warm_up_result,
                                                      const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config = createConfigFromDescs(model_config, parallelism_config, kv_cache_config, 0);

    config.linear_step_ = kv_cache_config.linear_step;
    setupKernelSeqSize(config, kv_cache_config, "cache");

    uint32_t block_num = computeBlockNum(
        config, model_config, runtime_config, kv_cache_config, parallelism_config, warm_up_result, sp_config);
    RTP_LLM_CHECK_WITH_INFO(block_num > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_num,
                            static_cast<long>(config.pagedBlockBudgetBytes() / 1024 / 1024));

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.cacheKeyBlockTokens();
    config.block_count_basis_   = block_num;
    config.projectAssemblyBlockCounts(block_num);
    RTP_LLM_LOG_INFO("kv cache block nums is %u, allows storing %ld tokens", block_num, kv_cache_seq_len);
    if (kv_cache_seq_len < model_config.max_seq_len) {
        RTP_LLM_LOG_WARNING("kv cache block nums %u can only store %ld tokens, less than max_seq_len %ld, "
                            "this is dangerous, consider decrease max_seq_len",
                            block_num,
                            kv_cache_seq_len,
                            model_config.max_seq_len);
    }
    return config;
}

CacheConfig CacheConfigCreator::createRankLocalSpeculativeConfig(const ModelConfig&                score_model_config,
                                                                 const ModelConfig&                propose_model_config,
                                                                 const ParallelismConfig&          parallelism_config,
                                                                 const RuntimeConfig&              runtime_config,
                                                                 const KVCacheConfig&              kv_cache_config,
                                                                 const SpeculativeExecutionConfig& sp_config,
                                                                 const std::optional<WarmUpResult>& warm_up_result) {
    CacheConfig score_config =
        createConfigFromDescs(score_model_config, parallelism_config, kv_cache_config, sp_config.gen_num_per_cycle);
    CacheConfig propose_config =
        createConfigFromDescs(propose_model_config, parallelism_config, kv_cache_config, sp_config.gen_num_per_cycle);

    const int joint_step        = std::max(1, kv_cache_config.linear_step);
    score_config.linear_step_   = joint_step;
    propose_config.linear_step_ = joint_step;

    setupKernelSeqSize(score_config, kv_cache_config, "score");
    setupKernelSeqSize(propose_config, kv_cache_config, "propose");

    const bool is_mtp =
        sp_config.type == SP_TYPE_MTP || sp_config.type == SP_TYPE_EAGLE || sp_config.type == SP_TYPE_DSPARK;
    // MTP contributes one one-layer module per proposal step. EAGLE and
    // DSpARK each contribute one multi-layer draft model; their gamma is a
    // proposal width, not an independent-module count. Other speculative
    // modes keep the historical single draft-model topology.
    const int num_mtp_modules = sp_config.type == SP_TYPE_MTP ? sp_config.gen_num_per_cycle : 1;
    RTP_LLM_CHECK_WITH_INFO(num_mtp_modules > 0, "speculative cache requires at least one propose module");

    score_config.projectAssemblyBlockCounts(0);
    propose_config.projectAssemblyBlockCounts(0);

    uint32_t total_layer_num = score_config.mainLayerCount();
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_layer_num += propose_config.mainLayerCount();
    }

    size_t total_block_size_bytes = score_config.pagedBlockBudgetBytes();
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_block_size_bytes += propose_config.pagedBlockBudgetBytes();
    }

    // Compose the complete root/sub-model topology before solving capacity.
    // No partially composed score config is published or used as the result of
    // the memory-budget decision.
    CacheConfig config               = score_config;
    config.linear_step_              = joint_step;
    config.paged_block_budget_bytes_ = total_block_size_bytes;
    const uint32_t main_layer_num    = score_config.mainLayerCount();
    config.mtp_sub_configs_.clear();
    config.mtp_sub_configs_.reserve(static_cast<size_t>(num_mtp_modules));
    for (int m = 0; m < num_mtp_modules; ++m) {
        auto sub_cfg                     = config.composeAssemblyMTPModule(propose_config, m, main_layer_num);
        sub_cfg->cache_key_block_tokens_ = std::numeric_limits<size_t>::max();
        sub_cfg->kernel_block_tokens_    = std::numeric_limits<size_t>::max();
        for (const auto& group : sub_cfg->groups()) {
            sub_cfg->cache_key_block_tokens_ = std::min(sub_cfg->cacheKeyBlockTokens(), group.seq_size_per_block);
            sub_cfg->kernel_block_tokens_    = std::min(sub_cfg->kernelBlockTokens(), group.kernel_seq_size_per_block);
        }
        config.mtp_sub_configs_.push_back(std::move(sub_cfg));
    }

    KVCacheBlockBudget joint_budget = blockBudgetForConfig(score_config);
    addBlockBudget(joint_budget, blockBudgetForConfig(propose_config), static_cast<size_t>(num_mtp_modules));
    const size_t explicit_pool_reserve = joint_budget.explicit_pool_reserve_bytes;

    size_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);

        if (explicit_pool_reserve > 0) {
            RTP_LLM_CHECK_WITH_INFO(
                kv_cache_mem_size > explicit_pool_reserve,
                "sp kv cache budget %zu MiB is smaller than explicitly-sized pool reservation %zu MiB "
                "(reduce explicitly sized pool blocks if needed)",
                kv_cache_mem_size / 1024 / 1024,
                explicit_pool_reserve / 1024 / 1024);
            RTP_LLM_LOG_INFO(
                "sp kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB (score=%zu MiB + propose=%zu MiB x %d)",
                kv_cache_mem_size / 1024 / 1024,
                explicit_pool_reserve / 1024 / 1024,
                score_config.explicitPoolReserveBytes() / 1024 / 1024,
                propose_config.explicitPoolReserveBytes() / 1024 / 1024,
                num_mtp_modules);
        }
        block_num = maxKVCacheBlockNumForBudget(kv_cache_mem_size, joint_budget, joint_step);
    }

    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    config = projectBlockCounts(config, static_cast<uint32_t>(block_num), /*validate_basis=*/false);
    config.explicit_pool_reserve_bytes_ = explicit_pool_reserve;

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.cacheKeyBlockTokens();
    RTP_LLM_LOG_INFO("CacheConfig created: is_mtp=%d, total_layers=%u, num_mtp_modules=%d, block_num=%zu, "
                     "allows storing %zu tokens, total_block_size=%zu bytes (main=%zu + %d*propose=%zu)",
                     is_mtp,
                     total_layer_num,
                     num_mtp_modules,
                     block_num,
                     kv_cache_seq_len,
                     total_block_size_bytes,
                     score_config.pagedBlockBudgetBytes(),
                     num_mtp_modules,
                     propose_config.pagedBlockBudgetBytes());

    RTP_LLM_LOG_INFO("CacheConfig debugString(main_score_model):\n%s", score_config.debugString().c_str());
    for (size_t i = 0; i < config.mtpModuleCount(); ++i) {
        RTP_LLM_LOG_INFO(
            "CacheConfig debugString(sub_propose_model[%zu]):\n%s", i, config.mtpModule(i).debugString().c_str());
    }

    return config;
}

CacheConfig CacheConfigCreator::createDecodeWarmupConfig(const ModelConfig&       model_config,
                                                         const ParallelismConfig& parallelism_config,
                                                         const KVCacheConfig&     kv_cache_config,
                                                         int                      gen_num_per_cycle) {
    CacheConfig config  = createConfigFromDescs(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
    config.linear_step_ = kv_cache_config.linear_step;
    setupKernelSeqSize(config, kv_cache_config, "decode warmup");
    config.projectAssemblyBlockCounts(1);
    return config;
}

CacheConfig
CacheConfigCreator::projectBlockCounts(const CacheConfig& config, uint32_t block_count_basis, bool validate_basis) {
    if (validate_basis) {
        RTP_LLM_CHECK_WITH_INFO(block_count_basis > 0, "rank-synchronized block count basis must be positive");
        RTP_LLM_CHECK_WITH_INFO(config.block_count_basis_ > 0,
                                "rank-local CacheConfig must have a positive block count basis");
        RTP_LLM_CHECK_WITH_INFO(block_count_basis <= config.block_count_basis_,
                                "rank-synchronized block count basis %u exceeds rank-local basis %u",
                                block_count_basis,
                                config.block_count_basis_);
    }
    CacheConfig result = config;
    result.mtp_sub_configs_.clear();
    result.mtp_sub_configs_.reserve(config.mtp_sub_configs_.size());
    for (const auto& sub_config : config.mtp_sub_configs_) {
        RTP_LLM_CHECK_WITH_INFO(sub_config != nullptr, "CacheConfig cannot project a null MTP module");
        result.mtp_sub_configs_.push_back(
            std::make_shared<const CacheConfig>(projectBlockCounts(*sub_config, block_count_basis, validate_basis)));
    }
    result.projectAssemblyBlockCounts(block_count_basis);
    return result;
}

CacheConfig CacheConfigCreator::withRankSynchronizedBlockCountBasis(const CacheConfig& config,
                                                                    uint32_t           block_count_basis) {
    return projectBlockCounts(config, block_count_basis, /*validate_basis=*/true);
}

}  // namespace rtp_llm
