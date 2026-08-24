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
    if (budget.paged_block_bytes == 0 && budget.swa_block_bytes == 0 && budget.explicit_pool_reserve_bytes > 0) {
        budget.paged_block_bytes = config.pagedBlockBudgetBytes();
    }
    return budget;
}

std::pair<size_t, size_t> canonicalBlockGranularity(const CacheConfig& main, const CacheConfig& draft) {
    size_t     cache_key_tokens      = 0;
    size_t     kernel_tokens         = 0;
    const auto include_active_groups = [&](const CacheConfig& config) {
        for (const auto& group : config.groups()) {
            if (group.layer_ids.empty()) {
                continue;
            }
            cache_key_tokens = std::gcd(cache_key_tokens, group.seq_size_per_block);
            kernel_tokens    = std::gcd(kernel_tokens, group.kernel_seq_size_per_block);
        }
    };
    include_active_groups(main);
    include_active_groups(draft);
    RTP_LLM_CHECK_WITH_INFO(cache_key_tokens > 0 && kernel_tokens > 0,
                            "speculative cache requires positive canonical block granularity");
    return {cache_key_tokens, kernel_tokens};
}

void validateCanonicalBlockGranularity(const CacheConfig& config) {
    const auto validate_groups = [](const CacheConfig& source, size_t cache_key_tokens, size_t kernel_tokens) {
        for (const auto& group : source.groups()) {
            if (group.layer_ids.empty()) {
                continue;
            }
            RTP_LLM_CHECK_WITH_INFO(
                group.seq_size_per_block % cache_key_tokens == 0,
                "cache group tag=%s pool granularity=%zu is not divisible by canonical cache-key granularity=%zu",
                group.tag.c_str(),
                group.seq_size_per_block,
                cache_key_tokens);
            RTP_LLM_CHECK_WITH_INFO(
                group.kernel_seq_size_per_block % kernel_tokens == 0,
                "cache group tag=%s kernel granularity=%zu is not divisible by canonical kernel granularity=%zu",
                group.tag.c_str(),
                group.kernel_seq_size_per_block,
                kernel_tokens);
        }
    };
    validate_groups(config, config.cacheKeyBlockTokens(), config.kernelBlockTokens());
    for (size_t module_index = 0; module_index < config.mtpModuleCount(); ++module_index) {
        validate_groups(config.mtpModule(module_index), config.cacheKeyBlockTokens(), config.kernelBlockTokens());
    }
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

    size_t compatibility_seq_size    = 0;
    size_t compatibility_kernel_size = 0;
    for (const auto& group : groups) {
        compatibility_seq_size    = std::gcd(compatibility_seq_size, group.seq_size_per_block);
        compatibility_kernel_size = std::gcd(compatibility_kernel_size, group.kernel_seq_size_per_block);
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

void validatePerGroupDescs(const ModelConfig& model_config, uint32_t kernel_tokens_per_block, int gen_num_per_cycle) {
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

}  // namespace

void CacheConfigCreator::populateGroupsFromLayerSpecs(CacheConfig&             config,
                                                      const LayerBuiltSpecs&   layer_specs,
                                                      const ModelConfig&       model_config,
                                                      const ParallelismConfig& parallelism_config) {
    RTP_LLM_CHECK_WITH_INFO(layer_specs.size() == static_cast<size_t>(config.mainLayerCount()),
                            "cache layer spec count %zu != layer_num %u",
                            layer_specs.size(),
                            config.mainLayerCount());

    std::map<std::string, CacheGroup> groups_by_tag;
    std::vector<std::string>          ordered_tags;
    std::vector<CacheLayer>           layers(static_cast<size_t>(config.mainLayerCount()));
    for (uint32_t layer_id = 0; layer_id < config.mainLayerCount(); ++layer_id) {
        const auto& specs = layer_specs[layer_id];
        RTP_LLM_CHECK_WITH_INFO(!specs.empty(), "cache layer %u has no specs", layer_id);
        std::set<std::string> layer_tags;
        for (const auto& built : specs) {
            RTP_LLM_CHECK_WITH_INFO(!built.tag.empty(), "cache layer %u has empty spec tag", layer_id);
            RTP_LLM_CHECK_WITH_INFO(
                built.spec != nullptr, "cache layer %u tag=%s has null spec", layer_id, built.tag.c_str());
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
    size_t                total_kv_block_bytes    = 0;
    size_t                total_scale_block_bytes = 0;
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
            (group.policy.group_type == CacheGroupType::FULL || group.policy.group_type == CacheGroupType::LINEAR)
            && group.policy.explicit_block_num == 0;
        if (contributes_to_paged_budget) {
            total_kv_block_bytes += static_cast<size_t>(layer_count) * group.kv_block_stride_bytes;
            total_scale_block_bytes += static_cast<size_t>(layer_count) * group.kv_scale_stride_bytes;
        }
        const bool opaque =
            group.spec->type == KVCacheSpecType::OpaqueKV || group.spec->type == KVCacheSpecType::OpaqueState;
        config.uses_typed_cache_regions_   = config.usesTypedCacheRegions() || opaque;
        config.uses_opaque_kv_cache_store_ = config.usesOpaqueKVCacheStore() || opaque;
        config.is_sparse_                  = config.isSparse() || group.spec->type == KVCacheSpecType::OpaqueKV;
    }

    const size_t paged_block_bytes = total_kv_block_bytes + total_scale_block_bytes;
    if (paged_block_bytes == 0) {
        const bool all_groups_have_explicit_capacity =
            !groups.empty() && std::all_of(groups.begin(), groups.end(), [](const CacheGroup& group) {
                return group.policy.explicit_block_num > 0;
            });
        RTP_LLM_CHECK_WITH_INFO(config.usesTypedCacheRegions() || all_groups_have_explicit_capacity,
                                "cache paged groups produced zero block bytes without explicit capacity");
        config.paged_block_budget_bytes_ = 1;
    } else {
        config.paged_block_budget_bytes_ = paged_block_bytes;
    }
    config.explicit_pool_reserve_bytes_ = 0;
    config.disables_decode_first_malloc_device_reuse_ =
        config.disablesDecodeFirstMallocDeviceReuse() || config.usesOpaqueKVCacheStore();
    config.replaceAssemblyTopology(std::move(groups), config.layers());
    const auto& finalized_groups   = config.groups();
    config.cache_key_block_tokens_ = 0;
    config.kernel_block_tokens_    = 0;
    for (const auto& group : finalized_groups) {
        config.cache_key_block_tokens_ = std::gcd(config.cacheKeyBlockTokens(), group.seq_size_per_block);
        config.kernel_block_tokens_    = std::gcd(config.kernelBlockTokens(), group.kernel_seq_size_per_block);
    }
    config.replaceAssemblyGroupBlockLayout(group_block_nums, group_kv_strides, group_scale_strides);
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

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  int                      gen_num_per_cycle) {
    const auto    dtype                  = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    constexpr int kDefaultKvCacheSeqSize = 64;
    const bool    has_seq_override =
        kv_cache_config.seq_size_per_block > 0 && kv_cache_config.seq_size_per_block != kDefaultKvCacheSeqSize;
    const auto physical_tokens_per_block = has_seq_override ?
                                               static_cast<uint32_t>(kv_cache_config.seq_size_per_block) :
                                               static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    const auto kernel_tokens_per_block   = kv_cache_config.kernel_seq_size_per_block > 0 ?
                                               static_cast<uint32_t>(kv_cache_config.kernel_seq_size_per_block) :
                                               physical_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block > 0, "cache seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block > 0, "cache kernel_seq_size_per_block must be > 0");
    (void)PoolBlockToKernelBlockProjection(physical_tokens_per_block, kernel_tokens_per_block);
    validatePerGroupDescs(model_config, kernel_tokens_per_block, gen_num_per_cycle);

    CacheConfig config;
    config.main_layer_count_       = static_cast<uint32_t>(model_config.num_layers);
    config.cache_key_block_tokens_ = 0;
    config.kernel_block_tokens_    = 0;
    config.uses_mla_               = model_config.attn_config.use_mla;
    config.dtype_                  = dtype;
    config.linear_step_            = 1;
    config.is_sparse_              = model_config.attn_config.is_sparse;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = physical_tokens_per_block;
    ctx.kernel_tokens_per_block = kernel_tokens_per_block;
    ctx.attn_config             = &model_config.attn_config;
    ctx.linear_attention_config = &model_config.linear_attention_config;
    ctx.parallelism_config      = &parallelism_config;
    ctx.gen_num_per_cycle       = static_cast<uint32_t>(gen_num_per_cycle);
    auto layer_specs = buildLayerSpecsFromDescs(model_config.kv_cache_spec_descs, ctx, model_config.num_layers);

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
    finalizeGroupStorage(config);
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

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config = createBasicConfig(model_config, parallelism_config, kv_cache_config, 0);

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

CacheConfig CacheConfigCreator::createSpConfig(const ModelConfig&                 draft_model_config,
                                               const ParallelismConfig&           parallelism_config,
                                               const RuntimeConfig&               runtime_config,
                                               const KVCacheConfig&               kv_cache_config,
                                               const SpeculativeExecutionConfig&  sp_config,
                                               const std::optional<WarmUpResult>& warm_up_result) {
    CacheConfig config =
        createBasicConfig(draft_model_config, parallelism_config, kv_cache_config, sp_config.gen_num_per_cycle);
    config.linear_step_ = kv_cache_config.linear_step;
    setupKernelSeqSize(config, kv_cache_config, "draft");

    const uint32_t block_num = computeBlockNum(
        config, draft_model_config, runtime_config, kv_cache_config, parallelism_config, warm_up_result, sp_config);
    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "draft kv cache needs at least 1 block");
    config = projectBlockCounts(config, block_num, /*validate_basis=*/false);
    return config;
}

CacheConfig CacheConfigCreator::mergeSpConfig(const CacheConfig&                 main_config,
                                              const CacheConfig&                 draft_config,
                                              const ModelConfig&                 main_model_config,
                                              const ParallelismConfig&           parallelism_config,
                                              const RuntimeConfig&               runtime_config,
                                              const KVCacheConfig&               kv_cache_config,
                                              const SpeculativeExecutionConfig&  sp_config,
                                              const std::optional<WarmUpResult>& warm_up_result) {
    RTP_LLM_CHECK_WITH_INFO(main_config.blockCountBasis() > 0, "mergeSpConfig requires a rank-local main config");
    RTP_LLM_CHECK_WITH_INFO(draft_config.blockCountBasis() > 0, "mergeSpConfig requires a rank-local draft config");
    RTP_LLM_CHECK_WITH_INFO(main_config.mtpModuleCount() == 0,
                            "mergeSpConfig main config must not already contain draft modules");
    RTP_LLM_CHECK_WITH_INFO(draft_config.mtpModuleCount() == 0,
                            "mergeSpConfig draft config must not contain nested draft modules");

    CacheConfig main       = projectBlockCounts(main_config, 0, /*validate_basis=*/false);
    CacheConfig draft      = projectBlockCounts(draft_config, 0, /*validate_basis=*/false);
    const int   joint_step = std::max(1, kv_cache_config.linear_step);
    main.linear_step_      = joint_step;
    draft.linear_step_     = joint_step;

    // MTP contributes one one-layer module per proposal step. EAGLE and
    // DSpARK each contribute one multi-layer draft model; their gamma is a
    // proposal width, not an independent-module count. Other speculative
    // modes keep the historical single draft-model topology.
    const int num_mtp_modules = sp_config.type == SP_TYPE_MTP ? sp_config.gen_num_per_cycle : 1;
    RTP_LLM_CHECK_WITH_INFO(num_mtp_modules > 0, "speculative cache requires at least one propose module");

    uint32_t total_layer_num = main.mainLayerCount();
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_layer_num += draft.mainLayerCount();
    }

    size_t total_block_size_bytes = main.pagedBlockBudgetBytes();
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_block_size_bytes += draft.pagedBlockBudgetBytes();
    }

    const auto [canonical_cache_key_tokens, canonical_kernel_tokens] = canonicalBlockGranularity(main, draft);

    CacheConfig config                 = main;
    config.linear_step_                = joint_step;
    config.paged_block_budget_bytes_   = total_block_size_bytes;
    config.cache_key_block_tokens_     = canonical_cache_key_tokens;
    config.kernel_block_tokens_        = canonical_kernel_tokens;
    config.uses_mla_                   = main.usesMla();
    config.is_sparse_                  = main.isSparse();
    config.uses_typed_cache_regions_   = main.usesTypedCacheRegions() || draft.usesTypedCacheRegions();
    config.uses_opaque_kv_cache_store_ = main.usesOpaqueKVCacheStore() || draft.usesOpaqueKVCacheStore();
    config.disables_decode_first_malloc_device_reuse_ =
        main.disablesDecodeFirstMallocDeviceReuse() || draft.disablesDecodeFirstMallocDeviceReuse();
    const uint32_t main_layer_num = main.mainLayerCount();
    config.mtp_sub_configs_.clear();
    config.mtp_sub_configs_.reserve(static_cast<size_t>(num_mtp_modules));
    for (int m = 0; m < num_mtp_modules; ++m) {
        auto sub_cfg                     = config.composeAssemblyMTPModule(draft, m, main_layer_num);
        sub_cfg->cache_key_block_tokens_ = 0;
        sub_cfg->kernel_block_tokens_    = 0;
        for (const auto& group : sub_cfg->groups()) {
            if (group.layer_ids.empty()) {
                continue;
            }
            sub_cfg->cache_key_block_tokens_ = std::gcd(sub_cfg->cacheKeyBlockTokens(), group.seq_size_per_block);
            sub_cfg->kernel_block_tokens_    = std::gcd(sub_cfg->kernelBlockTokens(), group.kernel_seq_size_per_block);
        }
        config.mtp_sub_configs_.push_back(std::move(sub_cfg));
    }
    validateCanonicalBlockGranularity(config);

    config.projectAssemblyBlockCounts(0);
    KVCacheBlockBudget joint_budget = blockBudgetForConfig(main);
    addBlockBudget(joint_budget, blockBudgetForConfig(draft), static_cast<size_t>(num_mtp_modules));
    const size_t explicit_pool_reserve = joint_budget.explicit_pool_reserve_bytes;

    size_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, main_model_config, parallelism_config, warm_up_result, sp_config);

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
                main.explicitPoolReserveBytes() / 1024 / 1024,
                draft.explicitPoolReserveBytes() / 1024 / 1024,
                num_mtp_modules);
        }
        block_num = maxKVCacheBlockNumForBudget(kv_cache_mem_size, joint_budget, joint_step);
    }

    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    config = projectBlockCounts(config, static_cast<uint32_t>(block_num), /*validate_basis=*/false);
    config.explicit_pool_reserve_bytes_ = explicit_pool_reserve;

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.cacheKeyBlockTokens();
    RTP_LLM_LOG_INFO("Speculative CacheConfig created: total_layers=%u, num_mtp_modules=%d, block_num=%zu, "
                     "allows storing %zu tokens, total_block_size=%zu bytes (main=%zu + %d*propose=%zu)",
                     total_layer_num,
                     num_mtp_modules,
                     block_num,
                     kv_cache_seq_len,
                     total_block_size_bytes,
                     main.pagedBlockBudgetBytes(),
                     num_mtp_modules,
                     draft.pagedBlockBudgetBytes());

    RTP_LLM_LOG_INFO("CacheConfig debugString(main_score_model):\n%s", main.debugString().c_str());
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
    CacheConfig config  = createBasicConfig(model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
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

CacheConfig CacheConfigCreator::withRankSyncBlockCount(const CacheConfig& config, uint32_t block_count) {
    return projectBlockCounts(config, block_count, /*validate_basis=*/true);
}

}  // namespace rtp_llm
