#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"

#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <utility>

#include "rtp_llm/cpp/cache/DSV4CacheConfigHelper.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

bool hasDsv4KvCacheSpecs(const ModelConfig& model_config) {
    if (model_config.kv_cache_specs.empty()) {
        return false;
    }
    constexpr const char* kExpectedTags[] = {
        "csa_kv", "hca_kv", "indexer_kv", "indexer_state", "csa_state", "hca_state", "swa_kv"};
    for (const char* expected_tag : kExpectedTags) {
        const auto it = std::find_if(model_config.kv_cache_specs.begin(),
                                     model_config.kv_cache_specs.end(),
                                     [expected_tag](const auto& spec) {
                                         return spec != nullptr && spec->tag == expected_tag;
                                     });
        if (it == model_config.kv_cache_specs.end()) {
            return false;
        }
    }
    return true;
}

struct HybridPoolLayers {
    std::vector<int> full_layers;
    std::vector<int> linear_layers;
    std::vector<int> swa_layers;
};

HybridPoolLayers splitHybridPoolLayers(const ModelConfig& model_config) {
    const auto layer_num = model_config.num_layers;
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "invalid model_config.num_layers=%ld", layer_num);
    RTP_LLM_CHECK_WITH_INFO(model_config.hybrid_attention_config.hybrid_attention_types.size()
                                == static_cast<size_t>(layer_num),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            model_config.hybrid_attention_config.hybrid_attention_types.size(),
                            layer_num);

    HybridPoolLayers layers;
    layers.full_layers.reserve(static_cast<size_t>(layer_num));
    layers.linear_layers.reserve(static_cast<size_t>(layer_num));
    layers.swa_layers.reserve(static_cast<size_t>(layer_num));
    for (int i = 0; i < static_cast<int>(layer_num); ++i) {
        switch (model_config.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)]) {
            case HybridAttentionType::LINEAR:
                layers.linear_layers.push_back(i);
                break;
            case HybridAttentionType::SLIDING_WINDOW:
                layers.swa_layers.push_back(i);
                break;
            case HybridAttentionType::NONE:
            default:
                layers.full_layers.push_back(i);
                break;
        }
    }
    return layers;
}

KVCacheSpecPtr getHybridSpecByTag(const ModelConfig& model_config, const std::string& tag) {
    KVCacheSpecPtr result;
    for (const auto& spec : model_config.kv_cache_specs) {
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "hybrid-pool kv_cache_specs must not contain null specs");
        RTP_LLM_CHECK_WITH_INFO(!spec->tag.empty(), "hybrid-pool kv_cache_specs must not contain empty tags");
        if (spec->tag == tag) {
            RTP_LLM_CHECK_WITH_INFO(result == nullptr, "duplicate hybrid-pool kv_cache spec tag=%s", tag.c_str());
            result = spec;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(result != nullptr, "missing hybrid-pool kv_cache spec tag=%s", tag.c_str());
    return result->clone();
}

void prepareFullAttentionSpec(KVCacheSpecPtr            spec,
                              const ModelConfig&       model_config,
                              const ParallelismConfig& parallelism_config,
                              rtp_llm::DataType        dtype,
                              uint32_t                 layer_num) {
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        auto* mla_spec = dynamic_cast<MLAKVCacheSpec*>(spec.get());
        RTP_LLM_CHECK_WITH_INFO(mla_spec != nullptr && spec->type == KVCacheSpecType::MultiHeadLatentAttention,
                                "full kv_cache spec must be MLAKVCacheSpec for MLA model");
        spec->local_head_num_kv = 1;
        mla_spec->kv_lora_rank  = static_cast<uint32_t>(model_config.attn_config.kv_lora_rank);
        mla_spec->rope_head_dim = static_cast<uint32_t>(model_config.attn_config.rope_head_dim);
    } else {
        auto* mha_spec = dynamic_cast<MHAKVCacheSpec*>(spec.get());
        RTP_LLM_CHECK_WITH_INFO(mha_spec != nullptr && spec->type == KVCacheSpecType::MultiHeadAttention,
                                "full kv_cache spec must be MHAKVCacheSpec for MHA/GQA model");
        spec->local_head_num_kv = static_cast<uint32_t>(
            (model_config.attn_config.kv_head_num % parallelism_config.get_attn_tp_size() == 0) ?
                model_config.attn_config.kv_head_num / parallelism_config.get_attn_tp_size() :
                model_config.attn_config.kv_head_num
                    / std::gcd(model_config.attn_config.kv_head_num, parallelism_config.get_attn_tp_size()));
        mha_spec->size_per_head = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    }
    spec->dtype              = dtype;
    spec->layer_num          = layer_num;
    spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
}

void prepareLinearAttentionSpec(KVCacheSpecPtr            spec,
                                const ModelConfig&       model_config,
                                const ParallelismConfig& parallelism_config,
                                rtp_llm::DataType        dtype,
                                uint32_t                 layer_num) {
    auto* linear_spec = dynamic_cast<LinearKVCacheSpec*>(spec.get());
    RTP_LLM_CHECK_WITH_INFO(linear_spec != nullptr && spec->type == KVCacheSpecType::LinearAttention,
                            "linear kv_cache spec must be LinearKVCacheSpec");
    const auto& linear_config = model_config.linear_attention_config;
    RTP_LLM_CHECK_WITH_INFO(linear_config.linear_key_head_dim > 0 && linear_config.linear_value_head_dim > 0,
                            "invalid linear head dim");
    RTP_LLM_CHECK_WITH_INFO(linear_config.linear_conv_kernel_dim > 1,
                            "invalid linear_conv_kernel_dim=%d",
                            linear_config.linear_conv_kernel_dim);
    RTP_LLM_CHECK_WITH_INFO(linear_config.linear_num_key_heads > 0 && linear_config.linear_num_value_heads > 0,
                            "invalid linear heads");
    RTP_LLM_CHECK_WITH_INFO(linear_config.linear_key_head_dim == linear_config.linear_value_head_dim,
                            "linear head dims must match (current impl): k=%d v=%d",
                            linear_config.linear_key_head_dim,
                            linear_config.linear_value_head_dim);
    const int tp = std::max(1, static_cast<int>(parallelism_config.get_attn_tp_size()));
    linear_spec->local_num_k_heads = static_cast<uint32_t>(linear_config.linear_num_key_heads / tp);
    linear_spec->local_num_v_heads = static_cast<uint32_t>(linear_config.linear_num_value_heads / tp);
    RTP_LLM_CHECK_WITH_INFO(linear_spec->local_num_k_heads > 0 && linear_spec->local_num_v_heads > 0,
                            "invalid local heads for linear attention: k=%d v=%d tp=%d",
                            linear_spec->local_num_k_heads,
                            linear_spec->local_num_v_heads,
                            tp);
    spec->local_head_num_kv = static_cast<uint32_t>(std::max(
        1,
        (linear_config.linear_num_value_heads > 1) ?
            static_cast<int>(linear_config.linear_num_value_heads / parallelism_config.get_attn_tp_size()) :
            static_cast<int>(linear_config.linear_num_value_heads)));
    spec->dtype                   = dtype;
    spec->layer_num               = layer_num;
    spec->seq_size_per_block      = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    linear_spec->head_k_dim       = static_cast<uint32_t>(linear_config.linear_key_head_dim);
    linear_spec->head_v_dim       = static_cast<uint32_t>(linear_config.linear_value_head_dim);
    linear_spec->conv_kernel_dim  = static_cast<uint32_t>(linear_config.linear_conv_kernel_dim);
    linear_spec->ssm_state_dtype  = linear_config.ssm_state_dtype;
    linear_spec->conv_state_dtype = linear_config.conv_state_dtype;
}

void appendGroup(CacheConfig&            config,
                 const std::vector<int>& layer_ids,
                 CacheGroupType          group_type,
                 KVCacheSpecPtr          spec,
                 KVCacheRegionName       region_name = KVCacheRegionName::DEFAULT,
                 std::string             tag = "") {
    if (layer_ids.empty()) {
        return;
    }
    config.global_layer_ids.push_back(layer_ids);
    config.layer_ids.push_back(layer_ids);
    config.cache_specs.push_back(spec);
    config.group_types.push_back(group_type);
    config.group_region_names.push_back(region_name);
    config.group_tags.push_back(std::move(tag));
}

struct Dsv4ExpectedGroupRoute {
    const char*       tag;
    KVCacheRegionName region_name;
};

constexpr std::array<Dsv4ExpectedGroupRoute, 7> kDsv4ExpectedGroupRoutes = {
    Dsv4ExpectedGroupRoute{"csa_kv", KVCacheRegionName::CSA_KV},
    Dsv4ExpectedGroupRoute{"hca_kv", KVCacheRegionName::HCA_KV},
    Dsv4ExpectedGroupRoute{"indexer_kv", KVCacheRegionName::INDEXER_KV},
    Dsv4ExpectedGroupRoute{"indexer_state", KVCacheRegionName::INDEXER_STATE},
    Dsv4ExpectedGroupRoute{"csa_state", KVCacheRegionName::CSA_STATE},
    Dsv4ExpectedGroupRoute{"hca_state", KVCacheRegionName::HCA_STATE},
    Dsv4ExpectedGroupRoute{"swa_kv", KVCacheRegionName::SWA_KV},
};

void validateDsv4TagRegionMappings(const CacheConfig& config) {
    RTP_LLM_CHECK_WITH_INFO(config.group_tags.size() == kDsv4ExpectedGroupRoutes.size(),
                            "DSV4 group_tags size %zu != %zu",
                            config.group_tags.size(),
                            kDsv4ExpectedGroupRoutes.size());
    RTP_LLM_CHECK_WITH_INFO(config.group_region_names.size() == kDsv4ExpectedGroupRoutes.size(),
                            "DSV4 group_region_names size %zu != %zu",
                            config.group_region_names.size(),
                            kDsv4ExpectedGroupRoutes.size());
    RTP_LLM_CHECK_WITH_INFO(config.cache_specs.size() == kDsv4ExpectedGroupRoutes.size(),
                            "DSV4 cache_specs size %zu != %zu",
                            config.cache_specs.size(),
                            kDsv4ExpectedGroupRoutes.size());

    const size_t region_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    RTP_LLM_CHECK_WITH_INFO(config.layer_region_to_group_id.size() >= config.layer_num,
                            "DSV4 layer_region_to_group_id size %zu < layer_num %u",
                            config.layer_region_to_group_id.size(),
                            config.layer_num);
    RTP_LLM_CHECK_WITH_INFO(config.layer_tag_to_group_id.size() >= config.layer_num,
                            "DSV4 layer_tag_to_group_id size %zu < layer_num %u",
                            config.layer_tag_to_group_id.size(),
                            config.layer_num);

    for (size_t gid = 0; gid < kDsv4ExpectedGroupRoutes.size(); ++gid) {
        const auto& expected = kDsv4ExpectedGroupRoutes[gid];
        RTP_LLM_CHECK_WITH_INFO(config.group_tags[gid] == expected.tag,
                                "DSV4 group %zu tag %s != %s",
                                gid,
                                config.group_tags[gid].c_str(),
                                expected.tag);
        RTP_LLM_CHECK_WITH_INFO(config.group_region_names[gid] == expected.region_name,
                                "DSV4 group %zu has unexpected legacy region id %d",
                                gid,
                                static_cast<int>(config.group_region_names[gid]));
        RTP_LLM_CHECK_WITH_INFO(config.cache_specs[gid] != nullptr, "DSV4 cache_specs[%zu] is null", gid);
        RTP_LLM_CHECK_WITH_INFO(config.cache_specs[gid]->tag == expected.tag,
                                "DSV4 cache_specs[%zu] tag %s != %s",
                                gid,
                                config.cache_specs[gid]->tag.c_str(),
                                expected.tag);
        RTP_LLM_CHECK_WITH_INFO(config.cache_specs[gid]->layers == config.global_layer_ids[gid],
                                "DSV4 cache_specs[%zu] layers differ from global_layer_ids",
                                gid);

        const auto region_id = static_cast<size_t>(expected.region_name);
        RTP_LLM_CHECK_WITH_INFO(
            region_id < region_count, "DSV4 group %zu invalid legacy region id %zu", gid, region_id);
        for (int layer_id : config.global_layer_ids[gid]) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config.layer_num,
                                    "DSV4 group %zu invalid layer id %d",
                                    gid,
                                    layer_id);
            const auto layer = static_cast<size_t>(layer_id);
            RTP_LLM_CHECK_WITH_INFO(config.layer_region_to_group_id[layer].size() == region_count,
                                    "DSV4 layer %zu region route size %zu != %zu",
                                    layer,
                                    config.layer_region_to_group_id[layer].size(),
                                    region_count);
            const int region_gid = config.layer_region_to_group_id[layer][region_id];
            const int tag_gid    = config.groupIdForLayerTag(layer_id, expected.tag);
            RTP_LLM_CHECK_WITH_INFO(region_gid == static_cast<int>(gid),
                                    "DSV4 layer %d region route for group %zu points to %d",
                                    layer_id,
                                    gid,
                                    region_gid);
            RTP_LLM_CHECK_WITH_INFO(tag_gid == region_gid,
                                    "DSV4 layer %d tag route %s points to %d but region points to %d",
                                    layer_id,
                                    expected.tag,
                                    tag_gid,
                                    region_gid);
        }
    }
}

void populateDefaultRegionMappings(CacheConfig& config) {
    config.layer_to_group_id.assign(config.layer_num, -1);
    config.layer_to_group_ids.assign(config.layer_num, std::vector<int>());
    config.layer_group_types.assign(config.layer_num, CacheGroupType::FULL);

    const size_t region_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    config.layer_region_to_group_id.assign(config.layer_num, std::vector<int>(region_count, -1));
    config.layer_tag_to_group_id.assign(config.layer_num, std::map<std::string, int>());

    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        const auto region_name =
            gid < config.group_region_names.size() ? config.group_region_names[gid] : KVCacheRegionName::DEFAULT;
        const auto region_id = static_cast<size_t>(region_name);
        RTP_LLM_CHECK_WITH_INFO(
            region_id < region_count, "invalid hybrid-pool region name %zu for group %zu", region_id, gid);
        for (int layer_id : config.layer_ids[gid]) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config.layer_num,
                                    "invalid hybrid-pool layer id %d",
                                    layer_id);
            const auto layer = static_cast<size_t>(layer_id);
            config.layer_to_group_ids[layer].push_back(static_cast<int>(gid));
            config.layer_region_to_group_id[layer][region_id] = static_cast<int>(gid);
            if (gid < config.group_tags.size() && !config.group_tags[gid].empty()) {
                config.layer_tag_to_group_id[layer][config.group_tags[gid]] = static_cast<int>(gid);
            }
            if (region_name == KVCacheRegionName::DEFAULT) {
                config.layer_to_group_id[layer] = static_cast<int>(gid);
                config.layer_group_types[layer] = config.group_types[gid];
            }
        }
    }

    const auto swa_region_id = static_cast<size_t>(KVCacheRegionName::SWA_KV);
    for (size_t layer = 0; layer < static_cast<size_t>(config.layer_num); ++layer) {
        if (config.layer_to_group_id[layer] >= 0) {
            continue;
        }
        int fallback_gid = -1;
        if (swa_region_id < config.layer_region_to_group_id[layer].size()) {
            fallback_gid = config.layer_region_to_group_id[layer][swa_region_id];
        }
        if (fallback_gid < 0 && !config.layer_to_group_ids[layer].empty()) {
            fallback_gid = config.layer_to_group_ids[layer].back();
        }
        RTP_LLM_CHECK_WITH_INFO(fallback_gid >= 0, "missing hybrid-pool group mapping for layer %zu", layer);
        config.layer_to_group_id[layer] = fallback_gid;
        if (static_cast<size_t>(fallback_gid) < config.group_types.size()) {
            config.layer_group_types[layer] = config.group_types[static_cast<size_t>(fallback_gid)];
        }
    }
}

size_t kernelBlocksPerKvBlockForGroup(const CacheConfig& config, size_t group_id) {
    RTP_LLM_CHECK_WITH_INFO(group_id < config.group_types.size(),
                            "missing cache group type for group %zu (group_types.size=%zu)",
                            group_id,
                            config.group_types.size());
    const bool is_full = config.group_types[group_id] == CacheGroupType::FULL;
    return is_full ? config.kernelBlocksPerKvBlock() : 1;
}

void setupIndependentPoolSizes(CacheConfig& config, bool is_mtp) {
    config.use_independent_block_pools = true;
    const auto group_num               = static_cast<size_t>(config.groupNums());
    config.group_block_nums.resize(group_num, 0);
    config.group_seq_size_per_block.resize(group_num, config.seq_size_per_block);
    config.group_kv_block_stride_bytes.resize(group_num, 0);
    config.group_kv_scale_stride_bytes.resize(group_num, 0);
    config.group_block_size_bytes.resize(group_num, 0);

    size_t   max_kv_stride           = 0;
    size_t   max_scale_stride        = 0;
    size_t   total_kv_block_bytes    = 0;
    size_t   total_scale_block_bytes = 0;
    size_t   swa_kv_block_bytes      = 0;
    size_t   swa_scale_block_bytes   = 0;
    size_t   state_kv_block_bytes    = 0;
    size_t   state_scale_block_bytes = 0;
    uint32_t max_group_layers        = 0;

    config.layer_to_block_stride_bytes.assign(config.layer_all_num, 0);
    for (size_t gid = 0; gid < config.cache_specs.size(); ++gid) {
        const auto& spec = config.cache_specs[gid];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", gid);
        const auto   layer_count                = static_cast<uint32_t>(config.global_layer_ids[gid].size());
        const size_t kernel_kv_stride           = spec->block_size_bytes();
        const auto   kernel_scale               = spec->scale_block_size_bytes();
        const size_t group_bpk                  = kernelBlocksPerKvBlockForGroup(config, gid);
        const size_t kv_stride                  = kernel_kv_stride * group_bpk;
        const size_t scale_stride               = kernel_scale * group_bpk;
        config.group_kv_block_stride_bytes[gid] = kv_stride;
        config.group_kv_scale_stride_bytes[gid] = scale_stride;
        config.group_block_size_bytes[gid]      = static_cast<size_t>(layer_count) * (kv_stride + scale_stride);
        const auto region =
            gid < config.group_region_names.size() ? config.group_region_names[gid] : KVCacheRegionName::DEFAULT;
        const bool is_state = isStateRegion(region);
        const bool is_swa   = gid < config.group_types.size() && config.group_types[gid] == CacheGroupType::SWA;
        if (is_state) {
            state_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            state_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        } else if (is_swa) {
            swa_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            swa_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        } else {
            total_kv_block_bytes += static_cast<size_t>(layer_count) * kv_stride;
            total_scale_block_bytes += static_cast<size_t>(layer_count) * scale_stride;
        }
        max_kv_stride    = std::max(max_kv_stride, kv_stride);
        max_scale_stride = std::max(max_scale_stride, scale_stride);
        max_group_layers = std::max(max_group_layers, layer_count);

        for (int layer_id : config.global_layer_ids[gid]) {
            config.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)] =
                static_cast<int>(kv_stride + scale_stride);
        }
    }

    config.group_layer_num         = static_cast<int>(std::max<uint32_t>(1, max_group_layers));
    config.kv_block_stride_bytes   = max_kv_stride;
    config.kv_scale_stride_bytes   = max_scale_stride;
    config.kv_block_size_bytes     = total_kv_block_bytes;
    config.kv_scale_size_bytes     = total_scale_block_bytes;
    config.swa_block_size_bytes    = swa_kv_block_bytes + swa_scale_block_bytes;
    config.state_block_size_bytes  = state_kv_block_bytes + state_scale_block_bytes;
    const size_t paged_block_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    if (paged_block_bytes == 0) {
        RTP_LLM_CHECK_WITH_INFO(is_mtp && config.use_typed_cache_regions,
                                "hybrid-pool paged groups produced zero block bytes");
        config.kv_block_size_bytes = 1;
        config.kv_scale_size_bytes = 0;
        config.block_size_bytes    = 1;
    } else {
        config.block_size_bytes = paged_block_bytes;
    }
    config.fixed_pool_reserve_bytes = 0;
}

void populateHybridAttentionGroups(CacheConfig&             config,
                                   const ModelConfig&       model_config,
                                   const ParallelismConfig& parallelism_config) {
    const auto dtype  = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    const auto layers = splitHybridPoolLayers(model_config);

    config.cache_specs.clear();
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.group_types.clear();
    config.group_region_names.clear();
    config.group_tags.clear();

    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_specs.size() == 2,
                            "hybrid-pool attention requires exactly full/linear kv_cache_specs, got %zu",
                            model_config.kv_cache_specs.size());
    auto full_spec   = getHybridSpecByTag(model_config, "full");
    auto swa_spec    = full_spec->clone();
    auto linear_spec = getHybridSpecByTag(model_config, "linear");
    prepareFullAttentionSpec(full_spec,
                             model_config,
                             parallelism_config,
                             dtype,
                             static_cast<uint32_t>(layers.full_layers.size()));
    prepareFullAttentionSpec(swa_spec,
                             model_config,
                             parallelism_config,
                             dtype,
                             static_cast<uint32_t>(layers.swa_layers.size()));
    prepareLinearAttentionSpec(
        linear_spec, model_config, parallelism_config, dtype, static_cast<uint32_t>(layers.linear_layers.size()));

    appendGroup(config, layers.full_layers, CacheGroupType::FULL, full_spec);
    appendGroup(config, layers.swa_layers, CacheGroupType::SWA, swa_spec);
    appendGroup(config, layers.linear_layers, CacheGroupType::LINEAR, linear_spec);
}

void setupGroupCounts(CacheConfig& config) {
    config.full_group_num   = 0;
    config.swa_group_num    = 0;
    config.linear_group_num = 0;
    config.linear_fixed_cap = 0;
    for (auto group_type : config.group_types) {
        if (group_type == CacheGroupType::FULL) {
            ++config.full_group_num;
        } else if (group_type == CacheGroupType::SWA) {
            ++config.swa_group_num;
        } else {
            ++config.linear_group_num;
        }
    }
}

CacheConfig createHybridAttentionPoolConfig(const ModelConfig&       model_config,
                                            const ParallelismConfig& parallelism_config,
                                            const KVCacheConfig&     kv_cache_config,
                                            bool                     is_mtp,
                                            int                      gen_num_per_cycle) {
    const auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(model_config.num_layers);
    config.layer_all_num      = config.layer_num;
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    config.use_mla            = model_config.attn_config.use_mla;
    config.dtype              = dtype;
    config.linear_step        = 1;
    config.is_sparse          = model_config.attn_config.is_sparse;

    RTP_LLM_CHECK_WITH_INFO(model_config.attn_config.layer_compress_ratios.empty() || hasDsv4KvCacheSpecs(model_config),
                            "DSV4 cache config requires model_config.kv_cache_specs; "
                            "layer_compress_ratios fallback is disabled");

    if (hasDsv4KvCacheSpecs(model_config)) {
        DSV4CacheConfigHelper::applyConfig(
            config, model_config, parallelism_config, kv_cache_config, gen_num_per_cycle);
    } else {
        RTP_LLM_CHECK_WITH_INFO(model_config.hybrid_attention_config.enable_hybrid_attention,
                                "HybridPoolConfigCreator requires DSV4 kv_cache_specs or hybrid attention");
        populateHybridAttentionGroups(config, model_config, parallelism_config);
    }

    RTP_LLM_CHECK_WITH_INFO(!config.cache_specs.empty(), "hybrid-pool config produced no cache specs");
    const bool is_dsv4_config = hasDsv4KvCacheSpecs(model_config);
    setupGroupCounts(config);
    populateDefaultRegionMappings(config);
    if (is_dsv4_config) {
        validateDsv4TagRegionMappings(config);
    }
    setupIndependentPoolSizes(config, is_mtp);
    if (is_dsv4_config) {
        config.dsv4_fixed_pool_blocks     = kv_cache_config.dsv4_fixed_pool_blocks;
        config.dsv4_hca_state_pool_blocks = kv_cache_config.dsv4_hca_state_pool_blocks;
    }
    return config;
}

}  // namespace

CacheConfig HybridPoolConfigCreator::createConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  bool                     is_mtp,
                                                  int                      gen_num_per_cycle) {
    return createHybridAttentionPoolConfig(
        model_config, parallelism_config, kv_cache_config, is_mtp, gen_num_per_cycle);
}

}  // namespace rtp_llm
