#include "rtp_llm/cpp/cache/SingleConfigCreator.h"

#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <numeric>

namespace rtp_llm {

namespace {

void validateDefaultSpecLayers(const KVCacheSpecPtr& spec, int64_t layer_num) {
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "single cache config requires non-null default kv_cache spec");
    RTP_LLM_CHECK_WITH_INFO(spec->tag == "default", "single cache config requires tag=default, got=%s", spec->tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(!spec->tag.empty(), "single cache config got empty kv_cache spec tag");
    RTP_LLM_CHECK_WITH_INFO(spec->layers.size() == static_cast<size_t>(layer_num),
                            "default kv_cache spec layer count %zu != num_layers %ld",
                            spec->layers.size(),
                            layer_num);
    for (int64_t i = 0; i < layer_num; ++i) {
        RTP_LLM_CHECK_WITH_INFO(spec->layers[static_cast<size_t>(i)] == static_cast<int>(i),
                                "default kv_cache spec must cover contiguous layer ids; index %ld got %d",
                                i,
                                spec->layers[static_cast<size_t>(i)]);
    }
}

KVCacheSpecPtr getDefaultSpecFromModel(const ModelConfig&       model_config,
                                       const ParallelismConfig& parallelism_config,
                                       rtp_llm::DataType        dtype) {
    RTP_LLM_CHECK_WITH_INFO(model_config.kv_cache_specs.size() == 1,
                            "single cache config requires exactly one default kv_cache spec, got %zu",
                            model_config.kv_cache_specs.size());
    auto spec = model_config.kv_cache_specs[0];
    validateDefaultSpecLayers(spec, model_config.num_layers);
    spec = spec->clone();

    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        auto* mla_spec = dynamic_cast<MLAKVCacheSpec*>(spec.get());
        RTP_LLM_CHECK_WITH_INFO(mla_spec != nullptr && spec->type == KVCacheSpecType::MultiHeadLatentAttention,
                                "default kv_cache spec must be MLAKVCacheSpec for MLA model");
        spec->local_head_num_kv  = 1;
        spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
        spec->layer_num          = static_cast<uint32_t>(model_config.num_layers);
        mla_spec->kv_lora_rank   = static_cast<uint32_t>(model_config.attn_config.kv_lora_rank);
        mla_spec->rope_head_dim  = static_cast<uint32_t>(model_config.attn_config.rope_head_dim);
    } else {
        auto* mha_spec = dynamic_cast<MHAKVCacheSpec*>(spec.get());
        RTP_LLM_CHECK_WITH_INFO(mha_spec != nullptr && spec->type == KVCacheSpecType::MultiHeadAttention,
                                "default kv_cache spec must be MHAKVCacheSpec for MHA/GQA model");
        spec->local_head_num_kv = static_cast<uint32_t>(
            (model_config.attn_config.kv_head_num % parallelism_config.get_attn_tp_size() == 0) ?
                model_config.attn_config.kv_head_num / parallelism_config.get_attn_tp_size() :
                model_config.attn_config.kv_head_num
                    / std::gcd(model_config.attn_config.kv_head_num, parallelism_config.get_attn_tp_size()));
        spec->seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
        spec->layer_num          = static_cast<uint32_t>(model_config.num_layers);
        mha_spec->size_per_head  = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    }
    spec->dtype = dtype;
    return spec;
}

}  // namespace

CacheConfig SingleConfigCreator::createSingleConfig(const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config,
                                                    bool                     is_mtp) {
    auto dtype = MemoryEvaluationHelper::getDataTypeForCache(model_config);

    auto layer_num = model_config.num_layers;

    std::vector<int> all_layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        all_layer_ids[i] = i;
    }

    CacheConfig config;
    config.layer_num          = static_cast<uint32_t>(layer_num);
    config.layer_all_num      = static_cast<uint32_t>(layer_num);
    config.block_num          = 0;
    config.seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);

    config.use_mla   = model_config.attn_config.use_mla;
    config.dtype     = dtype;
    config.is_sparse = model_config.attn_config.is_sparse;

    KVCacheSpecPtr spec = getDefaultSpecFromModel(model_config, parallelism_config, dtype);
    config.cache_specs.push_back(spec);
    config.group_types.push_back(CacheGroupType::FULL);

    // Using spec interface for block size and scale
    config.kv_block_stride_bytes = config.cache_specs[0]->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_block_stride_bytes;

    // Scale handling - no need to check dtype as scale_block_size_bytes() returns 0 if no scale support
    config.kv_scale_stride_bytes = config.cache_specs[0]->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;

    if (config.is_sparse) {
        auto indexer_dim             = model_config.attn_config.indexer_head_dim;
        config.kv_scale_stride_bytes = (indexer_dim + indexer_dim / 128 * 4) * spec->seq_size_per_block;
        config.kv_scale_size_bytes   = static_cast<size_t>(config.layer_num) * config.kv_scale_stride_bytes;
    }

    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.group_layer_num  = layer_num;  // only 1 group for SingleConfig

    // Per-layer block stride (kv + scale).
    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

    // Global layer ids are the indices used by BlockPool::convertIndexToAddr (0..N-1 in a single-model case).
    config.global_layer_ids.push_back(all_layer_ids);
    config.layer_ids.push_back(all_layer_ids);
    config.layer_to_group_id.assign(config.layer_num, 0);
    config.layer_group_types.assign(config.layer_num, CacheGroupType::FULL);
    // Populate region mapping: single group uses DEFAULT region.
    config.group_region_names.push_back(KVCacheRegionName::DEFAULT);
    const size_t region_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    config.layer_region_to_group_id.resize(config.layer_num);
    for (size_t i = 0; i < config.layer_num; i++) {
        config.layer_region_to_group_id[i].assign(region_count, -1);
        config.layer_region_to_group_id[i][static_cast<size_t>(KVCacheRegionName::DEFAULT)] = 0;
    }
    return config;
}

}  // namespace rtp_llm