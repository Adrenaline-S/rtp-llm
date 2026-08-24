#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

struct KVCacheBlockBudget {
    size_t explicit_pool_reserve_bytes = 0;
    size_t paged_block_bytes           = 0;
    size_t swa_block_bytes             = 0;
};

// Returns the largest global block count whose independent-pool backing fits
// in total_budget_bytes:
//   explicit reserve + N * paged bytes + ceil(N / linear_step) * SWA bytes.
uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget, int linear_step);

class CacheConfigCreator {
public:
    static CacheConfig createRankLocalConfig(const ModelConfig&                 model_config,
                                             const ParallelismConfig&           parallelism_config,
                                             const RuntimeConfig&               runtime_config,
                                             const KVCacheConfig&               kv_cache_config,
                                             const std::optional<WarmUpResult>& warm_up_result          = std::nullopt,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config = std::nullopt);
    static CacheConfig createRankLocalSpeculativeConfig(const ModelConfig&                 score_model_config,
                                                        const ModelConfig&                 propose_model_config,
                                                        const ParallelismConfig&           parallelism_config,
                                                        const RuntimeConfig&               runtime_config,
                                                        const KVCacheConfig&               kv_cache_config,
                                                        const SpeculativeExecutionConfig&  sp_config,
                                                        const std::optional<WarmUpResult>& warm_up_result);
    static CacheConfig createDecodeWarmupConfig(const ModelConfig&       model_config,
                                                const ParallelismConfig& parallelism_config,
                                                const KVCacheConfig&     kv_cache_config,
                                                int                      gen_num_per_cycle);

    // Reconciles only capacity. Descriptor lowering, group topology, geometry,
    // and MTP composition remain exactly as published by the rank-local creator.
    static CacheConfig withRankSynchronizedBlockCountBasis(const CacheConfig& config, uint32_t block_count_basis);

private:
    static void setupKernelSeqSize(CacheConfig& config, const KVCacheConfig& kv_cache_config, const char* config_name);
    static uint32_t    computeBlockNum(CacheConfig&                                     config,
                                       const ModelConfig&                               model_config,
                                       const RuntimeConfig&                             runtime_config,
                                       const KVCacheConfig&                             kv_cache_config,
                                       const ParallelismConfig&                         parallelism_config,
                                       const std::optional<WarmUpResult>&               warm_up_result,
                                       const std::optional<SpeculativeExecutionConfig>& sp_config);
    static void        populateGroupsFromLayerSpecs(CacheConfig&             config,
                                                    const LayerBuiltSpecs&   layer_specs,
                                                    const ModelConfig&       model_config,
                                                    const ParallelismConfig& parallelism_config);
    static void        finalizeGroupStorage(CacheConfig& config);
    static CacheConfig createConfigFromDescs(const ModelConfig&       model_config,
                                             const ParallelismConfig& parallelism_config,
                                             const KVCacheConfig&     kv_cache_config,
                                             int                      gen_num_per_cycle);
    static CacheConfig projectBlockCounts(const CacheConfig& config, uint32_t block_count_basis, bool validate_basis);

    // Removed functions moved to MemoryEvaluationHelper:
    // getDefaultRuntimeMemorySize
    // getKVCacheMemorySize
};

}  // namespace rtp_llm
