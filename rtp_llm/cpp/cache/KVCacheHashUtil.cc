#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   const CacheConfig&      cache_config) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    for (int i = 0; i < batch_size; ++i) {
        auto* token_ids = complete_token_ids->data(i);
        batch_kv_cache_resource->cacheResource(i).requestPrefix().rebuild(token_ids, static_cast<size_t>(seq_len));
        for (const auto& group : cache_config.topology().groups()) {
            const int span           = static_cast<int>(group.seq_size_per_block);
            const int desired_blocks = (seq_len + span - 1) / span;
            batch_kv_cache_resource->clearCacheKeys(i, group.tag);

            int64_t rolling_hash = 0;
            for (int index = 0; index < desired_blocks; ++index) {
                const int pos       = index * span;
                const int block_len = std::min(span, seq_len - pos);
                rolling_hash = rtp_llm::hashInt64Array(rolling_hash, token_ids + pos, token_ids + pos + block_len);
                batch_kv_cache_resource->pushBackCacheKey(i, group.tag, rolling_hash);
            }
            auto& resource = batch_kv_cache_resource->cacheResource(i);
            resource.setLastBlockAligned(group.tag, seq_len % span == 0);
            resource.ensureLinearBlockDependencies(group.tag);
        }
    }
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     const CacheConfig&      cache_config) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    for (int i = 0; i < batch_size; ++i) {
        auto* token_ids = complete_token_ids->data(i);
        batch_kv_cache_resource->cacheResource(i).requestPrefix().rebuild(token_ids, static_cast<size_t>(seq_len));
        for (const auto& group : cache_config.topology().groups()) {
            const int span         = static_cast<int>(group.seq_size_per_block);
            const int total_blocks = seq_len / span;
            auto&     resource     = batch_kv_cache_resource->cacheResource(i);
            auto&     keys         = resource.cacheKeys(group.tag);
            if (!resource.lastBlockAligned(group.tag) && !keys.empty()) {
                keys.pop_back();
            }
            int64_t hash      = keys.empty() ? 0 : keys.back();
            int     start_idx = static_cast<int>(keys.size());
            for (int index = start_idx; index < total_blocks; ++index) {
                const int pos = index * span;
                hash          = rtp_llm::hashInt64Array(hash, token_ids + pos, token_ids + pos + span);
                keys.push_back(hash);
            }
            resource.setLastBlockAligned(group.tag, true);
            resource.ensureLinearBlockDependencies(group.tag);
        }
    }
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource, const CacheConfig& cache_config) {
    for (const auto& group : cache_config.topology().groups()) {
        if (batch_kv_cache_resource->lastBlockAligned(group.tag)) {
            continue;
        }
        batch_kv_cache_resource->popBackAllBatchCacheKeys(group.tag);
        batch_kv_cache_resource->setLastBlockAligned(group.tag, true);
    }
}

}  // namespace rtp_llm
