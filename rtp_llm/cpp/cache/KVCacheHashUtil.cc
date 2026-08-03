#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    const int desired_blocks = (seq_len + seq_size_per_block - 1) / seq_size_per_block;  // ceil

    for (int i = 0; i < batch_size; ++i) {
        auto* token_ids = complete_token_ids->data(i);
        for (const auto& group : batch_kv_cache_resource->groupResources(i)) {
            batch_kv_cache_resource->clearCacheKeys(i, group.tag);
            int64_t rolling_hash = 0;
            for (int index = 0; index < desired_blocks; ++index) {
                const int pos       = index * seq_size_per_block;
                const int block_len = std::min(seq_size_per_block, seq_len - pos);
                rolling_hash = rtp_llm::hashInt64Array(rolling_hash, token_ids + pos, token_ids + pos + block_len);
                batch_kv_cache_resource->pushBackCacheKey(i, group.tag, rolling_hash);
            }
            auto& resource = batch_kv_cache_resource->cacheResource(i);
            resource.setLastBlockAligned(group.tag, seq_len % seq_size_per_block == 0);
            resource.ensureLinearBlockDependencies(group.tag);
        }
    }
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    for (int i = 0; i < batch_size; ++i) {
        auto* token_ids = complete_token_ids->data(i);
        for (const auto& group : batch_kv_cache_resource->groupResources(i)) {
            auto&     resource     = batch_kv_cache_resource->cacheResource(i);
            auto&     keys         = resource.cacheKeys(group.tag);
            const int total_blocks = seq_len / seq_size_per_block;  // floor, only full blocks
            if (!resource.lastBlockAligned(group.tag) && !keys.empty()) {
                keys.pop_back();
            }
            int64_t hash      = keys.empty() ? 0 : keys.back();
            int     start_idx = static_cast<int>(keys.size());
            for (int index = start_idx; index < total_blocks; ++index) {
                const int pos = index * seq_size_per_block;
                hash          = rtp_llm::hashInt64Array(hash, token_ids + pos, token_ids + pos + seq_size_per_block);
                keys.push_back(hash);
            }
            resource.setLastBlockAligned(group.tag, true);
            resource.ensureLinearBlockDependencies(group.tag);
        }
    }
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    for (int batch_id = 0; batch_id < batch_kv_cache_resource->batchSize(); ++batch_id) {
        auto& resource = batch_kv_cache_resource->cacheResource(batch_id);
        for (const auto& group : resource.groupResources()) {
            if (resource.lastBlockAligned(group.tag)) {
                continue;
            }
            auto& keys = resource.cacheKeys(group.tag);
            RTP_LLM_CHECK_WITH_INFO(!keys.empty(), "partial block is missing its cache key, tag=%s", group.tag.c_str());
            keys.pop_back();
            resource.setLastBlockAligned(group.tag, true);
            resource.ensureLinearBlockDependencies(group.tag);
        }
    }
}

}  // namespace rtp_llm
