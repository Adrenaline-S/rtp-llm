#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {
namespace {

// CP shard helpers: when mapper is null/passthrough, all helpers no-op.
inline CacheKeysType cpCanonicalCacheKeys(const std::shared_ptr<CPSlotMapper>& mapper, const CacheKeysType& full) {
    return (mapper && mapper->isSharded()) ? mapper->canonicalCacheKeys(full) : full;
}

inline bool
cpBlockRoundRobinGroup(const std::shared_ptr<CPSlotMapper>& mapper, const CacheConfig& config, std::string_view tag) {
    return mapper && mapper->isSharded() && mapper->blockRoundRobinGroup(config, tag);
}

inline int cpEffectiveSeqLenForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                     const CacheConfig&                   config,
                                     std::string_view                     tag,
                                     int                                  seq_len) {
    return cpBlockRoundRobinGroup(mapper, config, tag) ? mapper->effectiveSeqLenForAlloc(config, tag, seq_len) :
                                                         seq_len;
}

inline int cpLogicalSeqSizeForGroup(const std::shared_ptr<CPSlotMapper>& mapper,
                                    const CacheConfig&                   config,
                                    std::string_view                     tag,
                                    int                                  fallback) {
    return (mapper && mapper->isSharded()) ? static_cast<int>(mapper->logicalSeqSizePerBlock(config, tag)) : fallback;
}

BlockIndicesType validBlocksAfter(const BlockIndicesType& blocks, size_t begin) {
    BlockIndicesType valid;
    if (begin >= blocks.size()) {
        return valid;
    }
    valid.reserve(blocks.size() - begin);
    for (size_t i = begin; i < blocks.size(); ++i) {
        if (!isNullBlockIdx(blocks[i])) {
            valid.push_back(blocks[i]);
        }
    }
    return valid;
}

std::vector<PoolBlockId> toPoolBlockIds(const BlockIndicesType& block_indices) {
    std::vector<PoolBlockId> pool_block_ids;
    pool_block_ids.reserve(block_indices.size());
    for (const auto block_idx : block_indices) {
        RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(block_idx), "allocation result must not contain a missing pool block");
        pool_block_ids.push_back(PoolBlockId{block_idx});
    }
    return pool_block_ids;
}

BlockIndicesType encodeBindingForGroupPolicy(const GroupBlockToPoolBlockBinding& binding) {
    BlockIndicesType encoded;
    encoded.reserve(binding.size());
    for (const auto& pool_block_id : binding.snapshot()) {
        encoded.push_back(pool_block_id.has_value() ? pool_block_id->value : NULL_BLOCK_IDX);
    }
    return encoded;
}

GroupBlockToPoolBlockBinding::Snapshot decodeGroupPolicyBlocks(const BlockIndicesType& encoded) {
    GroupBlockToPoolBlockBinding::Snapshot snapshot;
    snapshot.reserve(encoded.size());
    for (const auto block_idx : encoded) {
        snapshot.push_back(isNullBlockIdx(block_idx) ? std::nullopt :
                                                       std::optional<PoolBlockId>{PoolBlockId{block_idx}});
    }
    return snapshot;
}

}  // namespace

bool HybridPoolKVCacheAllocator::skipReuseCacheGroup(std::string_view tag) const {
    return !groupStrategy(tag)->prefixReuseEnabled();
}

std::vector<std::string> HybridPoolKVCacheAllocator::independentEvictionGroupTags() const {
    std::vector<std::string> tags;
    for (const auto& group : config_.groups()) {
        if (groupStrategy(group.tag)->evictPolicy() == CacheEvictPolicy::INDEPENDENT) {
            tags.push_back(group.tag);
        }
    }
    return tags;
}

bool HybridPoolKVCacheAllocator::cpCompactSwaGroup(std::string_view                     tag,
                                                   const std::shared_ptr<CPSlotMapper>& mapper) const {
    return mapper && mapper->isSharded() && mapper->compactLastRankGroup(config_, tag);
}

int HybridPoolKVCacheAllocator::reuseCache(const CacheKeysType&                 cache_keys,
                                           BatchKVCacheResource&                kv_resource,
                                           const std::shared_ptr<CPSlotMapper>& cp_mapper) {
    const bool no_reusable_group = full_group_tags_.empty() && linear_group_tags_.empty()
                                   && std::all_of(swa_group_tags_.begin(),
                                                  swa_group_tags_.end(),
                                                  [this](const auto& tag) { return skipReuseCacheGroup(tag); });
    if (no_reusable_group) {
        return 0;
    }

    // Under cp shard, FULL groups index block_ids by cp-virtual-block units
    // (one entry covers cp_size physical blocks). LINEAR/SWA groups index by
    // raw block_size logical blocks. So when populating tail blocks for
    // LINEAR/SWA we need to scale the array length and matched-block position
    // back to the logical-block coordinate system.
    const bool                                        cp_sharded            = cp_mapper && cp_mapper->isSharded();
    const int                                         cp_scale              = cp_sharded ? cp_mapper->cpSize() : 1;
    int                                               min_full_reuse_blocks = static_cast<int>(cache_keys.size());
    std::unordered_map<std::string, BlockIndicesType> full_matched_blocks;

    for (const auto& tag : full_group_tags_) {
        CacheKeysType full_match_keys;
        if (cp_sharded) {
            full_match_keys = cache_keys;
        } else {
            GroupBlockToCacheKeyAnchor mapping(config_.seq_size_per_block,
                                               config_.group(tag).layout.seq_size_per_block);
            const size_t               complete_group_blocks = mapping.completeGroupBlockCount(cache_keys.size());
            full_match_keys.reserve(complete_group_blocks);
            for (size_t group_block_position = 0; group_block_position < complete_group_blocks;
                 ++group_block_position) {
                const auto anchor =
                    mapping.toCacheKeyAnchor(GroupBlockPosition{group_block_position}, cache_keys.size());
                full_match_keys.push_back(cache_keys[anchor.value]);
            }
        }

        auto         match_result = groupStrategy(tag)->match(full_match_keys);
        const size_t matched_cache_key_blocks =
            cp_sharded ?
                match_result.reuse_blocks :
                GroupBlockToCacheKeyAnchor(config_.seq_size_per_block, config_.group(tag).layout.seq_size_per_block)
                    .toCacheKeyPrefixLength(match_result.reuse_blocks);
        min_full_reuse_blocks = std::min(min_full_reuse_blocks, static_cast<int>(matched_cache_key_blocks));
        full_matched_blocks.emplace(tag, std::move(match_result.block_indices));
    }

    int                           group_block_position = min_full_reuse_blocks - 1;
    std::vector<BlockIdxType>     linear_tail_blocks(linear_group_tags_.size(), NULL_BLOCK_IDX);
    std::vector<BlockIndicesType> swa_tail_blocks(swa_group_tags_.size());
    const bool                    has_tail_groups = !linear_group_tags_.empty() || !swa_group_tags_.empty();
    for (; group_block_position >= 0 && has_tail_groups; --group_block_position) {
        bool                          all_tail_groups_matched = true;
        std::vector<BlockIdxType>     candidate_linear_tail_blocks(linear_group_tags_.size(), NULL_BLOCK_IDX);
        std::vector<BlockIndicesType> candidate_swa_tail_blocks(swa_group_tags_.size());
        for (size_t i = 0; i < linear_group_tags_.size(); ++i) {
            auto result = groupStrategy(linear_group_tags_[i])
                              ->matchSingleKey(cache_keys[static_cast<size_t>(group_block_position)]);
            if (result.block_indices.empty()) {
                all_tail_groups_matched = false;
                break;
            }
            candidate_linear_tail_blocks[i] = result.block_indices[0];
        }
        if (!all_tail_groups_matched) {
            continue;
        }
        for (size_t i = 0; i < swa_group_tags_.size(); ++i) {
            const auto& tag = swa_group_tags_[i];
            if (skipReuseCacheGroup(tag)) {
                continue;
            }
            auto result = groupStrategy(tag)->matchSingleKey(cache_keys[static_cast<size_t>(group_block_position)]);
            if (result.block_indices.empty()) {
                all_tail_groups_matched = false;
                break;
            }
            candidate_swa_tail_blocks[i].push_back(result.block_indices[0]);
        }
        if (all_tail_groups_matched) {
            linear_tail_blocks = std::move(candidate_linear_tail_blocks);
            swa_tail_blocks    = std::move(candidate_swa_tail_blocks);
            break;
        }
    }

    const int reuse_blocks_len =
        has_tail_groups ? std::max(group_block_position + 1, 0) : std::max(min_full_reuse_blocks, 0);
    if (reuse_blocks_len <= 0) {
        return 0;
    }

    for (const auto& tag : full_group_tags_) {
        BlockIndicesType full_blocks = full_matched_blocks.at(tag);
        const size_t     reusable_full_blocks =
            cp_sharded ?
                    static_cast<size_t>(reuse_blocks_len) :
                    GroupBlockToCacheKeyAnchor(config_.seq_size_per_block, config_.group(tag).layout.seq_size_per_block)
                    .completeGroupBlockCount(static_cast<size_t>(reuse_blocks_len));
        if (full_blocks.size() > reusable_full_blocks) {
            full_blocks.resize(reusable_full_blocks);
        }
        kv_resource.mutableBlockBinding(0, tag).assign(toPoolBlockIds(full_blocks));
    }

    // LINEAR/SWA arrays are sized in logical-block units (cp_size× larger
    // than the FULL groups' cp-virtual-block units). The matched tail block
    // corresponds to the LAST logical block in the canonical (last-rank)
    // namespace, so its index is `(reuse_blocks_len * cp_size) - 1` in
    // logical units, NOT `reuse_blocks_len - 1`.
    const int logical_reuse_len = reuse_blocks_len * cp_scale;
    for (size_t i = 0; i < linear_group_tags_.size(); ++i) {
        const auto& tag     = linear_group_tags_[i];
        auto&       binding = kv_resource.mutableBlockBinding(0, tag);
        binding.resize(static_cast<size_t>(logical_reuse_len));
        binding.bind(GroupBlockPosition{static_cast<size_t>(logical_reuse_len - 1)},
                     PoolBlockId{linear_tail_blocks[i]});
    }
    for (size_t i = 0; i < swa_group_tags_.size(); ++i) {
        const auto& tag             = swa_group_tags_[i];
        const int   group_reuse_len = cpCompactSwaGroup(tag, cp_mapper) ? reuse_blocks_len : logical_reuse_len;
        auto&       binding         = kv_resource.mutableBlockBinding(0, tag);
        binding.resize(static_cast<size_t>(group_reuse_len));
        if (skipReuseCacheGroup(tag)) {
            continue;
        }
        const size_t tail_begin =
            static_cast<size_t>(std::max(group_reuse_len - static_cast<int>(swa_tail_blocks[i].size()), 0));
        for (size_t compact_position = 0; compact_position < swa_tail_blocks[i].size(); ++compact_position) {
            binding.bind(GroupBlockPosition{tail_begin + compact_position},
                         PoolBlockId{swa_tail_blocks[i][compact_position]});
        }
    }
    return reuse_blocks_len;
}

MallocResult HybridPoolKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();

    const int   seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);
    const auto& cp_mapper      = cp_slot_mapper_;
    // A CP-sharded FULL group defines the canonical reuse unit. Topologies without
    // FULL groups use the logical cache-key block size for LINEAR/SWA matching.
    const KVCacheGroupPtr reuse_group =
        full_group_tags_.empty() ? KVCacheGroupPtr{} : groupStrategy(full_group_tags_.front());
    const int reuse_unit_tokens =
        (reuse_group && cp_mapper && cp_mapper->isSharded() ?
             cpLogicalSeqSizeForGroup(cp_mapper, config_, full_group_tags_.front(), seqSizePerBlock()) :
             seqSizePerBlock());

    const auto&                                cache_keys         = kv_resource->cacheKeys(0);
    int64_t                                    match_cost_time_us = 0;
    const size_t                               reserve_blocks     = reserveBlocksNum();
    int                                        reuse_blocks       = 0;
    std::map<std::string, BlockIndicesType>    referenced_blocks;
    std::map<std::string, size_t>              original_sizes;
    std::map<std::string, std::vector<size_t>> backfilled_positions;

    const CacheGroup* sole_full_group =
        (config_.groupNums() == 1 && full_group_tags_.size() == 1) ? &config_.group(full_group_tags_.front()) : nullptr;
    const auto sole_spec                   = sole_full_group != nullptr ? sole_full_group->layout.spec : nullptr;
    const bool ordinary_single_full_compat = !config_.use_independent_block_pools
                                             && !(cp_mapper && cp_mapper->isSharded()) && sole_full_group != nullptr
                                             && sole_full_group->policy.group_type == CacheGroupType::FULL && sole_spec
                                             && (sole_spec->type == KVCacheSpecType::MultiHeadAttention
                                                 || sole_spec->type == KVCacheSpecType::MultiHeadLatentAttention);
    const bool match_device_cache =
        malloc_info.enable_device_cache && (!ordinary_single_full_compat || reuse_group->prefixReuseEnabled());
    if (match_device_cache) {
        // CP-sharded: subsample to last-rank canonical key namespace before matching.
        CacheKeysType cp_keys = cpCanonicalCacheKeys(cp_mapper, cache_keys);
        // Always drop the last match key, CP-sharded or not. It may be a partial
        // tail; and even when it is a full block, fully reusing the input leaves
        // no prefill tokens to compute. Keeping every canonical key under CP
        // sharding is exactly that degenerate case: zero prefill tokens left.
        CacheKeysType match_keys(cp_keys.begin(), cp_keys.empty() ? cp_keys.end() : cp_keys.end() - 1);
        auto          begin_us = currentTimeUs();
        reuse_blocks           = reuseCache(match_keys, *kv_resource, cp_mapper);
        match_cost_time_us     = currentTimeUs() - begin_us;

        for (const auto& group : config_.groups()) {
            BlockIndicesType valid;
            const auto&      binding = kv_resource->blockBinding(0, group.tag);
            valid.reserve(binding.size());
            for (const auto& pool_block_id : binding.snapshot()) {
                if (pool_block_id.has_value()) {
                    valid.push_back(pool_block_id->value);
                }
            }
            if (!valid.empty()) {
                referenceBlocks(group.tag, valid);
                referenced_blocks[group.tag] = std::move(valid);
            }
        }
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
    }

    // The resource shape after reuse is the transaction's original shape.
    // Capture it before either capacity preflight so rollback never mistakes
    // reused references for newly appended allocations.
    for (const auto& group : config_.groups()) {
        original_sizes[group.tag]       = kv_resource->blocksNum(0, group.tag);
        backfilled_positions[group.tag] = {};
    }

    // Post-match capacity preflight. Device-cache matching has already run, so
    // the allocator now knows how many *new* physical blocks are required and can
    // separate "pools are momentarily full" (RETRYABLE, keeps the stream WAITING)
    // from "this request can never fit" (PERMANENT).
    const auto capacity_status =
        evaluateInitCapacity(malloc_info, reserve_blocks, InitCapacityMode::TOTAL_AND_AVAILABLE);
    if (capacity_status != MallocStatus::NONE) {
        logMallocFailure(malloc_info, "init_reserve", 0, {}, false, -1);
        rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes, backfilled_positions);
        return {false, 0, match_cost_time_us, capacity_status};
    }

    if (reserve_blocks > 0 && !hasAvailableBlocksForReserve(malloc_info, reserve_blocks)) {
        logMallocFailure(malloc_info, "init_reserve", 0, {}, false, -1);
        rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes, backfilled_positions);
        return {false, 0, match_cost_time_us, MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED};
    }

    for (const auto& group_config : config_.groups()) {
        const auto& tag           = group_config.tag;
        auto&       binding       = kv_resource->mutableBlockBinding(0, tag);
        const int   group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, common_seq_len);
        const auto& group         = groupStrategy(tag);
        // Snapshot the slot count before the call so a failure can report this
        // group's exact physical request in the error_code=602 record.
        const int blocks_before = static_cast<int>(binding.size());
        if (!group->malloc(binding, group_seq_len, malloc_info.reuse_cache, 0, &backfilled_positions[tag])) {
            logMallocFailure(
                malloc_info, "init_group_malloc", 0, tag, false, group->needBlocksNum(group_seq_len, blocks_before, 0));
            rollbackInitMalloc(*kv_resource, referenced_blocks, original_sizes, backfilled_positions);
            return {false, 0};
        }
    }

    for (int b = 1; b < batch_size; ++b) {
        for (const auto& group : config_.groups()) {
            groupStrategy(group.tag)->reference(kv_resource->mutableBlockBinding(b, group.tag),
                                                encodeBindingForGroupPolicy(kv_resource->blockBinding(0, group.tag)));
        }
    }
    return {true, reuse_blocks * reuse_unit_tokens, match_cost_time_us};
}

MallocResult HybridPoolKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto&       kv_resource  = malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper    = cp_slot_mapper_;
    const int   batch_size   = kv_resource->batchSize();
    const int   raw_seq_len  = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::map<std::string, size_t>>              batch_original_sizes(static_cast<size_t>(batch_size));
    std::vector<std::map<std::string, std::vector<size_t>>> batch_backfilled_positions(static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        for (const auto& group : config_.groups()) {
            batch_original_sizes[static_cast<size_t>(b)][group.tag]       = kv_resource->blocksNum(b, group.tag);
            batch_backfilled_positions[static_cast<size_t>(b)][group.tag] = {};
        }
    }

    bool        all_success  = true;
    int         failed_batch = -1;
    std::string failed_tag;
    int         failed_need_blocks = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (const auto& group : config_.groups()) {
            const auto& tag           = group.tag;
            auto&       binding       = kv_resource->mutableBlockBinding(b, tag);
            const int   group_seq_len = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_seq_len);
            // Snapshot the slot count before the call so a failure can report this
            // group's exact physical request in the error_code=602 record.
            const int  blocks_before = static_cast<int>(binding.size());
            const bool injected_failure =
                shouldInjectGroupAllocationFailureForTest(*kv_resource, b, tag, /*incremental=*/true);
            if (injected_failure
                || !groupStrategy(tag)->malloc(binding,
                                               group_seq_len,
                                               malloc_info.reuse_cache,
                                               reserve_step,
                                               &batch_backfilled_positions[static_cast<size_t>(b)][tag])) {
                all_success        = false;
                failed_batch       = b;
                failed_tag         = tag;
                failed_need_blocks = groupStrategy(tag)->needBlocksNum(group_seq_len, blocks_before, reserve_step);
                break;
            }
        }
        if (!all_success) {
            break;
        }
    }

    if (all_success) {
        if (!malloc_info.enable_remove_skipped_blocks) {
            return {true, 0};
        }
        for (int b = 0; b < batch_size; ++b) {
            for (const auto& group : config_.groups()) {
                groupStrategy(group.tag)->removeSkippedBlocks(
                    kv_resource->mutableBlockBinding(b, group.tag), malloc_info.reuse_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    // Emit the pool snapshot before rolling back: once the partially allocated
    // blocks go back to the pools, available_blocks no longer reflects the state
    // that caused the failure.
    logMallocFailure(malloc_info, "incremental_group_malloc", failed_batch, failed_tag, true, failed_need_blocks);
    rollbackIncrMalloc(
        *kv_resource, batch_original_sizes, batch_backfilled_positions, static_cast<size_t>(failed_batch));
    RTP_LLM_LOG_WARNING("Hybrid incrMalloc failed at batch=%d tag=%s", failed_batch, failed_tag.c_str());
    return {false, 0};
}

void HybridPoolKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;
    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }
    for (int batch_id = 0; batch_id < kv_cache_resource->batchSize(); ++batch_id) {
        for (const auto& [tag, binding] : kv_cache_resource->blocksByTag(batch_id)) {
            groupStrategy(tag)->free(encodeBindingForGroupPolicy(binding));
        }
    }
    kv_cache_resource->clearBlocks();
}

void HybridPoolKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);
    if (!shared_block_cache_) {
        return;
    }

    const auto& cp_mapper  = cp_slot_mapper_;
    const bool  cp_active  = cp_mapper && cp_mapper->isSharded();
    const int   batch_size = kv_cache_resource->batchSize();

    const CacheGroup* sole_full_group =
        (config_.groupNums() == 1 && full_group_tags_.size() == 1) ? &config_.group(full_group_tags_.front()) : nullptr;
    const auto sole_spec                   = sole_full_group != nullptr ? sole_full_group->layout.spec : nullptr;
    const bool legacy_ordinary_single_full = !config_.use_independent_block_pools && !cp_active
                                             && sole_full_group != nullptr
                                             && sole_full_group->policy.group_type == CacheGroupType::FULL && sole_spec
                                             && (sole_spec->type == KVCacheSpecType::MultiHeadAttention
                                                 || sole_spec->type == KVCacheSpecType::MultiHeadLatentAttention);
    if (legacy_ordinary_single_full) {
        if (!kv_cache_groups_[0]->prefixReuseEnabled() || batch_size == 0) {
            return;
        }
        const auto& cache_keys = kv_cache_resource->cacheKeys(/*batch_id=*/0);
        const auto& binding    = kv_cache_resource->blockBinding(/*batch_id=*/0, full_group_tags_.front());
        const auto  block_num  = std::min(cache_keys.size(), binding.size());
        if (block_num > 0) {
            BlockIndicesType blocks;
            blocks.reserve(block_num);
            for (size_t group_block_position = 0; group_block_position < block_num; ++group_block_position) {
                const auto pool_block_id = binding.lookup(GroupBlockPosition{group_block_position});
                RTP_LLM_CHECK_WITH_INFO(pool_block_id.has_value(),
                                        "ordinary FULL publication cannot contain a missing pool block");
                blocks.push_back(pool_block_id->value);
            }
            kv_cache_groups_[0]->insertIntoCache(
                CacheKeysType(cache_keys.begin(), cache_keys.begin() + block_num), blocks, insert_info.is_resident);
        }
        return;
    }

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const auto& full_keys = kv_cache_resource->cacheKeys(batch_id);
        if (full_keys.empty()) {
            continue;
        }
        const auto& full_dependencies = kv_cache_resource->cacheResource(batch_id).blockDependencies();

        if (!cp_active) {
            std::map<size_t, std::map<std::string, BlockIdxType>> groups_by_anchor;
            for (const auto& group : config_.groups()) {
                const auto& tag = group.tag;
                if (skipReuseCacheGroup(tag)) {
                    continue;
                }
                GroupBlockToCacheKeyAnchor anchor_mapping(config_.seq_size_per_block, group.layout.seq_size_per_block);
                const auto&                binding = kv_cache_resource->blockBinding(batch_id, tag);
                const size_t eligible_group_blocks = anchor_mapping.completeGroupBlockCount(full_keys.size());
                const size_t publish_count         = std::min(binding.size(), eligible_group_blocks);
                for (size_t group_block_position = 0; group_block_position < publish_count; ++group_block_position) {
                    const auto pool_block_id = binding.lookup(GroupBlockPosition{group_block_position});
                    if (!pool_block_id.has_value()) {
                        continue;
                    }
                    const auto anchor =
                        anchor_mapping.toCacheKeyAnchor(GroupBlockPosition{group_block_position}, full_keys.size());
                    groups_by_anchor[anchor.value].emplace(tag, pool_block_id->value);
                }
            }

            // Reverse iteration keeps the prefix-base key at the MRU end, exactly as before.
            for (auto anchor_it = groups_by_anchor.rbegin(); anchor_it != groups_by_anchor.rend(); ++anchor_it) {
                const size_t cache_key_position = anchor_it->first;
                const auto   dependency         = cache_key_position < full_dependencies.size() ?
                                                      full_dependencies[cache_key_position] :
                                                      BlockDependency{false, 0, static_cast<uint32_t>(cache_key_position)};
                shared_block_cache_->put(full_keys[cache_key_position],
                                         anchor_it->second,
                                         {},
                                         insert_info.is_resident,
                                         SharedBlockCache::kGpuLogicalNamespace,
                                         dependency);
            }
            continue;
        }

        // Per-group key namespace, per-(key, group) put. SharedBlockCache::put
        // merges multiple puts on the same key into one item with each group's block id
        // populated independently (NULL_BLOCK_IDX entries are skipped by the merge path).
        //
        // CP per-group key namespace: paged FULL groups use cp-subsampled (last-rank) keys
        // to align 1:1 with rank-local blocks; non-paged groups (SWA / LINEAR) keep the
        // full key sequence so their tail blocks (real entries at positions >= length-2)
        // get inserted alongside the keys that the reuseCache tail-loop later queries.
        CacheKeysType         cp_keys         = cpCanonicalCacheKeys(cp_mapper, full_keys);
        BlockDependenciesType cp_dependencies = cp_mapper->canonicalBlockDependencies(full_dependencies);
        auto                  token_ids       = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1) {
            continue;
        }
        const size_t token_len = token_ids.size() - 1;

        for (const auto& group : config_.groups()) {
            const auto& tag = group.tag;
            if (skipReuseCacheGroup(tag)) {
                continue;
            }
            const int            raw_group_seq = groupStrategy(tag)->seqSizePerBlock();
            const bool           gp_sharded    = cpBlockRoundRobinGroup(cp_mapper, config_, tag);
            const bool           compact_swa   = cpCompactSwaGroup(tag, cp_mapper);
            const bool           use_cp_keys   = cp_active && (gp_sharded || compact_swa);
            const CacheKeysType& src_keys      = use_cp_keys ? cp_keys : full_keys;
            const auto&          dependencies  = use_cp_keys ? cp_dependencies : full_dependencies;
            const auto           namespace_id =
                use_cp_keys ? SharedBlockCache::kGpuCpCanonicalNamespace : SharedBlockCache::kGpuLogicalNamespace;
            if (src_keys.empty()) {
                continue;
            }
            const int    group_seq_size  = cpLogicalSeqSizeForGroup(cp_mapper, config_, tag, raw_group_seq);
            const size_t full_blocks_num = token_len / static_cast<size_t>(group_seq_size);
            const size_t n               = std::min(src_keys.size(), full_blocks_num);
            const auto&  binding         = kv_cache_resource->blockBinding(batch_id, tag);
            const size_t loop_end        = std::min(n, binding.size());

            // Reverse iterate so prefix-base keys land at MRU end (matches non-CP path).
            for (size_t pos = loop_end; pos > 0; --pos) {
                const size_t i             = pos - 1;
                const auto   pool_block_id = binding.lookup(GroupBlockPosition{i});
                if (!pool_block_id.has_value()) {
                    continue;
                }
                std::map<std::string, BlockIdxType> groups;
                for (const auto& other_group : config_.groups()) {
                    groups.emplace(other_group.tag, other_group.tag == tag ? pool_block_id->value : NULL_BLOCK_IDX);
                }
                const auto dependency =
                    i < dependencies.size() ? dependencies[i] : BlockDependency{false, 0, static_cast<uint32_t>(i)};
                shared_block_cache_->put(src_keys[i], groups, {}, insert_info.is_resident, namespace_id, dependency);
            }
        }
    }
}

std::shared_ptr<KVCacheResource> HybridPoolKVCacheAllocator::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                            const CacheKeysType&   cache_keys,
                                                                            bool                   is_connector) {
    if (cache_keys.empty() || kvcache_resource.groupNums() <= 0) {
        return nullptr;
    }

    std::unordered_map<CacheKeyType, size_t> key_to_pos;
    const auto&                              resource_keys       = kvcache_resource.cacheKeys();
    const auto&                              source_dependencies = kvcache_resource.blockDependencies();
    RTP_LLM_CHECK_WITH_INFO(resource_keys.size() == source_dependencies.size(),
                            "incrKVCacheRef source timeline mismatch: keys=%zu dependencies=%zu",
                            resource_keys.size(),
                            source_dependencies.size());
    for (size_t i = 0; i < resource_keys.size(); ++i) {
        key_to_pos.emplace(resource_keys[i], i);
    }

    auto selected_resource_ptr = new KVCacheResource(kvcache_resource);
    auto deleter               = [self = shared_from_this(), is_connector](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource, is_connector);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(config_);

    CacheKeysType                                     selected_keys;
    BlockDependenciesType                             selected_dependencies;
    std::unordered_map<std::string, BlockIndicesType> selected_blocks;
    for (const auto& group : config_.groups()) {
        selected_blocks.emplace(group.tag, BlockIndicesType{});
    }

    selected_dependencies.reserve(cache_keys.size());
    selected_keys.reserve(cache_keys.size());
    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t                                  pos             = it->second;
        bool                                          any_valid_block = false;
        std::unordered_map<std::string, BlockIdxType> blocks_for_key;
        for (const auto& group : config_.groups()) {
            const auto& src_binding = kvcache_resource.blockBinding(group.tag);
            const auto  pool_block_id =
                pos < src_binding.size() ? src_binding.lookup(GroupBlockPosition{pos}) : std::nullopt;
            const auto block = pool_block_id.has_value() ? pool_block_id->value : NULL_BLOCK_IDX;
            blocks_for_key.emplace(group.tag, block);
            any_valid_block = any_valid_block || !isNullBlockIdx(block);
        }
        const bool preserve_connector_tail = is_connector && !kvcache_resource.lastBlockAligned()
                                             && pos + 1 == resource_keys.size() && !selected_keys.empty();
        if (!any_valid_block && !preserve_connector_tail) {
            continue;
        }
        selected_keys.push_back(key);
        selected_dependencies.push_back(source_dependencies[pos]);
        for (const auto& group : config_.groups()) {
            selected_blocks.at(group.tag).push_back(blocks_for_key.at(group.tag));
        }
    }

    if (selected_keys.empty()) {
        return nullptr;
    }

    selected_resource->setCacheKeysAndBlockDependencies(std::move(selected_keys), std::move(selected_dependencies));
    selected_resource->setCacheKeysAreCpCanonical(kvcache_resource.cacheKeysAreCpCanonical());
    for (const auto& group : config_.groups()) {
        BlockIndicesType valid;
        for (auto b : selected_blocks.at(group.tag)) {
            if (!isNullBlockIdx(b)) {
                valid.push_back(b);
            }
        }
        if (!valid.empty()) {
            referenceBlocks(group.tag, valid, is_connector);
        }
        selected_resource->mutableBlockBinding(group.tag).assign(
            decodeGroupPolicyBlocks(selected_blocks.at(group.tag)));
    }
    return selected_resource;
}

void HybridPoolKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    for (const auto& [tag, binding] : kvcache_resource.blocksByTag()) {
        BlockIndicesType valid;
        for (const auto& pool_block_id : binding.snapshot()) {
            if (pool_block_id.has_value()) {
                valid.push_back(pool_block_id->value);
            }
        }
        if (!valid.empty()) {
            freeBlocks(tag, valid, is_connector);
        }
    }
}

bool HybridPoolKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr&  batch_kv_cache_resource,
                                               const std::vector<int>&         block_src_batch,
                                               bool                            copy_last_block,
                                               std::vector<TaggedBlockIdPair>& block_update_mapping) {
    block_update_mapping.clear();
    if (block_src_batch.empty()) {
        return true;
    }

    const int old_batch_size = batch_kv_cache_resource->batchSize();
    const int new_batch_size = static_cast<int>(block_src_batch.size());

    std::vector<int> batch_fork_count(old_batch_size, 0);
    for (const int old_batch_idx : block_src_batch) {
        RTP_LLM_CHECK_WITH_INFO(old_batch_idx >= 0 && old_batch_idx < old_batch_size,
                                "try to reuse an old batch %d that out of range %d",
                                old_batch_idx,
                                old_batch_size);
        ++batch_fork_count[old_batch_idx];
    }

    std::map<std::string, int> new_blocks_num;
    for (const auto& group : config_.groups()) {
        new_blocks_num.emplace(group.tag, 0);
    }
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count > 1 && copy_last_block) {
            for (const auto& group : config_.groups()) {
                const auto& tag = group.tag;
                if (batch_kv_cache_resource->blockBinding(old_batch_idx, tag).size() > 0) {
                    new_blocks_num.at(tag) += fork_count - 1;
                }
            }
        }
    }

    // Transfer request ownership from dropped batches before allocating new
    // blocks. This keeps the operation transactional while allowing net-feasible
    // drop-and-fork updates to succeed when the pool is otherwise full.
    std::map<std::string, BlockIndicesType>                      replacement_blocks;
    std::map<std::string, BlockIndicesType>                      allocated_replacements;
    std::map<std::string, std::unordered_map<BlockIdxType, int>> transferred_ref_counts;
    for (const auto& group : config_.groups()) {
        const auto&                           tag = group.tag;
        std::unordered_set<BlockIdxType>      retained_blocks;
        std::unordered_map<BlockIdxType, int> dropped_block_counts;
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
            for (const auto& pool_block_id : batch_kv_cache_resource->blockBinding(old_batch_idx, tag).snapshot()) {
                if (!pool_block_id.has_value()) {
                    continue;
                }
                const auto block = pool_block_id->value;
                if (batch_fork_count[old_batch_idx] == 0) {
                    ++dropped_block_counts[block];
                } else {
                    retained_blocks.insert(block);
                }
            }
        }

        auto&     replacements = replacement_blocks[tag];
        auto&     transferred  = transferred_ref_counts[tag];
        const int need         = new_blocks_num.at(tag);
        for (int old_batch_idx = 0; old_batch_idx < old_batch_size && static_cast<int>(replacements.size()) < need;
             ++old_batch_idx) {
            if (batch_fork_count[old_batch_idx] != 0) {
                continue;
            }
            const auto& dropped = batch_kv_cache_resource->blockBinding(old_batch_idx, tag);
            if (dropped.size() == 0) {
                continue;
            }
            const auto pool_block_id = dropped.lookup(GroupBlockPosition{dropped.size() - 1});
            if (pool_block_id.has_value() && dropped_block_counts[pool_block_id->value] == 1
                && !retained_blocks.count(pool_block_id->value) && !transferred.count(pool_block_id->value)) {
                replacements.push_back(pool_block_id->value);
                transferred[pool_block_id->value] = 1;
            }
        }
    }

    auto rollback_replacements = [&]() {
        for (const auto& group : config_.groups()) {
            auto& blocks = allocated_replacements[group.tag];
            if (!blocks.empty()) {
                groupStrategy(group.tag)->free(blocks);
                blocks.clear();
            }
        }
    };
    for (const auto& group : config_.groups()) {
        const auto& tag         = group.tag;
        const int   need_blocks = new_blocks_num.at(tag);
        auto&       reserved    = replacement_blocks[tag];
        reserved.reserve(static_cast<size_t>(need_blocks));
        for (int i = static_cast<int>(reserved.size()); i < need_blocks; ++i) {
            GroupBlockToPoolBlockBinding one_block_binding;
            const bool ok = groupStrategy(tag)->malloc(one_block_binding, groupStrategy(tag)->seqSizePerBlock());
            const auto block =
                one_block_binding.size() == 1 ? one_block_binding.lookup(GroupBlockPosition{0}) : std::nullopt;
            if (ok && block.has_value()) {
                reserved.push_back(block->value);
                allocated_replacements[tag].push_back(block->value);
                continue;
            }
            if (block.has_value()) {
                allocated_replacements[tag].push_back(block->value);
            }
            RTP_LLM_LOG_WARNING(
                "reserve replacement block failed for hybrid kv cache update, tag=%s need=%d reserved=%zu",
                tag.c_str(),
                need_blocks,
                reserved.size());
            rollback_replacements();
            return false;
        }
    }

    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        if (batch_fork_count[old_batch_idx] != 0) {
            continue;
        }
        for (const auto& group : config_.groups()) {
            const auto&      tag = group.tag;
            BlockIndicesType to_free;
            auto&            transferred = transferred_ref_counts[tag];
            for (const auto& pool_block_id : batch_kv_cache_resource->blockBinding(old_batch_idx, tag).snapshot()) {
                if (!pool_block_id.has_value()) {
                    continue;
                }
                const auto block = pool_block_id->value;
                auto       it    = transferred.find(block);
                if (it != transferred.end() && it->second > 0) {
                    --it->second;
                } else {
                    to_free.push_back(block);
                }
            }
            if (!to_free.empty()) {
                groupStrategy(tag)->free(to_free);
            }
        }
    }

    std::vector<KVCacheResource> old_resources;
    batch_kv_cache_resource->resetAndReturnOldResources(new_batch_size, old_resources);
    batch_kv_cache_resource->initGroups(config_);
    std::map<std::string, size_t> next_replacement;

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        if (fork_count == 1) {
            batch_kv_cache_resource->moveBatchResource(new_batch_idx, std::move(old_resources[old_batch_idx]));
        } else {
            const auto& source_resource = old_resources[old_batch_idx];
            auto&       fork_resource   = batch_kv_cache_resource->cacheResource(new_batch_idx);
            fork_resource.setCacheKeysAndBlockDependencies(source_resource.cacheKeys(),
                                                           source_resource.blockDependencies());
            fork_resource.setCacheKeysAreCpCanonical(source_resource.cacheKeysAreCpCanonical());
            for (const auto& group : config_.groups()) {
                const auto& tag     = group.tag;
                auto&       binding = batch_kv_cache_resource->mutableBlockBinding(new_batch_idx, tag);
                groupStrategy(tag)->reference(
                    binding, encodeBindingForGroupPolicy(old_resources[old_batch_idx].blockBinding(tag)));

                if (copy_last_block && binding.size() > 0) {
                    const auto old_pool_block  = binding.popBack();
                    const bool old_block_valid = old_pool_block.has_value();
                    if (old_block_valid) {
                        groupStrategy(tag)->free({old_pool_block->value});
                    }

                    auto&      reserved     = replacement_blocks[tag];
                    const auto reserved_idx = next_replacement[tag]++;
                    RTP_LLM_CHECK_WITH_INFO(reserved_idx < reserved.size(),
                                            "missing reserved replacement block for hybrid kv cache update, tag=%s",
                                            tag.c_str());
                    const int new_block = reserved[reserved_idx];
                    binding.append(PoolBlockId{new_block});
                    if (old_block_valid && !isNullBlockIdx(new_block)) {
                        block_update_mapping.push_back({tag, old_pool_block->value, new_block});
                    }
                }
            }
        }
        --fork_count;
    }
    for (const auto& group : config_.groups()) {
        const auto& tag = group.tag;
        RTP_LLM_CHECK_WITH_INFO(next_replacement[tag] == replacement_blocks[tag].size(),
                                "unused replacement blocks after hybrid kv cache update, tag=%s used=%zu reserved=%zu",
                                tag.c_str(),
                                next_replacement[tag],
                                replacement_blocks[tag].size());
    }
    return true;
}

int HybridPoolKVCacheAllocator::seqSizePerBlock() const {
    return static_cast<int>(config_.seq_size_per_block);
}

void HybridPoolKVCacheAllocator::rollbackBindingToSize(std::string_view              tag,
                                                       GroupBlockToPoolBlockBinding& binding,
                                                       size_t                        original_size,
                                                       const std::vector<size_t>&    backfilled_positions) {
    const auto       blocks = encodeBindingForGroupPolicy(binding);
    BlockIndicesType blocks_to_free;
    blocks_to_free.reserve(backfilled_positions.size() + blocks.size() - std::min(original_size, blocks.size()));
    for (size_t group_block_position : backfilled_positions) {
        RTP_LLM_CHECK_WITH_INFO(group_block_position < original_size && group_block_position < blocks.size(),
                                "invalid hybrid rollback tag=%s backfill position=%zu original_size=%zu size=%zu",
                                std::string(tag).c_str(),
                                group_block_position,
                                original_size,
                                blocks.size());
        if (!isNullBlockIdx(blocks[group_block_position])) {
            blocks_to_free.push_back(blocks[group_block_position]);
        }
    }
    const auto appended_blocks = validBlocksAfter(blocks, original_size);
    blocks_to_free.insert(blocks_to_free.end(), appended_blocks.begin(), appended_blocks.end());
    if (!blocks_to_free.empty()) {
        freeBlocks(tag, blocks_to_free);
    }
    for (size_t group_block_position : backfilled_positions) {
        binding.unbind(GroupBlockPosition{group_block_position});
    }
    binding.resize(original_size);
}

void HybridPoolKVCacheAllocator::rollbackInitMalloc(
    BatchKVCacheResource&                             kv_resource,
    const std::map<std::string, BlockIndicesType>&    referenced_blocks,
    const std::map<std::string, size_t>&              original_sizes,
    const std::map<std::string, std::vector<size_t>>& backfilled_positions) {
    for (const auto& group : config_.groups()) {
        const auto& tag     = group.tag;
        auto&       binding = kv_resource.mutableBlockBinding(0, tag);
        rollbackBindingToSize(tag, binding, original_sizes.at(tag), backfilled_positions.at(tag));
        const auto referenced_it = referenced_blocks.find(tag);
        if (referenced_it != referenced_blocks.end() && !referenced_it->second.empty()) {
            freeBlocks(tag, referenced_it->second);
        }
        binding.resize(0);
    }
    kv_resource.cacheResource(0).setDeviceReuseBlockNum(0);
}

void HybridPoolKVCacheAllocator::rollbackIncrMalloc(
    BatchKVCacheResource&                                          kv_resource,
    const std::vector<std::map<std::string, size_t>>&              batch_original_sizes,
    const std::vector<std::map<std::string, std::vector<size_t>>>& batch_backfilled_positions,
    size_t                                                         last_touched_batch) {
    const size_t rollback_end = std::min(last_touched_batch + 1, batch_original_sizes.size());
    for (size_t batch_idx = 0; batch_idx < rollback_end; ++batch_idx) {
        for (const auto& group : config_.groups()) {
            const auto& tag     = group.tag;
            auto&       binding = kv_resource.mutableBlockBinding(static_cast<int>(batch_idx), tag);
            rollbackBindingToSize(
                tag, binding, batch_original_sizes[batch_idx].at(tag), batch_backfilled_positions[batch_idx].at(tag));
        }
    }
}

MemoryType HybridPoolKVCacheAllocator::memoryTypeForGroup(std::string_view tag) const {
    (void)config_.group(tag);
    return allocation_type_ == AllocationType::DEVICE ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
}

void HybridPoolKVCacheAllocator::copyBlockMappingForGroup(std::string_view                tag,
                                                          const std::vector<BlockIdPair>& block_update_mapping) const {
    if (block_update_mapping.empty()) {
        return;
    }

    const auto   memory_type         = memoryTypeForGroup(tag);
    const auto   copy_type           = BatchCopyParams::get_copy_type(memory_type, memory_type);
    const auto&  group               = config_.group(tag);
    const auto&  spec                = group.layout.spec;
    const size_t kv_block_size_bytes = spec->block_size_bytes();
    const size_t scale_block_bytes   = spec->scale_block_size_bytes();
    const size_t buffers_per_layer   = scale_block_bytes > 0 ? 2 : 1;

    BatchCopyParams copy_params;
    copy_params.reserve(copy_type, group.layer_ids.size() * block_update_mapping.size() * buffers_per_layer);

    for (const auto& [src_block_index, dest_block_index] : block_update_mapping) {
        for (int layer_id : group.layer_ids) {
            auto src_addr_info = groupStrategy(tag)->convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info = groupStrategy(tag)->convertIndexToAddr(layer_id, dest_block_index);

            RTP_LLM_CHECK_WITH_INFO(src_addr_info.kv_addr && dst_addr_info.kv_addr,
                                    "failed to get block address for tag=%s layer %d src_block %d dst_block %d",
                                    std::string(tag).c_str(),
                                    layer_id,
                                    src_block_index,
                                    dest_block_index);

            copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

            if (scale_block_bytes > 0 && src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                copy_params.add(dst_addr_info.kv_scale_addr, src_addr_info.kv_scale_addr, scale_block_bytes, copy_type);
            }
        }
    }

    execBatchCopy(copy_params);
}

int HybridPoolKVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }
    const auto& cp_mapper          = cp_slot_mapper_;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;
    const int   reuse_blocks_len   = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;

    int common_blocks_total = 0;
    int extra_blocks_total  = 0;
    for (const auto& group : config_.groups()) {
        const auto& tag              = group.tag;
        const int   group_common_seq = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_common_seq_len);
        const int   group_seq_len    = cpEffectiveSeqLenForGroup(cp_mapper, config_, tag, raw_seq_len);
        const auto  need             = groupStrategy(tag)->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
        common_blocks_total += need.common_blocks;
        extra_blocks_total += need.extra_blocks;
    }
    return common_blocks_total + batch_size * extra_blocks_total;
}

int HybridPoolKVCacheAllocator::estimatePeakNeedBlocks(const KVCacheResource& kv_cache_resource,
                                                       int                    seq_len,
                                                       int                    remaining_tokens,
                                                       int                    reserve_step,
                                                       bool                   enable_reuse_cache) const {
    int need_blocks = 0;
    for (const auto& group : config_.groups()) {
        need_blocks += groupStrategy(group.tag)->estimatePeakNeedBlocks(
            seq_len,
            encodeBindingForGroupPolicy(kv_cache_resource.blockBinding(group.tag)),
            remaining_tokens,
            reserve_step,
            enable_reuse_cache);
    }
    return need_blocks;
}

int HybridPoolKVCacheAllocator::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                                   int  common_seq_len,
                                                                   int  remaining_tokens,
                                                                   int  reserve_step,
                                                                   bool enable_reuse_cache,
                                                                   int  target_batch_size) const {
    int peak_blocks = 0;
    for (const auto& group : kv_cache_groups_) {
        peak_blocks += group->estimateInitialBatchPeakNeedBlocks(
            seq_len, common_seq_len, remaining_tokens, reserve_step, enable_reuse_cache, target_batch_size);
    }
    return peak_blocks;
}

void HybridPoolKVCacheAllocator::checkCPShardedMallocResult(const MallocInfo& malloc_info) const {
    if (!cp_slot_mapper_ || !cp_slot_mapper_->isSharded()) {
        return;
    }

    const auto& kv_resource  = malloc_info.batch_kv_cache_resource;
    const int   seq_len      = malloc_info.incrSeqLen();
    const int   reserve_step = malloc_info.complete_token_ids->getReserveStep();

    for (int batch_id = 0; batch_id < kv_resource->batchSize(); ++batch_id) {
        for (const auto& group : config_.groups()) {
            const auto& tag = group.tag;
            if (!cpBlockRoundRobinGroup(cp_slot_mapper_, config_, tag)) {
                continue;
            }
            const int effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, tag, seq_len);
            const int expected_blocks   = groupStrategy(tag)->needBlocksNum(effective_seq_len, 0, reserve_step);
            const int actual_blocks     = kv_resource->blocksNum(batch_id, tag);
            RTP_LLM_CHECK_WITH_INFO(actual_blocks == expected_blocks,
                                    "CP invariant violated: batch=%d tag=%s blocks=%d != expected_local_blocks=%d "
                                    "(seq_len=%d, effective_seq_len=%d, reserve_step=%d, cp_size=%d, "
                                    "block_size=%d, cacheKeys=%zu)",
                                    batch_id,
                                    tag.c_str(),
                                    actual_blocks,
                                    expected_blocks,
                                    seq_len,
                                    effective_seq_len,
                                    reserve_step,
                                    cp_slot_mapper_->cpSize(),
                                    cp_slot_mapper_->blockSize(),
                                    kv_resource->cacheKeys(batch_id).size());
        }
    }
}

int HybridPoolKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                      int                            seq_len,
                                                      int                            reserve_step) const {
    int need_blocks = 0;
    for (const auto& group : config_.groups()) {
        const auto& tag               = group.tag;
        const int   effective_seq_len = cpEffectiveSeqLenForGroup(cp_slot_mapper_, config_, tag, seq_len);
        const int   cur_blocks        = batch_kv_cache_resource->blocksNum(0, tag);
        need_blocks += groupStrategy(tag)->needBlocksNum(effective_seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
}

}  // namespace rtp_llm
