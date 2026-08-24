#include "rtp_llm/cpp/cache/SWACacheManager.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool isActiveTailBlock(int block_idx, int seq_slots, int active_tail_blocks) {
    if (seq_slots <= 0 || block_idx >= seq_slots) {
        return false;
    }
    return block_idx >= std::max(seq_slots - active_tail_blocks, 0);
}

bool shouldAllocateBlock(
    int block_idx, int seq_slots, int reserve_step, int step, bool enable_reuse_cache, int active_tail_blocks) {
    const bool is_reserve = reserve_step > 0 && block_idx >= seq_slots;
    const bool step_hit   = ((block_idx + 1) % step) == 0;
    return is_reserve || isActiveTailBlock(block_idx, seq_slots, active_tail_blocks)
           || (enable_reuse_cache && step_hit);
}

}  // namespace

bool SWACacheManager::shouldCheckSWATailBlockIds() const {
    return policy().validate_tail_blocks;
}

bool SWACacheManager::effectiveReuseCacheForAllocation(bool enable_reuse_cache) const {
    return enable_reuse_cache && policy().enable_prefix_reuse;
}

int SWACacheManager::activeTailBlockCount() const {
    return static_cast<int>(std::max(1u, policy().active_tail_blocks));
}

void SWACacheManager::checkSWATailBindings(const GroupBlockToPoolBlockBinding& binding, const char* caller) const {
    if (!shouldCheckSWATailBlockIds()) {
        return;
    }

    const auto blocks = binding.snapshot();
    if (blocks.empty()) {
        return;
    }

    const size_t block_num = blocks.size();
    RTP_LLM_CHECK_WITH_INFO(blocks[block_num - 1].has_value(),
                            "%s invalid SWA block ids: tail block is NULL, block_num=%zu",
                            caller,
                            block_num);
    if (activeTailBlockCount() >= 2 && block_num >= 2) {
        RTP_LLM_CHECK_WITH_INFO(blocks[block_num - 2].has_value(),
                                "%s invalid SWA block ids: tail-1 block is NULL, block_num=%zu",
                                caller,
                                block_num);
    }
}

void SWACacheManager::filterValidBlocks(const BlockIndicesType& in, BlockIndicesType& out) const {
    out.clear();
    out.reserve(in.size());
    for (auto b : in) {
        if (!isNullBlockIdx(b)) {
            out.push_back(b);
        }
    }
}

int SWACacheManager::needBlocksNum(int seq_len, int current_blocks, int reserve_step) const {
    const int block_size = seqSizePerBlock();
    return std::max((seq_len + reserve_step + block_size - 1) / block_size - current_blocks, 0);
}

// Conservative upper bound: sliding-window peak usage never exceeds full-attention usage.
int SWACacheManager::estimatePeakNeedBlocks(int                     seq_len,
                                            const BlockIndicesType& current_block_indices,
                                            int                     remaining_tokens,
                                            int                     reserve_step,
                                            bool                    enable_reuse_cache) const {
    (void)enable_reuse_cache;
    int allocated_blocks = 0;
    for (const auto block_index : current_block_indices) {
        allocated_blocks += !isNullBlockIdx(block_index);
    }
    return std::max(needBlocksNum(seq_len + remaining_tokens, 0, reserve_step) - allocated_blocks, 0);
}

int SWACacheManager::estimateInitialBatchPeakNeedBlocks(int  seq_len,
                                                        int  common_seq_len,
                                                        int  remaining_tokens,
                                                        int  reserve_step,
                                                        bool enable_reuse_cache,
                                                        int  target_batch_size) const {
    (void)enable_reuse_cache;
    const int batch_size    = std::max(target_batch_size, 1);
    const int common_blocks = needBlocksNum(common_seq_len, 0);
    const int peak_blocks   = needBlocksNum(seq_len + remaining_tokens, 0, reserve_step);
    return common_blocks + batch_size * std::max(peak_blocks - common_blocks, 0);
}

NeedBlocksInfo SWACacheManager::getNeedBlocks(
    int common_seq_len, int seq_len, int reserve_step, int reuse_blocks_len, bool reuse_enabled) const {
    (void)common_seq_len;
    const int  step                    = std::max(1, linear_step_);
    const bool effective_reuse_enabled = effectiveReuseCacheForAllocation(reuse_enabled);
    const int  active_tail_blocks      = activeTailBlockCount();

    NeedBlocksInfo info;

    const int seq_slots   = needBlocksNum(seq_len, 0);
    const int total_slots = needBlocksNum(seq_len, 0, reserve_step);

    info.common_blocks = 0;
    for (int group_block_position = reuse_blocks_len; group_block_position < seq_slots; ++group_block_position) {
        if (shouldAllocateBlock(group_block_position,
                                seq_slots,
                                /*reserve_step=*/0,
                                step,
                                effective_reuse_enabled,
                                active_tail_blocks)) {
            ++info.extra_blocks;
        }
    }
    info.extra_blocks += std::max(total_slots - std::max(seq_slots, reuse_blocks_len), 0);

    info.extra_blocks = std::max(info.extra_blocks, 0);
    return info;
}

MatchResult SWACacheManager::matchSingleKey(CacheKeyType cache_key) const {
    MatchResult result;
    if (!shared_cache_) {
        return result;
    }
    auto block_idx = shared_cache_->matchGroup(cache_key, tag());
    if (!isNullBlockIdx(block_idx)) {
        result.block_indices = {block_idx};
    }
    return result;
}

bool SWACacheManager::malloc(GroupBlockToPoolBlockBinding& binding,
                             int                           seq_len,
                             bool                          enable_reuse_cache,
                             int                           reserve_step,
                             std::vector<size_t>*          backfilled_positions) {
    if (backfilled_positions != nullptr) {
        backfilled_positions->clear();
    }
    const int  step                    = std::max(1, linear_step_);
    const bool effective_reuse_enabled = effectiveReuseCacheForAllocation(enable_reuse_cache);
    const int  active_tail_blocks      = activeTailBlockCount();
    const int  current_blocks_len      = static_cast<int>(binding.size());
    const int  seq_slots               = needBlocksNum(seq_len, 0, 0);
    const int  new_blocks_len          = needBlocksNum(seq_len, current_blocks_len, reserve_step);

    if (new_blocks_len == 0) {
        checkSWATailBindings(binding, "SWACacheManager::malloc");
        return true;
    }

    int need_alloc_blocks = 0;
    for (int group_block_position = current_blocks_len; group_block_position < current_blocks_len + new_blocks_len;
         group_block_position++) {
        if (shouldAllocateBlock(
                group_block_position, seq_slots, reserve_step, step, effective_reuse_enabled, active_tail_blocks)) {
            need_alloc_blocks++;
        }
    }

    if (need_alloc_blocks > 0) {
        const auto free_blocks_num = freeBlocksNum();
        if (free_blocks_num < static_cast<size_t>(need_alloc_blocks)) {
            if (!ensureFreeBlocks(need_alloc_blocks)) {
                RTP_LLM_LOG_WARNING("Insufficient free blocks for SWACacheManager: need %d, have %zu",
                                    need_alloc_blocks,
                                    free_blocks_num);
                return false;
            }
        }
    }

    BlockIndicesType allocated_blocks;
    if (need_alloc_blocks > 0) {
        allocated_blocks = block_pool_->malloc(need_alloc_blocks);
        if (allocated_blocks.size() != static_cast<size_t>(need_alloc_blocks)) {
            if (!allocated_blocks.empty()) {
                block_pool_->requestFree(allocated_blocks);
            }
            return false;
        }
    }

    GroupBlockToPoolBlockBinding::Snapshot new_ids;
    new_ids.reserve(static_cast<size_t>(new_blocks_len));
    size_t compact_position = 0;
    for (int group_block_position = current_blocks_len; group_block_position < current_blocks_len + new_blocks_len;
         group_block_position++) {
        const bool should_alloc = shouldAllocateBlock(
            group_block_position, seq_slots, reserve_step, step, effective_reuse_enabled, active_tail_blocks);
        if (should_alloc) {
            new_ids.push_back(PoolBlockId{allocated_blocks[compact_position++]});
        } else {
            new_ids.push_back(std::nullopt);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(compact_position == allocated_blocks.size(),
                            "swa kv allocation accounting mismatch, used=%zu allocated=%zu",
                            compact_position,
                            allocated_blocks.size());
    binding.append(new_ids);
    checkSWATailBindings(binding, "SWACacheManager::malloc");
    return true;
}

void SWACacheManager::removeSkippedBlocks(GroupBlockToPoolBlockBinding& binding,
                                          bool                          enable_reuse_cache,
                                          int                           reserve_step) {
    const auto block_indices = binding.snapshot();
    if (block_indices.empty()) {
        checkSWATailBindings(binding, "SWACacheManager::removeSkippedBlocks");
        return;
    }
    const int  step                    = std::max(1, linear_step_);
    const bool effective_reuse_enabled = effectiveReuseCacheForAllocation(enable_reuse_cache);
    const int  active_tail_blocks      = activeTailBlockCount();
    const int  block_size              = static_cast<int>(block_indices.size());

    BlockIndicesType                blocks_to_free;
    std::vector<GroupBlockPosition> positions_to_unbind;
    for (int group_block_position = block_size - active_tail_blocks - 1 - reserve_step; group_block_position >= 0;
         group_block_position--) {
        if (!block_indices[group_block_position].has_value()) {
            continue;
        }
        if (effective_reuse_enabled && ((group_block_position + 1) % step) == 0) {
            continue;
        }
        blocks_to_free.push_back(block_indices[group_block_position]->value);
        positions_to_unbind.push_back(GroupBlockPosition{static_cast<size_t>(group_block_position)});
    }
    if (!blocks_to_free.empty()) {
        block_pool_->requestFree(blocks_to_free);
        binding.remove(positions_to_unbind);
    }
    checkSWATailBindings(binding, "SWACacheManager::removeSkippedBlocks");
}

void SWACacheManager::free(const BlockIndicesType& block_indices) {
    if (block_indices.empty()) {
        return;
    }
    BlockIndicesType valid;
    filterValidBlocks(block_indices, valid);
    if (!valid.empty()) {
        block_pool_->requestFree(valid);
    }
}

void SWACacheManager::reference(GroupBlockToPoolBlockBinding& binding, const BlockIndicesType& new_block_indices) {
    GroupBlockToPoolBlockBinding::Snapshot appended;
    appended.reserve(new_block_indices.size());
    for (const auto block_idx : new_block_indices) {
        appended.push_back(isNullBlockIdx(block_idx) ? std::nullopt :
                                                       std::optional<PoolBlockId>{PoolBlockId{block_idx}});
    }
    binding.append(appended);
    BlockIndicesType valid;
    filterValidBlocks(new_block_indices, valid);
    if (!valid.empty()) {
        block_pool_->requestReference(valid);
    }
}

}  // namespace rtp_llm
