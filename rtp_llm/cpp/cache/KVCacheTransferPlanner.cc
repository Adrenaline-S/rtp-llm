#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <algorithm>
#include <unordered_set>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {

void validateGroupGeometry(const CacheConfig& config, const GroupBase& group, const char* operation) {
    RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "%s requires a non-empty cache group tag", operation);
    const auto& configured_group = config.group(group.tag);
    RTP_LLM_CHECK_WITH_INFO(config.seq_size_per_block > 0,
                            "%s requires a positive global cache-key span for tag=%s",
                            operation,
                            group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        group.seq_size_per_block > 0, "%s requires a positive physical span for tag=%s", operation, group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(configured_group.policy.group_type == group.policy.group_type
                                && configured_group.seq_size_per_block == group.seq_size_per_block
                                && configured_group.kernel_seq_size_per_block == group.kernel_seq_size_per_block
                                && configured_group.policy.cp_mapping == group.policy.cp_mapping
                                && configured_group.policy.active_tail_blocks == group.policy.active_tail_blocks,
                            "%s received geometry that does not match CacheConfig tag=%s",
                            operation,
                            group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(group.seq_size_per_block % config.seq_size_per_block == 0,
                            "%s tag=%s physical span=%zu must be divisible by global cache-key span=%zu",
                            operation,
                            group.tag.c_str(),
                            group.seq_size_per_block,
                            config.seq_size_per_block);
    RTP_LLM_CHECK_WITH_INFO(group.kernel_seq_size_per_block > 0
                                && group.seq_size_per_block % group.kernel_seq_size_per_block == 0,
                            "%s tag=%s has invalid physical/kernel spans B=%zu K=%zu",
                            operation,
                            group.tag.c_str(),
                            group.seq_size_per_block,
                            group.kernel_seq_size_per_block);
}

size_t canonicalKeyOrdinalForPhysicalBlock(const CacheConfig& config,
                                           const GroupBase&   group,
                                           size_t             physical_block_position,
                                           size_t             last_key_ordinal) {
    const size_t keys_per_physical_block = group.seq_size_per_block / config.seq_size_per_block;
    const size_t physical_end_key        = (physical_block_position + 1) * keys_per_physical_block - 1;
    return std::min(physical_end_key, last_key_ordinal);
}

std::vector<size_t> blockPositions(size_t block_num,
                                   size_t reuse_block_size,
                                   bool   use_hybrid,
                                   bool   transfer_tail_blocks,
                                   size_t tail_block_count,
                                   bool   hybrid_full_from_begin) {
    std::vector<size_t> block_pos_list;
    block_pos_list.reserve(block_num);
    if (use_hybrid && block_num > 0 && transfer_tail_blocks) {
        const size_t tail_count = std::max<size_t>(1, tail_block_count);
        const size_t start      = block_num > tail_count ? block_num - tail_count : 0;
        for (size_t block_pos = start; block_pos < block_num; ++block_pos) {
            block_pos_list.push_back(block_pos);
        }
        return block_pos_list;
    }
    const size_t start = use_hybrid && hybrid_full_from_begin ? 0 : reuse_block_size;
    for (size_t block_pos = start; block_pos < block_num; ++block_pos) {
        block_pos_list.push_back(block_pos);
    }
    return block_pos_list;
}

}  // namespace

PhysicalBlockTransferPlan planPhysicalBlocksForCacheTransfer(
    const GroupBase& group, size_t block_num, size_t reuse_block_size, bool use_hybrid, bool hybrid_full_from_begin) {
    RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "cache transfer plan requires a non-empty group tag");
    PhysicalBlockTransferPlan result;
    result.tag = group.tag;
    const size_t tail_block_count =
        group.policy.active_tail_blocks > 0 ? static_cast<size_t>(group.policy.active_tail_blocks) : 0;
    result.physical_block_positions = blockPositions(
        block_num, reuse_block_size, use_hybrid, tail_block_count > 0, tail_block_count, hybrid_full_from_begin);
    return result;
}

void validateTransferSelections(const CacheConfig&              config,
                                const NativeTransferSelections& selections,
                                size_t                          cache_key_count) {
    std::unordered_set<std::string> seen_tags;
    seen_tags.reserve(selections.size());
    for (const auto& selection : selections) {
        RTP_LLM_CHECK_WITH_INFO(!selection.tag.empty(), "transfer selection requires a non-empty cache group tag");
        // Rejects a tag the finalized cache plan does not own.
        config.group(selection.tag);
        RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(selection.tag).second,
                                "transfer selections contain duplicate tag=%s",
                                selection.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(selection.cp_size > 0 && selection.cp_rank >= 0
                                    && selection.cp_rank < selection.cp_size,
                                "transfer selection has invalid CP geometry for tag=%s",
                                selection.tag.c_str());

        std::unordered_set<size_t> seen_positions;
        seen_positions.reserve(selection.global_positions.size());
        for (const size_t global_position : selection.global_positions) {
            RTP_LLM_CHECK_WITH_INFO(global_position < cache_key_count,
                                    "transfer selection tag=%s position=%zu is past the request cache keys=%zu",
                                    selection.tag.c_str(),
                                    global_position,
                                    cache_key_count);
            RTP_LLM_CHECK_WITH_INFO(seen_positions.emplace(global_position).second,
                                    "transfer selection tag=%s repeats position=%zu",
                                    selection.tag.c_str(),
                                    global_position);
        }
    }
}

std::string layerTagCacheTransferKey(size_t request_id, size_t layer_id, const std::string& tag) {
    auto key = std::to_string(request_id) + "-" + std::to_string(layer_id);
    if (!tag.empty() && tag != "default") {
        key += "-tag-" + tag;
    }
    return key;
}

NativeTransferSelection projectTokenRangeForGroup(const CacheConfig& config,
                                                  const GroupBase&   group,
                                                  size_t             start_token,
                                                  size_t             end_token,
                                                  bool               require_aligned_range,
                                                  int                cp_rank,
                                                  int                cp_size) {
    validateGroupGeometry(config, group, "transfer projector");
    RTP_LLM_CHECK_WITH_INFO(
        end_token >= start_token, "transfer projector received an inverted token range for tag=%s", group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(cp_size > 0 && cp_rank >= 0 && cp_rank < cp_size,
                            "transfer projector received invalid CP rank/size for tag=%s",
                            group.tag.c_str());

    if (require_aligned_range) {
        RTP_LLM_CHECK_WITH_INFO(
            start_token % config.seq_size_per_block == 0 && end_token % config.seq_size_per_block == 0
                && start_token % group.seq_size_per_block == 0 && end_token % group.seq_size_per_block == 0,
            "transfer range [%zu,%zu) is not aligned to tag=%s global/physical spans=%zu/%zu",
            start_token,
            end_token,
            group.tag.c_str(),
            config.seq_size_per_block,
            group.seq_size_per_block);
    }

    const size_t first_key_ordinal = start_token / config.seq_size_per_block;
    const size_t key_ordinal_end =
        end_token / config.seq_size_per_block + static_cast<size_t>(end_token % config.seq_size_per_block != 0);

    NativeTransferSelection result;
    result.tag        = group.tag;
    result.cp_mapping = cp_size > 1 ? group.policy.cp_mapping : CpBlockMappingMode::NONE;
    result.cp_rank    = cp_rank;
    result.cp_size    = cp_size;
    if (key_ordinal_end <= first_key_ordinal) {
        return result;
    }

    const size_t physical_begin = start_token / group.seq_size_per_block;
    const size_t physical_end =
        end_token / group.seq_size_per_block + static_cast<size_t>(end_token % group.seq_size_per_block != 0);
    const size_t last_key_ordinal = key_ordinal_end - 1;

    const auto world_size = static_cast<size_t>(cp_size);
    const auto rank       = static_cast<size_t>(cp_rank);
    if (result.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK) {
        const size_t compact_begin          = physical_begin / world_size;
        const size_t compact_end            = (physical_end + world_size - 1) / world_size;
        size_t       selected_compact_begin = compact_begin;
        if (group.policy.group_type != CacheGroupType::FULL) {
            const size_t tail =
                group.policy.active_tail_blocks > 0 ? static_cast<size_t>(group.policy.active_tail_blocks) : 1;
            selected_compact_begin = compact_end > tail ? std::max(compact_begin, compact_end - tail) : compact_begin;
        }
        for (size_t compact_position = selected_compact_begin; compact_position < compact_end; ++compact_position) {
            const size_t physical_position = std::min((compact_position + 1) * world_size - 1, physical_end - 1);
            result.global_positions.push_back(
                canonicalKeyOrdinalForPhysicalBlock(config, group, physical_position, last_key_ordinal));
        }
        return result;
    }

    size_t selected_begin = physical_begin;
    if (group.policy.group_type != CacheGroupType::FULL) {
        const size_t tail =
            group.policy.active_tail_blocks > 0 ? static_cast<size_t>(group.policy.active_tail_blocks) : 1;
        selected_begin = physical_end > tail ? std::max(physical_begin, physical_end - tail) : physical_begin;
    }
    for (size_t physical_position = selected_begin; physical_position < physical_end; ++physical_position) {
        if (result.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN && physical_position % world_size != rank) {
            continue;
        }
        result.global_positions.push_back(
            canonicalKeyOrdinalForPhysicalBlock(config, group, physical_position, last_key_ordinal));
    }
    return result;
}

}  // namespace rtp_llm
