#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"

#include <algorithm>
#include <unordered_map>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

BlockIdxType blockIdForWirePacking(const CacheConfig&             config,
                                   const GroupBase&               group,
                                   const BlockIds&                block_ids,
                                   const NativeTransferSelection& selection,
                                   size_t                         global_key_ordinal) {
    RTP_LLM_CHECK_WITH_INFO(selection.cp_size > 0 && selection.cp_rank >= 0 && selection.cp_rank < selection.cp_size,
                            "P2P transfer selection has invalid CP geometry for tag=%s",
                            selection.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(config.seq_size_per_block > 0 && group.seq_size_per_block > 0
                                && group.seq_size_per_block % config.seq_size_per_block == 0,
                            "P2P transfer tag=%s has invalid global/physical spans=%zu/%zu",
                            selection.tag.c_str(),
                            config.seq_size_per_block,
                            group.seq_size_per_block);
    RTP_LLM_CHECK_WITH_INFO(group.kernel_seq_size_per_block > 0
                                && group.seq_size_per_block % group.kernel_seq_size_per_block == 0,
                            "P2P transfer tag=%s has invalid physical/kernel spans=%zu/%zu",
                            selection.tag.c_str(),
                            group.seq_size_per_block,
                            group.kernel_seq_size_per_block);
    const size_t physical_kernel_blocks = group.seq_size_per_block / group.kernel_seq_size_per_block;
    const size_t stored_kernel_blocks   = group.policy.group_type == CacheGroupType::FULL ? physical_kernel_blocks : 1;
    RTP_LLM_CHECK_WITH_INFO(block_ids.kernelBlocksPerKvBlock() == stored_kernel_blocks,
                            "P2P transfer tag=%s block table K=%zu does not match stored/group K=%zu/%zu",
                            selection.tag.c_str(),
                            block_ids.kernelBlocksPerKvBlock(),
                            stored_kernel_blocks,
                            physical_kernel_blocks);
    const auto world_size       = static_cast<size_t>(selection.cp_size);
    const auto rank             = static_cast<size_t>(selection.cp_rank);
    const auto expected_mapping = selection.cp_size > 1 ? group.policy.cp_mapping : CpBlockMappingMode::NONE;
    RTP_LLM_CHECK_WITH_INFO(selection.cp_mapping == expected_mapping,
                            "P2P transfer selection mapping does not match tag=%s policy",
                            selection.tag.c_str());
    const size_t physical_block_position =
        ((global_key_ordinal + 1) * config.seq_size_per_block - 1) / group.seq_size_per_block;
    if (selection.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN) {
        RTP_LLM_CHECK_WITH_INFO(physical_block_position % world_size == rank,
                                "P2P transfer key ordinal=%zu maps to unowned physical block=%zu rank=%zu tag=%s",
                                global_key_ordinal,
                                physical_block_position,
                                rank,
                                selection.tag.c_str());
    }
    // Final wire adapter: this process-local ordinal is consumed immediately
    // and neither returned nor stored in a transfer record.
    const size_t local_ordinal      = selection.cp_mapping == CpBlockMappingMode::NONE ?
                                          physical_block_position :
                                          physical_block_position / world_size;
    const auto&  physical_block_ids = block_ids.blocks();
    RTP_LLM_CHECK_WITH_INFO(local_ordinal < physical_block_ids.size(),
                            "P2P transfer key ordinal=%zu maps past tag=%s physical blocks=%zu",
                            global_key_ordinal,
                            selection.tag.c_str(),
                            physical_block_ids.size());
    return physical_block_ids[local_ordinal];
}

}  // namespace

NativeTransferSelection LayerCacheBufferUtil::selectBlocksForTag(const CacheConfig&     config,
                                                                 const KVCacheResource& resource,
                                                                 std::string_view       tag,
                                                                 int                    start_key_ordinal,
                                                                 int                    key_count,
                                                                 int                    cp_rank,
                                                                 int                    cp_size) {
    NativeTransferSelection selection;
    selection.tag     = std::string(tag);
    const auto& group = config.group(tag);

    if (start_key_ordinal < 0 || key_count == 0 || key_count < -1 || cp_size < 1 || cp_rank < 0 || cp_rank >= cp_size) {
        RTP_LLM_LOG_WARNING("invalid tagged cache selection arguments for tag=%s", selection.tag.c_str());
        return selection;
    }

    RTP_LLM_CHECK_WITH_INFO(config.seq_size_per_block > 0 && group.seq_size_per_block > 0
                                && group.seq_size_per_block % config.seq_size_per_block == 0,
                            "P2P selector tag=%s has invalid global/physical spans=%zu/%zu",
                            selection.tag.c_str(),
                            config.seq_size_per_block,
                            group.seq_size_per_block);
    const auto& block_ids         = resource.blockIds(tag);
    const auto  mapping           = cp_size > 1 ? group.policy.cp_mapping : CpBlockMappingMode::NONE;
    const auto  world_size        = static_cast<size_t>(cp_size);
    const auto  rank              = static_cast<size_t>(cp_rank);
    const auto  local_block_count = block_ids.blocks().size();
    size_t      physical_capacity = local_block_count;
    if (mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN && local_block_count > 0) {
        physical_capacity = rank + (local_block_count - 1) * world_size + 1;
    } else if (mapping == CpBlockMappingMode::COMPACT_LAST_RANK) {
        physical_capacity = local_block_count * world_size;
    }
    const size_t keys_per_physical_block = group.seq_size_per_block / config.seq_size_per_block;
    const size_t available_key_count =
        std::min(resource.cacheKeys().size(), physical_capacity * keys_per_physical_block);
    const size_t begin = static_cast<size_t>(start_key_ordinal);
    if (begin >= available_key_count) {
        return selection;
    }
    const size_t remaining = available_key_count - begin;
    const size_t count     = key_count > 0 ? std::min(static_cast<size_t>(key_count), remaining) : remaining;
    return projectTokenRangeForGroup(config,
                                     group,
                                     begin * config.seq_size_per_block,
                                     (begin + count) * config.seq_size_per_block,
                                     /*require_aligned_range=*/false,
                                     cp_rank,
                                     cp_size);
}

NativeTransferSelections LayerCacheBufferUtil::selectBlocks(const CacheConfig&     config,
                                                            const KVCacheResource& resource,
                                                            int                    start_key_ordinal,
                                                            int                    key_count,
                                                            int                    cp_rank,
                                                            int                    cp_size) {
    NativeTransferSelections selections;
    selections.reserve(resource.groupBlocks().size());
    for (const auto& group_blocks : resource.groupBlocks()) {
        RTP_LLM_CHECK_WITH_INFO(group_blocks != nullptr, "P2P selection got null cache group record");
        selections.push_back(
            selectBlocksForTag(config, resource, group_blocks->tag, start_key_ordinal, key_count, cp_rank, cp_size));
    }
    return selections;
}

std::vector<std::shared_ptr<LayerCacheBuffer>> LayerCacheBufferUtil::convertBySelections(
    const CacheConfig& config, KVCacheResource& resource, int batch_id, const NativeTransferSelections& selections) {
    // Whole-set prevalidation: identity, CP geometry and selected positions are
    // checked before any buffer or wire output is created.
    validateTransferSelections(config, selections, resource.cacheKeys().size());

    std::vector<std::shared_ptr<LayerCacheBuffer>>                  layer_cache_buffers;
    std::unordered_map<std::string, const NativeTransferSelection*> selections_by_tag;
    selections_by_tag.reserve(selections.size());
    for (const auto& selection : selections) {
        // The request resource must own the selected group as well.
        resource.blockIds(selection.tag);
        RTP_LLM_CHECK_WITH_INFO(selections_by_tag.emplace(selection.tag, &selection).second,
                                "P2P transfer selections contain duplicate tag=%s",
                                selection.tag.c_str());
    }

    for (int layer_id = 0; layer_id < resource.layerNum(); ++layer_id) {
        for (const auto& tag : resource.groupTagsForLayer(layer_id)) {
            const auto selection_it = selections_by_tag.find(tag);
            if (selection_it == selections_by_tag.end()) {
                continue;
            }
            auto buffer = convertLayer(config, resource, batch_id, layer_id, *selection_it->second);
            if (buffer) {
                layer_cache_buffers.push_back(std::move(buffer));
            }
        }
    }
    return layer_cache_buffers;
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferUtil::convertLayer(const CacheConfig&             config,
                                                                     KVCacheResource&               resource,
                                                                     int                            batch_id,
                                                                     int                            layer_id,
                                                                     const NativeTransferSelection& selection) {
    (void)batch_id;
    const auto& group              = config.groupForLayer(layer_id, selection.tag);
    const auto& block_ids          = resource.blockIdsForLayer(layer_id, selection.tag);
    const auto& cache_keys         = resource.cacheKeys();
    auto        layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id, selection.tag);
    for (const size_t global_position : selection.global_positions) {
        if (global_position >= cache_keys.size()) {
            break;
        }
        const auto block_id = blockIdForWirePacking(config, group, block_ids, selection, global_position);
        if (!isNullBlockIdx(block_id)) {
            layer_cache_buffer->addBlockId(cache_keys[global_position], block_id);
        }
    }
    return layer_cache_buffer->blockIdMap().empty() ? nullptr : layer_cache_buffer;
}

bool LayerCacheBufferUtil::hasTransferableBlocks(const CacheConfig&             config,
                                                 const KVCacheResource&         resource,
                                                 int                            layer_id,
                                                 const NativeTransferSelection& selection) {
    const auto& group     = config.groupForLayer(layer_id, selection.tag);
    const auto& block_ids = resource.blockIdsForLayer(layer_id, selection.tag);
    for (const size_t global_position : selection.global_positions) {
        if (global_position >= resource.cacheKeys().size()) {
            break;
        }
        if (!isNullBlockIdx(blockIdForWirePacking(config, group, block_ids, selection, global_position))) {
            return true;
        }
    }
    return false;
}

transfer::KeyBlockInfoMap
LayerCacheBufferUtil::buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                         const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                         int                                         partition_count,
                                         int                                         partition_id) {
    transfer::KeyBlockInfoMap key_block_infos;
    int                       layer_id = layer_cache_buffer->getLayerId();

    for (const auto& [cache_key, block_id] : layer_cache_buffer->blockIdMap()) {
        auto block_infos = converter->convertIndexToBufferByTag(
            layer_id, layer_cache_buffer->cacheTag(), block_id, partition_count, partition_id);

        transfer::KeyBlockInfo kbi;
        kbi.cache_key              = cache_key;
        kbi.blocks                 = std::move(block_infos);
        key_block_infos[cache_key] = std::make_shared<const transfer::KeyBlockInfo>(std::move(kbi));
    }
    return key_block_infos;
}

}  // namespace rtp_llm
