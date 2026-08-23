#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"

namespace rtp_llm {

std::string layerTagCacheTransferKey(size_t request_id, size_t layer_id, const std::string& tag);

struct NativeTransferSelection {
    std::string        tag;
    CpBlockMappingMode cp_mapping = CpBlockMappingMode::NONE;
    int                cp_rank    = 0;
    int                cp_size    = 1;
    // Global cache-key ordinals selected for this semantic group. A consumer
    // may derive a packed local position only while building its final
    // tensor/wire payload; that local position is never returned or stored.
    std::vector<size_t> global_positions;
};

using NativeTransferSelections = std::vector<NativeTransferSelection>;

// Validate a complete tagged transfer selection set before any buffer, tensor or
// wire payload exists: every tag is non-empty, owned by the finalized cache plan
// and unique across the set, CP rank/size are well formed, and every selected
// global cache-key position is inside the request timeline and unique inside its
// selection. An invalid set fails as a whole, so a transfer is never partial.
void validateTransferSelections(const CacheConfig&              config,
                                const NativeTransferSelections& selections,
                                size_t                          cache_key_count);

// Decode/cache-store consumes physical block-table positions rather than
// canonical cache-key ordinals. Keep that unit in a distinct tag-bearing
// internal plan so it cannot be passed to a NativeTransferSelection adapter.
struct PhysicalBlockTransferPlan {
    std::string         tag;
    std::vector<size_t> physical_block_positions;
};

PhysicalBlockTransferPlan planPhysicalBlocksForCacheTransfer(
    const GroupBase& group, size_t block_num, size_t reuse_block_size, bool use_hybrid, bool hybrid_full_from_begin);

NativeTransferSelection projectTokenRangeForGroup(const CacheConfig& config,
                                                  const GroupBase&   group,
                                                  size_t             start_token,
                                                  size_t             end_token,
                                                  bool               require_aligned_range,
                                                  int                cp_rank = 0,
                                                  int                cp_size = 1);

// The cache_store registration plan (CacheStoreBlockPair + buildCacheStorePlan)
// lives in CacheGroupType.h, keyed on CacheGroupPolicy rather than on a bare
// CacheGroupType, and is reached through CPSlotMapper::buildStorePlan(). Do not
// redeclare either here: this header includes CacheGroupType.h, so a second
// definition of CacheStoreBlockPair in namespace rtp_llm is a redefinition
// error in every translation unit that includes this file.

}  // namespace rtp_llm
