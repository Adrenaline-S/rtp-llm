#pragma once

#include <cstdint>

namespace rtp_llm {

// Cache group type for hybrid KV-cache:
// - LINEAR: linear attention group (PD cache-store transfer keeps the last block)
// - FULL: full attention group (all blocks are needed for cache-store transfer)
// - SWA: sliding-window attention group (PD cache-store transfer keeps the last two blocks)
enum class CacheGroupType : int8_t {
    LINEAR = 0,
    FULL   = 1,
    SWA    = 2,
};

enum class CacheReusePolicy : int8_t {
    REUSABLE     = 0,
    NON_REUSABLE = 1,
};

enum class CacheEvictPolicy : int8_t {
    CHAIN       = 0,
    INDEPENDENT = 1,
    NONE        = 2,
};

enum class CacheMemoryPlacement : int8_t {
    DEVICE      = 0,
    HOST        = 1,
    HOST_PINNED = 2,
};

enum class CpBlockMappingMode : int8_t {
    NONE              = 0,
    BLOCK_ROUND_ROBIN = 1,
    COMPACT_LAST_RANK = 2,
};

enum class CpBlockSliceMode : int8_t {
    NONE          = 0,
    EQUAL_BYTES   = 1,
    PAYLOAD_BYTES = 2,
};

struct CacheGroupPolicy {
    CacheGroupType       group_type             = CacheGroupType::FULL;
    bool                 enable_prefix_reuse    = true;
    CacheEvictPolicy     evict_policy           = CacheEvictPolicy::CHAIN;
    bool                 reservable             = true;
    uint32_t             explicit_block_num     = 0;
    bool                 charge_to_paged_budget = false;
    CacheMemoryPlacement memory_placement       = CacheMemoryPlacement::DEVICE;
    uint32_t             active_tail_blocks     = 0;
    bool                 validate_tail_blocks   = true;
    CpBlockMappingMode   cp_mapping             = CpBlockMappingMode::NONE;
    CpBlockSliceMode     cp_slice               = CpBlockSliceMode::NONE;
};

inline const char* cacheGroupTypeName(CacheGroupType group_type) {
    switch (group_type) {
        case CacheGroupType::LINEAR:
            return "LINEAR";
        case CacheGroupType::FULL:
            return "FULL";
        case CacheGroupType::SWA:
            return "SWA";
    }
    return "UNKNOWN";
}

inline const char* cacheEvictPolicyName(CacheEvictPolicy evict_policy) {
    switch (evict_policy) {
        case CacheEvictPolicy::CHAIN:
            return "chain";
        case CacheEvictPolicy::INDEPENDENT:
            return "independent";
        case CacheEvictPolicy::NONE:
            return "none";
    }
    return "unknown";
}

inline CacheGroupPolicy defaultCacheGroupPolicy(CacheGroupType group_type) {
    CacheGroupPolicy policy;
    policy.group_type          = group_type;
    policy.enable_prefix_reuse = group_type == CacheGroupType::FULL || group_type == CacheGroupType::LINEAR;
    policy.active_tail_blocks  = group_type == CacheGroupType::LINEAR ? 1 : (group_type == CacheGroupType::SWA ? 2 : 0);
    if (group_type == CacheGroupType::FULL) {
        policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    } else if (group_type == CacheGroupType::SWA) {
        policy.cp_mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
    }
    return policy;
}

}  // namespace rtp_llm
