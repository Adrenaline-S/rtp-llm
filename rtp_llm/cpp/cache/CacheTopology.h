#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"

namespace rtp_llm {

// Immutable cache-group configuration published by CacheConfig. The tag is the
// only business identity; the storage position of a group carries no meaning.
struct GroupBase {
    std::string                        tag;
    std::shared_ptr<const KVCacheSpec> spec;
    CacheGroupPolicy                   policy;
    std::vector<int>                   layer_ids;

    uint32_t block_num                 = 0;
    uint32_t local_kv_head_num         = 1;
    size_t   seq_size_per_block        = 0;
    size_t   kernel_seq_size_per_block = 0;
    size_t   kv_block_stride_bytes     = 0;
    size_t   kv_scale_stride_bytes     = 0;
};

// Order is deterministic but carries no business meaning.
struct LayerBase {
    int                      layer_id = -1;
    std::vector<std::string> group_tags;
};

struct CacheTopology {
public:
    using GroupRefs = std::vector<std::reference_wrapper<const GroupBase>>;

    static std::shared_ptr<const CacheTopology> create(std::vector<GroupBase> groups, std::vector<LayerBase> layers);

    const std::vector<GroupBase>& groups() const {
        return groups_;
    }

    const std::vector<LayerBase>& layers() const {
        return layers_;
    }

    const GroupBase& group(std::string_view tag) const;
    const LayerBase& layer(int layer_id) const;
    GroupRefs        groupsForLayer(int layer_id) const;
    const GroupBase& groupForLayer(int layer_id, std::string_view tag) const;
    const GroupBase& soleGroupForLayer(int layer_id) const;

    bool hasSingleGlobalGroup() const;
    bool hasOneGroupPerLayer() const;

private:
    CacheTopology(std::vector<GroupBase> groups, std::vector<LayerBase> layers);
    void validateAndBuildIndex();

    std::vector<GroupBase> groups_;
    std::vector<LayerBase> layers_;
    // Private storage slot of each tag inside groups_. Never an interface value:
    // it is not returned, bound, serialized, or used as a cross-module key.
    std::unordered_map<std::string, size_t> tag_to_slot_;
};

// Canonical physical-layout identity of a cache topology.
//
// The signature is raw canonical bytes -- never a hash, checksum, std::hash,
// schema field, version field, or pointer value -- built from fixed-width
// little-endian integers and length-prefixed strings so the encoding is
// unambiguous and directly comparable byte-for-byte across ranks and processes.
//
// It INCLUDES exactly: sorted unique group tags, per-layer group membership,
// and each group's physical layout (spec layout type, local KV heads, physical
// block size B from the spec, kernel blocks per KV block K, KV block stride, and
// KV scale stride). It EXCLUDES policy/behavior, block counts and capacity,
// addresses, rank-local budgets, ranks, versions, and any positional ordinal.
// Local group declaration order therefore never changes the signature.
using CacheTopologySignature = std::string;

CacheTopologySignature physicalTopologySignature(const CacheTopology& topology);

// Fails fast when two physical topology signatures disagree. Used before
// allocation and transfer at cross-process / cross-rank cache boundaries.
void checkPhysicalTopologyMatches(const CacheTopologySignature& expected,
                                  const CacheTopologySignature& actual,
                                  const char*                   boundary);

}  // namespace rtp_llm
