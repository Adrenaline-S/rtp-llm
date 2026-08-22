#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/BufferTypes.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

inline GroupBlockToPoolBlockBinding::Snapshot poolBlockSnapshotForTest(const BlockIndicesType& encoded) {
    GroupBlockToPoolBlockBinding::Snapshot snapshot;
    snapshot.reserve(encoded.size());
    for (const auto block_idx : encoded) {
        RTP_LLM_CHECK_WITH_INFO(block_idx >= 0 || isNullBlockIdx(block_idx),
                                "test pool block id must be nonnegative or missing, got %d",
                                block_idx);
        snapshot.push_back(isNullBlockIdx(block_idx) ? std::nullopt :
                                                       std::optional<PoolBlockId>{PoolBlockId{block_idx}});
    }
    return snapshot;
}

inline BlockIndicesType encodedPoolBlocksForTest(const GroupBlockToPoolBlockBinding& binding) {
    BlockIndicesType encoded;
    encoded.reserve(binding.size());
    for (const auto& pool_block_id : binding.snapshot()) {
        encoded.push_back(pool_block_id.has_value() ? pool_block_id->value : NULL_BLOCK_IDX);
    }
    return encoded;
}

namespace test {

using ::rtp_llm::encodedPoolBlocksForTest;
using ::rtp_llm::poolBlockSnapshotForTest;

struct CacheSpecSemanticSnapshot {
    DataType    dtype                 = DataType::TYPE_INVALID;
    size_t      block_elems           = 0;
    size_t      k_block_elems         = 0;
    size_t      v_block_elems         = 0;
    size_t      block_bytes           = 0;
    size_t      k_block_bytes         = 0;
    size_t      v_block_bytes         = 0;
    size_t      block_payload_bytes   = 0;
    size_t      k_block_payload_bytes = 0;
    size_t      v_block_payload_bytes = 0;
    size_t      scale_block_bytes     = 0;
    size_t      k_scale_block_bytes   = 0;
    size_t      v_scale_block_bytes   = 0;
    std::string fingerprint;

    bool operator==(const CacheSpecSemanticSnapshot& other) const {
        return dtype == other.dtype && block_elems == other.block_elems && k_block_elems == other.k_block_elems
               && v_block_elems == other.v_block_elems && block_bytes == other.block_bytes
               && k_block_bytes == other.k_block_bytes && v_block_bytes == other.v_block_bytes
               && block_payload_bytes == other.block_payload_bytes
               && k_block_payload_bytes == other.k_block_payload_bytes
               && v_block_payload_bytes == other.v_block_payload_bytes && scale_block_bytes == other.scale_block_bytes
               && k_scale_block_bytes == other.k_scale_block_bytes && v_scale_block_bytes == other.v_scale_block_bytes
               && fingerprint == other.fingerprint;
    }
};

inline void PrintTo(const CacheSpecSemanticSnapshot& snapshot, std::ostream* os) {
    *os << "{dtype=" << static_cast<int>(snapshot.dtype) << ", block_elems=" << snapshot.block_elems
        << ", k_block_elems=" << snapshot.k_block_elems << ", v_block_elems=" << snapshot.v_block_elems
        << ", block_bytes=" << snapshot.block_bytes << ", k_block_bytes=" << snapshot.k_block_bytes
        << ", v_block_bytes=" << snapshot.v_block_bytes << ", block_payload_bytes=" << snapshot.block_payload_bytes
        << ", k_block_payload_bytes=" << snapshot.k_block_payload_bytes
        << ", v_block_payload_bytes=" << snapshot.v_block_payload_bytes
        << ", scale_block_bytes=" << snapshot.scale_block_bytes
        << ", k_scale_block_bytes=" << snapshot.k_scale_block_bytes
        << ", v_scale_block_bytes=" << snapshot.v_scale_block_bytes << ", fingerprint=" << snapshot.fingerprint << "}";
}

inline CacheSpecSemanticSnapshot makeExpectedSpecSemanticSnapshot(KVCacheSpecType type,
                                                                  DataType        dtype,
                                                                  size_t          seq_size_per_block,
                                                                  size_t          block_elems,
                                                                  size_t          k_block_elems,
                                                                  size_t          v_block_elems,
                                                                  size_t          block_bytes,
                                                                  size_t          k_block_bytes,
                                                                  size_t          v_block_bytes,
                                                                  size_t          block_payload_bytes,
                                                                  size_t          k_block_payload_bytes,
                                                                  size_t          v_block_payload_bytes,
                                                                  size_t          scale_block_bytes,
                                                                  size_t          k_scale_block_bytes,
                                                                  size_t          v_scale_block_bytes) {
    std::ostringstream fingerprint;
    fingerprint << "type=" << static_cast<int>(type) << ";dtype=" << static_cast<int>(dtype)
                << ";seq_size_per_block=" << seq_size_per_block << ";block_elems=" << block_elems
                << ";k_block_elems=" << k_block_elems << ";v_block_elems=" << v_block_elems
                << ";block_bytes=" << block_bytes << ";k_block_bytes=" << k_block_bytes
                << ";v_block_bytes=" << v_block_bytes << ";block_payload_bytes=" << block_payload_bytes
                << ";k_block_payload_bytes=" << k_block_payload_bytes
                << ";v_block_payload_bytes=" << v_block_payload_bytes << ";scale_block_bytes=" << scale_block_bytes
                << ";k_scale_block_bytes=" << k_scale_block_bytes << ";v_scale_block_bytes=" << v_scale_block_bytes;
    return {dtype,
            block_elems,
            k_block_elems,
            v_block_elems,
            block_bytes,
            k_block_bytes,
            v_block_bytes,
            block_payload_bytes,
            k_block_payload_bytes,
            v_block_payload_bytes,
            scale_block_bytes,
            k_scale_block_bytes,
            v_scale_block_bytes,
            fingerprint.str()};
}

struct CacheGroupSemanticSnapshot {
    std::string               tag;
    KVCacheSpecType           spec_type;
    CacheGroupType            group_type;
    bool                      enable_prefix_reuse;
    CacheEvictPolicy          evict_policy;
    bool                      reservable;
    uint32_t                  explicit_block_num;
    uint32_t                  active_tail_blocks;
    bool                      validate_tail_blocks;
    CpBlockMappingMode        cp_mapping;
    CpBlockSliceMode          cp_slice;
    std::vector<int>          layer_ids;
    uint32_t                  block_num;
    size_t                    physical_tokens_per_block;
    size_t                    kernel_tokens_per_block;
    size_t                    block_bytes;
    size_t                    kv_block_stride_bytes;
    size_t                    kv_scale_stride_bytes;
    uint32_t                  local_kv_head_num;
    CacheSpecSemanticSnapshot spec;

    bool operator==(const CacheGroupSemanticSnapshot& other) const {
        return tag == other.tag && spec_type == other.spec_type && group_type == other.group_type
               && enable_prefix_reuse == other.enable_prefix_reuse && evict_policy == other.evict_policy
               && reservable == other.reservable && explicit_block_num == other.explicit_block_num
               && active_tail_blocks == other.active_tail_blocks && validate_tail_blocks == other.validate_tail_blocks
               && cp_mapping == other.cp_mapping && cp_slice == other.cp_slice && layer_ids == other.layer_ids
               && block_num == other.block_num && physical_tokens_per_block == other.physical_tokens_per_block
               && kernel_tokens_per_block == other.kernel_tokens_per_block && block_bytes == other.block_bytes
               && kv_block_stride_bytes == other.kv_block_stride_bytes
               && kv_scale_stride_bytes == other.kv_scale_stride_bytes && local_kv_head_num == other.local_kv_head_num
               && spec == other.spec;
    }
};

inline void PrintTo(const CacheGroupSemanticSnapshot& snapshot, std::ostream* os) {
    *os << "{tag=" << snapshot.tag << ", spec_type=" << static_cast<int>(snapshot.spec_type)
        << ", group_type=" << static_cast<int>(snapshot.group_type)
        << ", enable_prefix_reuse=" << snapshot.enable_prefix_reuse
        << ", evict_policy=" << static_cast<int>(snapshot.evict_policy) << ", reservable=" << snapshot.reservable
        << ", explicit_block_num=" << snapshot.explicit_block_num
        << ", active_tail_blocks=" << snapshot.active_tail_blocks
        << ", validate_tail_blocks=" << snapshot.validate_tail_blocks
        << ", cp_mapping=" << static_cast<int>(snapshot.cp_mapping)
        << ", cp_slice=" << static_cast<int>(snapshot.cp_slice) << ", layer_ids={";
    for (size_t i = 0; i < snapshot.layer_ids.size(); ++i) {
        *os << (i == 0 ? "" : ",") << snapshot.layer_ids[i];
    }
    *os << "}, block_num=" << snapshot.block_num << ", physical_tokens_per_block=" << snapshot.physical_tokens_per_block
        << ", kernel_tokens_per_block=" << snapshot.kernel_tokens_per_block << ", block_bytes=" << snapshot.block_bytes
        << ", kv_block_stride_bytes=" << snapshot.kv_block_stride_bytes
        << ", kv_scale_stride_bytes=" << snapshot.kv_scale_stride_bytes
        << ", local_kv_head_num=" << snapshot.local_kv_head_num << ", spec=";
    PrintTo(snapshot.spec, os);
    *os << "}";
}

using CacheSemanticSnapshot = std::vector<CacheGroupSemanticSnapshot>;

class TestCacheConfigBuilder {
public:
    TestCacheConfigBuilder& addGroup(CacheGroup group) {
        groups_.push_back(std::move(group));
        return *this;
    }

    TestCacheConfigBuilder& setLayerTags(int layer_id, std::vector<std::string> tags) {
        RTP_LLM_CHECK_WITH_INFO(layer_id >= 0, "test cache config requires non-negative layer id");
        if (layers_.size() <= static_cast<size_t>(layer_id)) {
            layers_.resize(static_cast<size_t>(layer_id) + 1);
        }
        layers_[static_cast<size_t>(layer_id)] = {layer_id, std::move(tags)};
        return *this;
    }

    TestCacheConfigBuilder& configure(CacheConfig config) {
        base_ = std::move(config);
        return *this;
    }

    CacheConfig build() const {
        CacheConfig config   = base_;
        config.layer_num     = static_cast<uint32_t>(layers_.size());
        config.layer_all_num = config.layer_num;
        if (config.block_num == 0 && !groups_.empty()) {
            std::set<uint32_t> group_block_nums;
            for (const auto& group : groups_) {
                group_block_nums.insert(group.layout.block_num);
            }
            if (group_block_nums.size() == 1 && *group_block_nums.begin() > 0) {
                config.block_num = *group_block_nums.begin();
            }
        }
        config.setResolvedData({groups_, layers_});
        return config;
    }

    static CacheConfig withGroupBlockLayout(CacheConfig                  config,
                                            const std::vector<uint32_t>& block_nums,
                                            const std::vector<size_t>&   kv_strides,
                                            const std::vector<size_t>&   scale_strides) {
        config.setGroupBlockLayout(block_nums, kv_strides, scale_strides);
        return config;
    }

    static CacheConfig withGroupPolicies(CacheConfig config, const std::vector<CacheGroupPolicy>& policies) {
        config.setGroupPolicies(policies);
        return config;
    }

    static void setGroupBlockLayout(CacheConfig&                 config,
                                    const std::vector<uint32_t>& block_nums,
                                    const std::vector<size_t>&   kv_strides,
                                    const std::vector<size_t>&   scale_strides) {
        config.setGroupBlockLayout(block_nums, kv_strides, scale_strides);
    }

    static void setGroupPolicies(CacheConfig& config, const std::vector<CacheGroupPolicy>& policies) {
        config.setGroupPolicies(policies);
    }

    static void fromGroupedSpecs(CacheConfig&                         config,
                                 const std::vector<KVCacheSpecPtr>&   specs,
                                 const std::vector<std::vector<int>>& layers_by_group,
                                 const std::vector<CacheGroupType>&   types,
                                 const std::vector<std::string>&      tags     = {},
                                 const std::vector<CacheGroupPolicy>& policies = {}) {
        config.fromGroupedSpecs(specs, layers_by_group, types, tags, policies);
    }

    static CacheConfig
    withResolvedData(CacheConfig config, std::vector<CacheGroup> groups, std::vector<CacheLayerMembership> layers) {
        config.layer_num     = static_cast<uint32_t>(layers.size());
        config.layer_all_num = config.layer_num;
        config.setResolvedData({std::move(groups), std::move(layers)});
        return config;
    }

    static void
    setResolvedData(CacheConfig& config, std::vector<CacheGroup> groups, std::vector<CacheLayerMembership> layers) {
        config.setResolvedData({std::move(groups), std::move(layers)});
    }

    static std::shared_ptr<CacheConfig>
    mergeMTPModule(CacheConfig& config, const CacheConfig& propose, int module_index, uint32_t main_layer_num) {
        return config.mergeMTPModule(propose, module_index, main_layer_num);
    }

    static CacheConfig withMTPModules(CacheConfig config, std::vector<CacheConfig> modules) {
        config.mtp_sub_configs_.clear();
        config.mtp_sub_configs_.reserve(modules.size());
        for (auto& module : modules) {
            config.mtp_sub_configs_.push_back(std::make_shared<const CacheConfig>(std::move(module)));
        }
        return config;
    }

private:
    CacheConfig                       base_;
    std::vector<CacheGroup>           groups_;
    std::vector<CacheLayerMembership> layers_;
};

// Tag set of a cache plan. Group storage order is deterministic but carries no
// business meaning, so tests assert the tag set.
inline std::set<std::string> groupTagSet(const CacheConfig& config) {
    std::set<std::string> tags;
    for (const auto& group : config.groups()) {
        tags.insert(group.tag);
    }
    return tags;
}

// Layout shaped like `config` but with no materialized buffers. Stub and mock
// allocators use this wherever the component under test reads projected routing;
// a default-constructed GroupedCacheLayerLayout carries no routing and throws on access.
inline GroupedCacheLayerLayout makeTopologyOnlyLayerLayout(const CacheConfig& config) {
    GroupedCacheLayerLayout::GroupLayouts groups;
    for (const auto& group : config.groups()) {
        groups.emplace(group.tag, CacheLayerLayout(std::vector<BlockBufferPtrInfo>(config.layerMemberships().size())));
    }
    return GroupedCacheLayerLayout(config, std::move(groups));
}

inline CacheSemanticSnapshot snapshotCacheConfig(const CacheConfig& config) {
    CacheSemanticSnapshot snapshot;
    const auto&           groups = config.groups();
    snapshot.reserve(groups.size());
    for (const auto& group : groups) {
        RTP_LLM_CHECK_WITH_INFO(group.layout.spec != nullptr,
                                "cache semantic snapshot requires group %s to have a spec",
                                group.tag.c_str());
        const auto& policy = group.policy;
        snapshot.push_back({group.tag,
                            group.layout.spec->type,
                            policy.group_type,
                            policy.enable_prefix_reuse,
                            policy.evict_policy,
                            policy.reservable,
                            policy.explicit_block_num,
                            policy.active_tail_blocks,
                            policy.validate_tail_blocks,
                            policy.cp_mapping,
                            policy.cp_slice,
                            group.layer_ids,
                            group.layout.block_num,
                            group.layout.seq_size_per_block,
                            group.layout.kernel_seq_size_per_block,
                            config.blockSizeBytes(group.tag),
                            group.layout.kv_block_stride_bytes,
                            group.layout.kv_scale_stride_bytes,
                            group.layout.local_kv_head_num,
                            {group.layout.spec->memoryLayoutDType(),
                             group.layout.spec->block_size(),
                             group.layout.spec->k_block_size(),
                             group.layout.spec->v_block_size(),
                             group.layout.spec->block_size_bytes(),
                             group.layout.spec->k_block_size_bytes(),
                             group.layout.spec->v_block_size_bytes(),
                             group.layout.spec->block_payload_bytes(),
                             group.layout.spec->k_block_payload_bytes(),
                             group.layout.spec->v_block_payload_bytes(),
                             group.layout.spec->scale_block_size_bytes(),
                             group.layout.spec->k_scale_block_size_bytes(),
                             group.layout.spec->v_scale_block_size_bytes(),
                             group.layout.spec->fingerprint()}});
    }
    std::sort(snapshot.begin(), snapshot.end(), [](const auto& lhs, const auto& rhs) { return lhs.tag < rhs.tag; });
    return snapshot;
}

inline constexpr uint32_t DSV4_FP8_KV_ENTRY_BYTES            = 584;
inline constexpr uint32_t DSV4_FP8_INDEXER_ENTRY_BYTES       = 132;
inline constexpr size_t   DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES = 576;
inline constexpr uint32_t DSV4_SWA_WINDOW_ENTRIES            = 128;

inline size_t alignDsv4Fp8KvBlockBytes(size_t natural, size_t extra_multiple = 1) {
    const size_t align = std::lcm(DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES, std::max<size_t>(extra_multiple, 1));
    return ((natural + align - 1) / align) * align;
}

inline std::shared_ptr<MHAKVCacheSpec> makeResolvedMhaSpec(rtp_llm::DataType  dtype,
                                                           uint32_t           local_head_num_kv,
                                                           uint32_t           size_per_head,
                                                           uint32_t           seq_size_per_block,
                                                           const std::string& tag = "") {
    RTP_LLM_CHECK_WITH_INFO(local_head_num_kv > 0, "local_head_num_kv must be > 0");
    RTP_LLM_CHECK_WITH_INFO(size_per_head > 0, "size_per_head must be > 0");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be > 0");

    AttentionConfigs attn{};
    attn.kv_head_num      = static_cast<int>(local_head_num_kv);
    attn.size_per_head    = static_cast<int>(size_per_head);
    attn.tokens_per_block = seq_size_per_block;
    ParallelismConfig parallelism;
    parallelism.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag.empty() ? "default" : tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = seq_size_per_block;
    ctx.attn_config             = &attn;
    ctx.parallelism_config      = &parallelism;
    ctx.kernel_tokens_per_block = seq_size_per_block;
    return std::dynamic_pointer_cast<MHAKVCacheSpec>(SpecBuilder::build(desc, ctx).spec);
}

// Synthetic topology whose groups are named "group0".."group{group_num-1}".
// Per-layer membership is given by tag, so no positional group identity is used.
// `group_types`, when provided, is indexed the same way the names are numbered.
inline CacheConfig makeTestCacheConfigByTag(int                                          group_num,
                                            int                                          layer_num,
                                            const std::vector<std::vector<std::string>>& layer_group_tags,
                                            size_t                                       kernel_blocks_per_kv_block = 1,
                                            const std::vector<CacheGroupType>&           group_types = {}) {
    RTP_LLM_CHECK_WITH_INFO(group_num > 0, "test topology requires at least one group");
    RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "test topology requires at least one layer");
    RTP_LLM_CHECK_WITH_INFO(layer_group_tags.size() == static_cast<size_t>(layer_num),
                            "test topology layer map size=%zu layer_num=%d",
                            layer_group_tags.size(),
                            layer_num);
    RTP_LLM_CHECK_WITH_INFO(group_types.empty() || group_types.size() == static_cast<size_t>(group_num),
                            "test topology group type size=%zu group_num=%d",
                            group_types.size(),
                            group_num);

    std::vector<std::string> tags;
    tags.reserve(static_cast<size_t>(group_num));
    for (int i = 0; i < group_num; ++i) {
        tags.push_back("group" + std::to_string(i));
    }

    std::map<std::string, std::vector<int>> group_layer_ids;
    std::vector<CacheLayerMembership>       layers;
    layers.reserve(static_cast<size_t>(layer_num));
    for (int layer_id = 0; layer_id < layer_num; ++layer_id) {
        CacheLayerMembership layer;
        layer.layer_id = layer_id;
        for (const auto& tag : layer_group_tags[static_cast<size_t>(layer_id)]) {
            RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), tag) != tags.end(),
                                    "test topology unknown tag=%s for layer=%d",
                                    tag.c_str(),
                                    layer_id);
            group_layer_ids[tag].push_back(layer_id);
            layer.group_tags.push_back(tag);
        }
        layers.push_back(std::move(layer));
    }

    const size_t            blocks_per_kv_block = std::max<size_t>(1, kernel_blocks_per_kv_block);
    std::vector<CacheGroup> groups;
    groups.reserve(tags.size());
    for (size_t i = 0; i < tags.size(); ++i) {
        const auto& tag  = tags[i];
        auto        spec = makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, blocks_per_kv_block, tag);

        CacheGroup group;
        group.tag              = tag;
        group.layout.spec      = std::move(spec);
        group.policy           = defaultCacheGroupPolicy(group_types.empty() ? CacheGroupType::FULL : group_types[i]);
        group.layer_ids        = group_layer_ids[tag];
        group.layout.block_num = 16;
        group.layout.seq_size_per_block        = blocks_per_kv_block;
        group.layout.kernel_seq_size_per_block = 1;
        groups.push_back(std::move(group));
    }
    TestCacheConfigBuilder builder;
    for (auto& group : groups) {
        builder.addGroup(std::move(group));
    }
    for (auto& layer : layers) {
        builder.setLayerTags(layer.layer_id, std::move(layer.group_tags));
    }
    return builder.build();
}

inline std::shared_ptr<MLAKVCacheSpec> makeResolvedMlaSpec(rtp_llm::DataType  dtype,
                                                           uint32_t           kv_lora_rank,
                                                           uint32_t           rope_head_dim,
                                                           uint32_t           seq_size_per_block,
                                                           const std::string& tag = "") {
    RTP_LLM_CHECK_WITH_INFO(kv_lora_rank > 0, "kv_lora_rank must be > 0");
    RTP_LLM_CHECK_WITH_INFO(rope_head_dim > 0, "rope_head_dim must be > 0");
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0, "seq_size_per_block must be > 0");

    AttentionConfigs attn{};
    attn.kv_lora_rank  = static_cast<int>(kv_lora_rank);
    attn.rope_head_dim = static_cast<int>(rope_head_dim);

    KVCacheSpecDesc desc;
    desc.tag        = tag.empty() ? "mla" : tag;
    desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = seq_size_per_block;
    ctx.attn_config        = &attn;
    return std::dynamic_pointer_cast<MLAKVCacheSpec>(SpecBuilder::build(desc, ctx).spec);
}

inline std::shared_ptr<LinearKVCacheSpec>
makeResolvedLinearSpec(rtp_llm::DataType  dtype,
                       uint32_t           local_num_k_heads,
                       uint32_t           local_num_v_heads,
                       uint32_t           head_k_dim,
                       uint32_t           head_v_dim,
                       uint32_t           conv_kernel_dim,
                       uint32_t           seq_size_per_block,
                       rtp_llm::DataType  ssm_state_dtype  = rtp_llm::DataType::TYPE_INVALID,
                       rtp_llm::DataType  conv_state_dtype = rtp_llm::DataType::TYPE_INVALID,
                       const std::string& tag              = "") {
    RTP_LLM_CHECK_WITH_INFO(local_num_k_heads > 0 && local_num_v_heads > 0, "linear head counts must be > 0");
    RTP_LLM_CHECK_WITH_INFO(head_k_dim > 0 && head_v_dim > 0, "linear head dims must be > 0");
    RTP_LLM_CHECK_WITH_INFO(conv_kernel_dim > 1, "conv_kernel_dim must be > 1");

    LinearAttentionConfig linear{};
    linear.linear_num_key_heads   = static_cast<int>(local_num_k_heads);
    linear.linear_num_value_heads = static_cast<int>(local_num_v_heads);
    linear.linear_key_head_dim    = static_cast<int>(head_k_dim);
    linear.linear_value_head_dim  = static_cast<int>(head_v_dim);
    linear.linear_conv_kernel_dim = static_cast<int>(conv_kernel_dim);
    linear.ssm_state_dtype        = ssm_state_dtype == rtp_llm::DataType::TYPE_INVALID ? dtype : ssm_state_dtype;
    linear.conv_state_dtype       = conv_state_dtype == rtp_llm::DataType::TYPE_INVALID ? dtype : conv_state_dtype;
    ParallelismConfig parallelism;
    parallelism.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag.empty() ? "linear" : tag;
    desc.cache_type = KVCacheSpecType::LinearAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = seq_size_per_block;
    ctx.linear_attention_config = &linear;
    ctx.parallelism_config      = &parallelism;
    ctx.kernel_tokens_per_block = seq_size_per_block;
    return std::dynamic_pointer_cast<LinearKVCacheSpec>(SpecBuilder::build(desc, ctx).spec);
}

inline KVCacheSpecPtr makeResolvedOpaqueSpec(bool               state_cache,
                                             const std::string& tag,
                                             rtp_llm::DataType  dtype,
                                             size_t             block_bytes,
                                             uint32_t           seq_size_per_block) {
    const size_t dtype_size = getTypeSize(dtype);
    RTP_LLM_CHECK_WITH_INFO(dtype_size > 0, "invalid dtype=%d", static_cast<int>(dtype));
    RTP_LLM_CHECK_WITH_INFO(block_bytes % dtype_size == 0,
                            "opaque block_bytes=%zu must be divisible by dtype size=%zu",
                            block_bytes,
                            dtype_size);
    const auto block_elems = static_cast<uint32_t>(block_bytes / dtype_size);

    KVCacheSpecDesc desc;
    desc.tag                         = tag.empty() ? "opaque" : tag;
    desc.cache_type                  = state_cache ? KVCacheSpecType::OpaqueState : KVCacheSpecType::OpaqueKV;
    desc.dtype                       = dtype;
    desc.entry_dtype                 = dtype;
    desc.entry_elems                 = 1;
    desc.explicit_entry_count        = block_elems;
    desc.block_stride_bytes_override = block_bytes;
    desc.is_state_cache              = state_cache;

    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = seq_size_per_block;
    return SpecBuilder::build(desc, ctx).spec;
}

inline KVCacheSpecDesc makeDsv4Desc(const std::string& tag,
                                    const std::string& kind,
                                    uint32_t           entry_elems,
                                    DataType           dtype,
                                    uint32_t           compression_ratio = 1) {
    KVCacheSpecDesc desc;
    desc.tag         = tag;
    desc.dtype       = dtype;
    desc.entry_elems = entry_elems;
    desc.entry_dtype = dtype;
    if (kind == "compressed_kv") {
        desc.cache_type                        = KVCacheSpecType::OpaqueKV;
        desc.is_state_cache                    = false;
        desc.entry_count_mode                  = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
        desc.compression_ratio                 = compression_ratio;
        desc.kernel_tokens_per_block_alignment = 128;
        if (desc.entry_elems == DSV4_FP8_KV_ENTRY_BYTES) {
            desc.block_stride_bytes_alignment = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES;
        }
        return desc;
    }

    desc.cache_type          = KVCacheSpecType::OpaqueState;
    desc.is_state_cache      = true;
    desc.entry_count_mode    = OpaqueBlockEntryCountMode::STATE_RING;
    desc.reuse               = CacheReusePolicyDesc{};
    desc.reuse->evict_policy = CacheEvictPolicy::INDEPENDENT;
    desc.cp                  = CacheCpPolicyDesc{};
    if (desc.tag == "indexer_state" || desc.tag == "csa_state") {
        desc.compression_ratio        = 4;
        desc.state_ring_overlap       = 1;
        desc.cp->align_payload        = true;
        desc.cp->prefill_slice_layout = CpPrefillSliceLayout::PAYLOAD;
        desc.cp->slice                = CpBlockSliceMode::PAYLOAD_BYTES;
    } else if (desc.tag == "hca_state") {
        desc.compression_ratio            = 128;
        desc.cp->align_payload            = true;
        desc.cp->prefill_slice_layout     = CpPrefillSliceLayout::PAYLOAD;
        desc.cp->slice                    = CpBlockSliceMode::PAYLOAD_BYTES;
        desc.capacity                     = CacheCapacityPolicyDesc{};
        desc.capacity->explicit_block_num = 256;
        desc.reuse->enable_prefix_reuse   = false;
        desc.tail                         = CacheTailPolicyDesc{};
        desc.tail->active_tail_blocks     = 1;
        desc.tail->validate_tail_blocks   = false;
    } else if (desc.tag == "swa_kv") {
        desc.compression_ratio        = DSV4_SWA_WINDOW_ENTRIES;
        desc.cp->align_payload        = true;
        desc.cp->prefill_slice_layout = CpPrefillSliceLayout::BLOCK_STRIDE;
        desc.cp->slice                = CpBlockSliceMode::EQUAL_BYTES;
        if (desc.entry_elems == DSV4_FP8_KV_ENTRY_BYTES) {
            desc.block_stride_bytes_alignment = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES;
        }
    }
    desc.state_ring_include_gen_num_per_cycle = true;
    desc.cp->scale_seq_size                   = true;
    desc.block_stride_alignment_min_entries   = DSV4_SWA_WINDOW_ENTRIES;
    return desc;
}

inline void setDefaultKvCacheSpec(ModelConfig& model_config) {
    KVCacheSpecDesc desc;
    desc.tag = "default";
    if (model_config.attn_config.use_mla && model_config.mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;
    } else {
        desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    }
    model_config.kv_cache_spec_descs.assign(static_cast<size_t>(model_config.num_layers), {desc});
}

inline void setHybridAttentionKvCacheSpecs(ModelConfig& model_config) {
    std::vector<int> full_layers;
    std::vector<int> swa_layers;
    std::vector<int> linear_layers;
    const auto&      types = model_config.hybrid_attention_config.hybrid_attention_types;
    RTP_LLM_CHECK_WITH_INFO(types.size() == static_cast<size_t>(model_config.num_layers),
                            "hybrid_attention_types size %zu != num_layers %ld",
                            types.size(),
                            model_config.num_layers);
    for (int i = 0; i < static_cast<int>(model_config.num_layers); ++i) {
        switch (types[static_cast<size_t>(i)]) {
            case HybridAttentionType::LINEAR:
                linear_layers.push_back(i);
                break;
            case HybridAttentionType::SLIDING_WINDOW:
                swa_layers.push_back(i);
                break;
            case HybridAttentionType::NONE:
            default:
                full_layers.push_back(i);
                break;
        }
    }

    KVCacheSpecDesc full_desc;
    full_desc.tag        = "full";
    full_desc.cache_type = KVCacheSpecType::MultiHeadAttention;

    KVCacheSpecDesc swa_desc = full_desc;
    swa_desc.tag             = "swa";
    swa_desc.cache_type      = KVCacheSpecType::OpaqueState;
    swa_desc.entry_elems     = static_cast<uint32_t>(model_config.attn_config.size_per_head)
                           * static_cast<uint32_t>(model_config.attn_config.kv_head_num) * 2;
    swa_desc.explicit_entry_count = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    swa_desc.entry_dtype          = DataType::TYPE_FP16;

    KVCacheSpecDesc linear_desc;
    linear_desc.tag        = "linear";
    linear_desc.cache_type = KVCacheSpecType::LinearAttention;

    model_config.kv_cache_spec_descs.assign(static_cast<size_t>(model_config.num_layers), {});
    for (int layer_id : full_layers) {
        model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)] = {full_desc};
    }
    for (int layer_id : swa_layers) {
        model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)] = {swa_desc};
    }
    for (int layer_id : linear_layers) {
        model_config.kv_cache_spec_descs[static_cast<size_t>(layer_id)] = {linear_desc};
    }
}

inline void setDsv4KvCacheSpecs(ModelConfig& model_config, const std::vector<int>& layer_compress_ratios) {
    const int layer_num = static_cast<int>(model_config.num_layers);
    model_config.hybrid_attention_config.hybrid_attention_types.assign(static_cast<size_t>(layer_num),
                                                                       HybridAttentionType::NONE);

    const bool     fp8_kv = model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8;
    const uint32_t kv_entry_elems =
        fp8_kv ? DSV4_FP8_KV_ENTRY_BYTES : static_cast<uint32_t>(model_config.attn_config.size_per_head) * 2;
    const uint32_t indexer_entry_elems =
        fp8_kv ? DSV4_FP8_INDEXER_ENTRY_BYTES : static_cast<uint32_t>(model_config.attn_config.indexer_head_dim) * 2;
    const uint32_t head_dim         = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    const uint32_t indexer_head_dim = static_cast<uint32_t>(model_config.attn_config.indexer_head_dim);

    auto csa_kv        = makeDsv4Desc("csa_kv", "compressed_kv", kv_entry_elems, DataType::TYPE_UINT8, 4);
    auto hca_kv        = makeDsv4Desc("hca_kv", "compressed_kv", kv_entry_elems, DataType::TYPE_UINT8, 128);
    auto indexer_kv    = makeDsv4Desc("indexer_kv", "compressed_kv", indexer_entry_elems, DataType::TYPE_UINT8, 4);
    auto indexer_state = makeDsv4Desc("indexer_state", "fixed_state", 4 * indexer_head_dim, DataType::TYPE_FP32);
    auto csa_state     = makeDsv4Desc("csa_state", "fixed_state", 4 * head_dim, DataType::TYPE_FP32);
    auto hca_state     = makeDsv4Desc("hca_state", "fixed_state", 2 * head_dim, DataType::TYPE_FP32);
    auto swa_kv        = makeDsv4Desc("swa_kv", "sliding_window_kv", kv_entry_elems, DataType::TYPE_UINT8);

    model_config.kv_cache_spec_descs.clear();
    model_config.kv_cache_spec_descs.resize(static_cast<size_t>(layer_num));
    for (int i = 0; i < layer_num; ++i) {
        const int ratio =
            i < static_cast<int>(layer_compress_ratios.size()) ? layer_compress_ratios[static_cast<size_t>(i)] : 0;
        if (ratio == 4) {
            model_config.kv_cache_spec_descs[static_cast<size_t>(i)] = {
                csa_kv, indexer_kv, indexer_state, csa_state, swa_kv};
        } else if (ratio == 128) {
            model_config.kv_cache_spec_descs[static_cast<size_t>(i)] = {hca_kv, hca_state, swa_kv};
        } else {
            model_config.kv_cache_spec_descs[static_cast<size_t>(i)] = {swa_kv};
        }
    }
}

inline void setDsv4ExplicitPoolBlocks(ModelConfig& model_config, const std::string& tag, uint32_t block_num) {
    for (auto& descs : model_config.kv_cache_spec_descs) {
        for (auto& desc : descs) {
            if (desc.tag == tag) {
                if (!desc.capacity.has_value()) {
                    desc.capacity = CacheCapacityPolicyDesc{};
                }
                desc.capacity->explicit_block_num = block_num;
            }
        }
    }
}

inline KVCacheSpecPtr makeMhaSpec(const std::string& tag,
                                  size_t             tokens_per_block,
                                  rtp_llm::DataType  dtype,
                                  uint32_t           local_head_num_kv,
                                  uint32_t           size_per_head) {
    AttentionConfigs attn_config;
    attn_config.kv_head_num      = local_head_num_kv;
    attn_config.size_per_head    = size_per_head;
    attn_config.tokens_per_block = static_cast<uint32_t>(tokens_per_block);

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = static_cast<uint32_t>(tokens_per_block);
    ctx.attn_config        = &attn_config;
    ctx.parallelism_config = &parallelism_config;
    return SpecBuilder::build(desc, ctx).spec;
}

inline KVCacheSpecPtr makeLinearSpec(const std::string& tag,
                                     size_t             tokens_per_block,
                                     rtp_llm::DataType  dtype,
                                     uint32_t           local_head_num_kv,
                                     uint32_t           size_per_head) {
    LinearAttentionConfig linear_config;
    linear_config.linear_conv_kernel_dim = 2;
    linear_config.linear_key_head_dim    = static_cast<int>(size_per_head);
    linear_config.linear_value_head_dim  = static_cast<int>(size_per_head);
    linear_config.linear_num_key_heads   = static_cast<int>(local_head_num_kv);
    linear_config.linear_num_value_heads = static_cast<int>(local_head_num_kv);

    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;

    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::LinearAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype                   = dtype;
    ctx.seq_size_per_block      = static_cast<uint32_t>(tokens_per_block);
    ctx.linear_attention_config = &linear_config;
    ctx.parallelism_config      = &parallelism_config;
    return SpecBuilder::build(desc, ctx).spec;
}

inline CacheConfig buildTestCacheConfigFromGroupedSpecs(CacheConfig                          base,
                                                        const std::vector<KVCacheSpecPtr>&   specs,
                                                        const std::vector<std::vector<int>>& layers_by_group,
                                                        const std::vector<CacheGroupType>&   types,
                                                        const std::vector<std::string>&      tags,
                                                        const std::vector<CacheGroupPolicy>& policies = {}) {
    RTP_LLM_CHECK_WITH_INFO(specs.size() == layers_by_group.size() && specs.size() == types.size()
                                && specs.size() == tags.size(),
                            "test grouped cache config inputs must have equal sizes");
    const size_t           layer_num = base.layer_num;
    TestCacheConfigBuilder builder;
    builder.configure(std::move(base));
    std::vector<std::vector<std::string>> layer_tags(layer_num);
    for (size_t idx = 0; idx < specs.size(); ++idx) {
        CacheGroup group;
        group.tag         = tags[idx];
        group.layout.spec = specs[idx];
        group.policy      = policies.empty() ? defaultCacheGroupPolicy(types[idx]) : policies[idx];
        group.layer_ids   = layers_by_group[idx];
        for (int layer_id : group.layer_ids) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layer_tags.size(),
                                    "test group tag=%s has invalid layer=%d",
                                    group.tag.c_str(),
                                    layer_id);
            layer_tags[static_cast<size_t>(layer_id)].push_back(group.tag);
        }
        builder.addGroup(std::move(group));
    }
    for (size_t layer_id = 0; layer_id < layer_tags.size(); ++layer_id) {
        builder.setLayerTags(static_cast<int>(layer_id), std::move(layer_tags[layer_id]));
    }
    return builder.build();
}

inline CacheConfig makeSingleGroupCacheConfig(
    KVCacheSpecPtr spec, CacheGroupType group_type, int layer_num, int block_num, std::string tag) {
    CacheConfig config;
    config.dtype                     = spec->memoryLayoutDType();
    config.layer_num                 = static_cast<uint32_t>(layer_num);
    config.layer_all_num             = static_cast<uint32_t>(layer_num);
    config.block_num                 = static_cast<uint32_t>(block_num);
    config.seq_size_per_block        = spec->seq_size_per_block;
    config.kernel_seq_size_per_block = spec->seq_size_per_block;

    std::vector<int> layer_ids(static_cast<size_t>(layer_num));
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    config =
        buildTestCacheConfigFromGroupedSpecs(std::move(config), {spec}, {layer_ids}, {group_type}, {std::move(tag)});

    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));
    return config;
}

inline CacheConfig
makeSingleLayerCacheConfig(KVCacheSpecPtr spec, CacheGroupType group_type, std::string tag, int block_num = 4) {
    auto config = makeSingleGroupCacheConfig(std::move(spec), group_type, /*layer_num=*/1, block_num, std::move(tag));
    config.group_layer_num = 1;
    return config;
}

inline CacheConfig makeSimpleMhaCacheConfig(int               layer_num,
                                            int               block_num,
                                            size_t            tokens_per_block,
                                            rtp_llm::DataType dtype,
                                            uint32_t          local_head_num_kv = 1,
                                            uint32_t          size_per_head     = 1) {
    auto spec = makeMhaSpec("default", tokens_per_block, dtype, local_head_num_kv, size_per_head);
    return makeSingleGroupCacheConfig(std::move(spec), CacheGroupType::FULL, layer_num, block_num, "default");
}

inline CacheConfig makeSimpleLinearCacheConfig(int               layer_num,
                                               int               block_num,
                                               size_t            tokens_per_block,
                                               rtp_llm::DataType dtype,
                                               uint32_t          local_head_num_kv = 1,
                                               uint32_t          size_per_head     = 1) {
    auto spec = makeLinearSpec("linear", tokens_per_block, dtype, local_head_num_kv, size_per_head);
    return makeSingleGroupCacheConfig(std::move(spec), CacheGroupType::LINEAR, layer_num, block_num, "linear");
}

inline CacheConfig makeSimpleHybridMhaCacheConfig(int               layer_num,
                                                  int               block_num,
                                                  size_t            tokens_per_block,
                                                  rtp_llm::DataType dtype,
                                                  int               group_layer_num   = 2,
                                                  uint32_t          local_head_num_kv = 1,
                                                  uint32_t          size_per_head     = 1) {
    CacheConfig config;
    config.dtype                     = dtype;
    config.layer_num                 = static_cast<uint32_t>(layer_num);
    config.layer_all_num             = static_cast<uint32_t>(layer_num);
    config.block_num                 = static_cast<uint32_t>(block_num);
    config.seq_size_per_block        = tokens_per_block;
    config.kernel_seq_size_per_block = tokens_per_block;
    config.group_layer_num           = std::max(group_layer_num, 1);
    config.linear_step               = 2;

    if (layer_num <= 0 || (layer_num % config.group_layer_num) != 0 || (layer_num / config.group_layer_num) < 2) {
        return makeSimpleMhaCacheConfig(
            layer_num, block_num, tokens_per_block, dtype, local_head_num_kv, size_per_head);
    }

    const int group_cnt = layer_num / config.group_layer_num;

    auto linear_spec = makeLinearSpec("linear", tokens_per_block, dtype, local_head_num_kv, size_per_head);
    auto full_spec   = makeMhaSpec("full", tokens_per_block, dtype, local_head_num_kv, size_per_head);

    std::vector<KVCacheSpecPtr>   specs;
    std::vector<std::vector<int>> layers_by_group;
    std::vector<CacheGroupType>   types;
    std::vector<std::string>      tags;
    specs.reserve(static_cast<size_t>(group_cnt));
    layers_by_group.reserve(static_cast<size_t>(group_cnt));
    types.reserve(static_cast<size_t>(group_cnt));
    tags.reserve(static_cast<size_t>(group_cnt));

    // The first declared group is the linear one; the rest are full groups named
    // after their declaration ordinal, which is only how the tag is spelled.
    for (int ordinal = 0; ordinal < group_cnt; ++ordinal) {
        std::vector<int> group_layers;
        group_layers.reserve(static_cast<size_t>(config.group_layer_num));
        for (int local = 0; local < config.group_layer_num; ++local) {
            group_layers.push_back(ordinal * config.group_layer_num + local);
        }
        if (ordinal == 0) {
            specs.push_back(linear_spec);
            types.push_back(CacheGroupType::LINEAR);
            tags.push_back("linear");
        } else {
            specs.push_back(full_spec);
            types.push_back(CacheGroupType::FULL);
            tags.push_back("full" + std::to_string(ordinal));
        }
        layers_by_group.push_back(std::move(group_layers));
    }
    config = buildTestCacheConfigFromGroupedSpecs(std::move(config), specs, layers_by_group, types, tags);

    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = full_spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));
    return config;
}

}  // namespace test
}  // namespace rtp_llm
