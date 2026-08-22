#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

struct CacheGroupLayout {
    std::shared_ptr<const KVCacheSpec> spec;
    uint32_t                          block_num                 = 0;
    uint32_t                          local_kv_head_num         = 1;
    size_t                            seq_size_per_block        = 0;
    size_t                            kernel_seq_size_per_block = 0;
    size_t                            kv_block_stride_bytes     = 0;
    size_t                            kv_scale_stride_bytes     = 0;

    size_t kernelBlocksPerPoolBlock() const;
};

struct CacheGroup {
    std::string      tag;
    CacheGroupLayout layout;
    CacheGroupPolicy policy;
    std::vector<int> layer_ids;
};

struct CacheLayerMembership {
    int                      layer_id = -1;
    std::vector<std::string> group_tags;
};

struct ResolvedCacheConfigData {
    std::vector<CacheGroup>           groups;
    std::vector<CacheLayerMembership> layers;
};

class CacheConfigResolverAccess;
namespace test {
class TestCacheConfigBuilder;
}

struct CacheConfig {
public:
    std::vector<int> layer_to_block_stride_bytes;
    bool             group_block_layout_initialized           = false;
    bool             use_independent_block_pools              = false;
    bool             use_typed_cache_regions                  = false;
    bool             use_opaque_kv_cache_store                = false;
    bool             disable_decode_first_malloc_device_reuse = false;

    rtp_llm::DataType dtype         = rtp_llm::DataType::TYPE_INVALID;
    uint32_t          layer_num     = 0;
    uint32_t          layer_all_num = 0;
    bool              use_mla       = false;
    bool              is_sparse     = false;

    uint32_t block_num                 = 0;
    size_t   seq_size_per_block        = 1;
    size_t   kernel_seq_size_per_block = 0;

    size_t kernelBlocksPerKvBlock(std::string_view tag) const {
        return group(tag).layout.kernelBlocksPerPoolBlock();
    }

    size_t kernelBlocksPerKvBlock() const {
        if (kernel_seq_size_per_block == 0) {
            return 1;
        }
        RTP_LLM_CHECK_WITH_INFO(seq_size_per_block % kernel_seq_size_per_block == 0,
                                "seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu)",
                                seq_size_per_block,
                                kernel_seq_size_per_block);
        return std::max<size_t>(1, seq_size_per_block / kernel_seq_size_per_block);
    }

    size_t kv_block_size_bytes = 0;
    size_t kv_scale_size_bytes = 0;
    size_t block_size_bytes    = 0;

    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride_bytes = 0;

    int    linear_step     = 1;
    int    group_layer_num = 1;
    size_t explicitly_sized_pool_reserve_bytes = 0;

    CacheConfig() = default;

    static uint32_t
    mtpGlobalLayerId(uint32_t main_layer_num, int module_index, uint32_t module_layer_num, int local_layer_id) {
        constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
        if (module_index < 0 || module_layer_num == 0 || local_layer_id < 0
            || static_cast<uint32_t>(local_layer_id) >= module_layer_num) {
            return invalid;
        }
        const uint64_t global_layer_id = static_cast<uint64_t>(main_layer_num)
                                         + static_cast<uint64_t>(module_index) * module_layer_num
                                         + static_cast<uint32_t>(local_layer_id);
        return global_layer_id < invalid ? static_cast<uint32_t>(global_layer_id) : invalid;
    }

    int groupNums() const noexcept {
        return static_cast<int>(groups_.size());
    }

    const std::vector<CacheGroup>& groups() const noexcept {
        return groups_;
    }

    const std::vector<CacheLayerMembership>& layerMemberships() const noexcept {
        return layers_;
    }

    const CacheGroup& group(std::string_view tag) const;
    const std::vector<std::string>& groupTagsForLayer(int layer_id) const;
    const CacheGroup& groupForLayer(int layer_id, std::string_view tag) const;
    const CacheGroup& soleGroupForLayer(int layer_id) const;
    bool              hasSingleGlobalGroup() const noexcept;
    bool              hasOneGroupPerLayer() const noexcept;
    size_t            mtpModuleCount() const noexcept;
    const CacheConfig& mtpModule(size_t module_index) const;

    size_t blockSizeBytes(std::string_view tag) const {
        const auto& group_config = group(tag);
        return group_config.layer_ids.size()
               * (group_config.layout.kv_block_stride_bytes + group_config.layout.kv_scale_stride_bytes);
    }

    uint32_t localKvHeadNum(std::string_view tag) const {
        const auto& group_config = group(tag);
        RTP_LLM_CHECK_WITH_INFO(group_config.layout.local_kv_head_num > 0,
                                "CacheConfig::localKvHeadNum invalid local_kv_head_num=%u tag=%s",
                                group_config.layout.local_kv_head_num,
                                group_config.tag.c_str());
        return group_config.layout.local_kv_head_num;
    }

    uint32_t explicitIndependentBlocks(std::string_view tag) const {
        return group(tag).policy.explicit_block_num;
    }

    bool usesExplicitIndependentBlocks(std::string_view tag) const {
        return explicitIndependentBlocks(tag) > 0;
    }

    static bool samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs);
    std::string debugString(size_t indent = 0) const;

private:
    friend class CacheConfigResolverAccess;
    friend class test::TestCacheConfigBuilder;

    void setResolvedData(ResolvedCacheConfigData data);
    void setGroupPolicies(const std::vector<CacheGroupPolicy>& policies);
    void setGroupBlockLayout(const std::vector<uint32_t>& block_nums,
                             const std::vector<size_t>&   kv_block_stride_bytes,
                             const std::vector<size_t>&   kv_scale_stride_bytes);
    std::shared_ptr<CacheConfig>
    mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num);
    void fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                          const std::vector<std::vector<int>>& layers_by_group,
                          const std::vector<CacheGroupType>&   types,
                          const std::vector<std::string>&      tags     = {},
                          const std::vector<CacheGroupPolicy>& policies = {});
    void finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config);
    CacheConfig withFinalizedBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config) const;
    void validateAndBuildIndex();

    std::vector<CacheGroup>                 groups_;
    std::vector<CacheLayerMembership>       layers_;
    std::unordered_map<std::string, size_t> tag_to_idx_;
    std::vector<std::shared_ptr<const CacheConfig>> mtp_sub_configs_;
};

}  // namespace rtp_llm
