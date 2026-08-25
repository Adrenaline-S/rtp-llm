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

struct CacheGroup {
    std::string                        tag;
    std::shared_ptr<const KVCacheSpec> spec;
    CacheGroupPolicy                   policy;
    std::vector<int>                   layer_ids;
    uint32_t                           block_num                 = 0;
    uint32_t                           local_kv_head_num         = 1;
    size_t                             seq_size_per_block        = 0;
    size_t                             kernel_seq_size_per_block = 0;
    size_t                             kv_block_stride_bytes     = 0;
    size_t                             kv_scale_stride_bytes     = 0;

    size_t kernelBlocksPerPoolBlock() const;
    bool   requiresWholeBlockTransfer() const;
};

using CacheLayer = std::vector<std::string>;

class CacheConfigCreator;
namespace test {
class TestCacheConfigBuilder;
}

struct CacheConfig {
public:
    rtp_llm::DataType dtype() const noexcept {
        return dtype_;
    }
    uint32_t mainLayerCount() const noexcept {
        return main_layer_count_;
    }
    size_t layerCount() const noexcept {
        return layers_.size();
    }
    bool usesMla() const noexcept {
        return uses_mla_;
    }
    bool isSparse() const noexcept {
        return is_sparse_;
    }
    uint32_t blockCountBasis() const noexcept {
        return block_count_basis_;
    }
    size_t cacheKeyBlockTokens() const noexcept {
        return cache_key_block_tokens_;
    }
    size_t kernelBlockTokens() const noexcept {
        return kernel_block_tokens_;
    }
    size_t pagedBlockBudgetBytes() const noexcept {
        return paged_block_budget_bytes_;
    }
    int linearStep() const noexcept {
        return linear_step_;
    }
    size_t explicitPoolReserveBytes() const noexcept {
        return explicit_pool_reserve_bytes_;
    }
    bool usesTypedCacheRegions() const noexcept {
        return uses_typed_cache_regions_;
    }
    bool usesOpaqueKVCacheStore() const noexcept {
        return uses_opaque_kv_cache_store_;
    }
    bool disablesDecodeFirstMallocDeviceReuse() const noexcept {
        return disables_decode_first_malloc_device_reuse_;
    }

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

    const std::vector<CacheLayer>& layers() const noexcept {
        return layers_;
    }

    const CacheGroup&               group(std::string_view tag) const;
    const std::vector<std::string>& groupTagsForLayer(int layer_id) const;
    const CacheGroup&               groupForLayer(int layer_id, std::string_view tag) const;
    const CacheGroup&               physicalGroupForGlobalLayer(int layer_id, std::string_view tag) const;
    const CacheGroup&               soleGroupForLayer(int layer_id) const;
    bool                            hasSingleGlobalGroup() const noexcept;
    bool                            hasOneGroupPerLayer() const noexcept;
    bool                            usesSingleFullAttentionContract() const noexcept;
    size_t                          mtpModuleCount() const noexcept;
    const CacheConfig&              mtpModule(size_t module_index) const;

    size_t blockSizeBytes(std::string_view tag) const;

    uint32_t localKvHeadNum(std::string_view tag) const {
        const auto& group_config = group(tag);
        RTP_LLM_CHECK_WITH_INFO(group_config.local_kv_head_num > 0,
                                "CacheConfig::localKvHeadNum invalid local_kv_head_num=%u tag=%s",
                                group_config.local_kv_head_num,
                                group_config.tag.c_str());
        return group_config.local_kv_head_num;
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
    friend class CacheConfigCreator;
    friend class test::TestCacheConfigBuilder;

    CacheConfig() = default;

    void replaceAssemblyTopology(std::vector<CacheGroup> groups, std::vector<CacheLayer> layers);
    void replaceAssemblyGroupBlockLayout(const std::vector<uint32_t>& block_nums,
                                         const std::vector<size_t>&   kv_block_stride_bytes,
                                         const std::vector<size_t>&   kv_scale_stride_bytes);
    std::shared_ptr<CacheConfig>
         composeAssemblyMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num);
    void projectAssemblyBlockCounts(uint32_t block_count_basis);
    void validateAndBuildIndex();

    std::vector<CacheGroup>                         groups_;
    std::vector<CacheLayer>                         layers_;
    std::unordered_map<std::string, size_t>         tag_to_idx_;
    std::vector<std::shared_ptr<const CacheConfig>> mtp_sub_configs_;

    bool              group_block_layout_initialized_            = false;
    bool              uses_typed_cache_regions_                  = false;
    bool              uses_opaque_kv_cache_store_                = false;
    bool              disables_decode_first_malloc_device_reuse_ = false;
    rtp_llm::DataType dtype_                                     = rtp_llm::DataType::TYPE_INVALID;
    uint32_t          main_layer_count_                          = 0;
    bool              uses_mla_                                  = false;
    bool              is_sparse_                                 = false;
    uint32_t          block_count_basis_                         = 0;
    size_t            cache_key_block_tokens_                    = 1;
    size_t            kernel_block_tokens_                       = 0;
    size_t            paged_block_budget_bytes_                  = 0;
    int               linear_step_                               = 1;
    size_t            explicit_pool_reserve_bytes_               = 0;
};

}  // namespace rtp_llm
