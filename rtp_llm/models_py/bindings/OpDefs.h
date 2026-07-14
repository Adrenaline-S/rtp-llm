#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/models_py/bindings/ParamsBase.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

// Forward declare for opaque pointers in PyCacheStoreInputs
namespace rtp_llm {
class CacheStore;
class CacheStoreAsyncWriter;
}  // namespace rtp_llm

namespace torch_ext {

// Per-layer KV cache view. Returned by KVCache::getLayerCache().
// When kernel_seq_size_per_block < seq_size_per_block the tensor is presented at
// kernel-block granularity:
//   MHA: [kernel_block_num, 2, num_kv_heads, kernel_seq_size_per_block, head_dim]
//   MLA: [kernel_block_num, kernel_seq_size_per_block, physical_elements_per_token]
struct LayerKVCache {
    torch::Tensor kv_cache_base;
    torch::Tensor kv_scale_base;
    int           seq_size_per_block = 0;
    int           layer_id           = -1;
    std::string   tag;
};

// Whole-model KV cache holding tensors for all layers.
// Call getLayerCache(global_layer_id) to obtain a per-layer LayerKVCache.
struct KVCache {
    // Per-layer views
    std::vector<torch::Tensor> kv_cache_base_by_layer;
    std::vector<torch::Tensor> kv_scale_base_by_layer;
    int                        seq_size_per_block        = 0;
    int                        kernel_seq_size_per_block = 0;
    int                        num_kv_heads              = 0;
    int                        head_dim                  = 0;
    bool                       use_mla                   = false;
    int                        kv_lora_rank              = 0;
    int                        rope_head_dim             = 0;

    // Per-layer attention type (CacheGroupType::FULL or LINEAR).
    std::vector<rtp_llm::CacheGroupType> layer_attn_types;

    // Per-group topology from CacheLayerLayout.
    std::vector<rtp_llm::CacheGroupType>    group_types;
    std::vector<size_t>                     group_seq_block_sizes;
    std::vector<size_t>                     group_kernel_seq_block_sizes;
    std::vector<size_t>                     group_kernel_blocks_per_kv_block;
    std::vector<std::string>                group_tags;
    std::vector<std::vector<int>>           layer_to_group_ids;
    std::vector<std::map<std::string, int>> layer_tag_to_group_id;
    std::vector<std::vector<torch::Tensor>> kv_cache_base_by_layer_group;
    std::vector<std::vector<torch::Tensor>> kv_scale_base_by_layer_group;

    int groupSeqBlockSize(int gid) const {
        if (gid >= 0 && static_cast<size_t>(gid) < group_seq_block_sizes.size() && group_seq_block_sizes[gid] > 0) {
            return static_cast<int>(group_seq_block_sizes[gid]);
        }
        return seq_size_per_block;
    }

    int groupKernelSeqBlockSize(int gid) const {
        if (gid >= 0 && static_cast<size_t>(gid) < group_kernel_seq_block_sizes.size()
            && group_kernel_seq_block_sizes[gid] > 0) {
            return static_cast<int>(group_kernel_seq_block_sizes[gid]);
        }
        return kernel_seq_size_per_block > 0 ? kernel_seq_size_per_block : groupSeqBlockSize(gid);
    }

    int groupKernelBlocksPerKvBlock(int gid) const {
        if (gid >= 0 && static_cast<size_t>(gid) < group_kernel_blocks_per_kv_block.size()
            && group_kernel_blocks_per_kv_block[gid] > 0) {
            return static_cast<int>(group_kernel_blocks_per_kv_block[gid]);
        }
        const auto group_kernel = groupKernelSeqBlockSize(gid);
        const auto group_seq    = groupSeqBlockSize(gid);
        return group_kernel > 0 ? std::max(1, group_seq / group_kernel) : 1;
    }

    void setFullAttentionView(
        LayerKVCache& layer_cache, const torch::Tensor& base, const torch::Tensor& scale, int gid) const {
        const int physical_seq_size = groupSeqBlockSize(gid);
        RTP_LLM_CHECK_WITH_INFO(physical_seq_size > 0,
                                "physical seq_size_per_block must be positive, got %d",
                                physical_seq_size);
        layer_cache.seq_size_per_block = groupKernelSeqBlockSize(gid);
        RTP_LLM_CHECK_WITH_INFO(layer_cache.seq_size_per_block > 0,
                                "kernel seq_size_per_block must be positive, got %d",
                                layer_cache.seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(physical_seq_size % layer_cache.seq_size_per_block == 0,
                                "physical seq_size_per_block=%d must be divisible by kernel seq_size_per_block=%d",
                                physical_seq_size,
                                layer_cache.seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(base.defined() && base.dim() > 0,
                                "full-attention KV cache base must be a defined tensor with at least one dimension");

        const int64_t physical_block_num = base.size(0);
        const int64_t kernel_blocks_per_kv_block = groupKernelBlocksPerKvBlock(gid);
        const int64_t kernel_block_num = physical_block_num * kernel_blocks_per_kv_block;

        if (use_mla) {
            RTP_LLM_CHECK_WITH_INFO(base.is_contiguous(), "MLA KV cache base must be contiguous");
            const int64_t elements_per_kernel_page =
                kernel_block_num * static_cast<int64_t>(layer_cache.seq_size_per_block);
            RTP_LLM_CHECK_WITH_INFO(elements_per_kernel_page > 0 && base.numel() % elements_per_kernel_page == 0,
                                    "MLA KV cache elements=%ld must be divisible by kernel pages=%ld",
                                    base.numel(),
                                    elements_per_kernel_page);
            const int64_t stride = base.numel() / elements_per_kernel_page;
            layer_cache.kv_cache_base =
                base.reshape({kernel_block_num, static_cast<int64_t>(layer_cache.seq_size_per_block), stride});
        } else if (base.dim() == 2 && num_kv_heads > 0 && head_dim > 0) {
            layer_cache.kv_cache_base = base.reshape({kernel_block_num,
                                                      2,
                                                      static_cast<int64_t>(num_kv_heads),
                                                      static_cast<int64_t>(layer_cache.seq_size_per_block),
                                                      static_cast<int64_t>(head_dim)});
        } else {
            layer_cache.kv_cache_base = base;
        }

        if (!scale.defined()) {
            return;
        }

        if (use_mla) {
            RTP_LLM_CHECK_WITH_INFO(scale.dim() > 0 && scale.size(0) == physical_block_num,
                                    "MLA scale physical block count must match KV cache base: scale=%ld base=%ld",
                                    scale.dim() > 0 ? scale.size(0) : -1,
                                    physical_block_num);
            RTP_LLM_CHECK_WITH_INFO(scale.is_contiguous(), "MLA scale/indexer cache base must be contiguous");
            const int64_t elements_per_kernel_page =
                kernel_block_num * static_cast<int64_t>(layer_cache.seq_size_per_block);
            RTP_LLM_CHECK_WITH_INFO(elements_per_kernel_page > 0 && scale.numel() % elements_per_kernel_page == 0,
                                    "MLA scale/indexer elements=%ld must be divisible by kernel pages=%ld",
                                    scale.numel(),
                                    elements_per_kernel_page);
            const int64_t scale_stride = scale.numel() / elements_per_kernel_page;
            layer_cache.kv_scale_base =
                scale.reshape({kernel_block_num, static_cast<int64_t>(layer_cache.seq_size_per_block), scale_stride});
        } else {
            layer_cache.kv_scale_base = scale.reshape({kernel_block_num, scale.size(1) / kernel_blocks_per_kv_block});
        }
    }

    LayerKVCache getLayerCache(int idx) {
        LayerKVCache layer_cache;
        layer_cache.layer_id = idx;

        const auto layer = static_cast<size_t>(idx);
        if (idx < 0 || layer >= kv_cache_base_by_layer.size()) {
            throw std::runtime_error("Invalid layer index: " + std::to_string(idx));
        }
        if (!layer_to_group_ids.empty()) {
            if (layer >= layer_to_group_ids.size()) {
                throw std::runtime_error("Invalid layer index for KV cache groups: " + std::to_string(idx));
            }
            if (layer_to_group_ids[layer].empty()) {
                throw std::runtime_error("Layer " + std::to_string(idx) + " owns no KV cache group");
            }
            if (layer_to_group_ids[layer].size() > 1) {
                throw std::runtime_error("Layer " + std::to_string(idx)
                                         + " owns multiple KV cache groups; use get_layer_cache(layer, tag) "
                                           "or get_layer_caches");
            }
        }

        auto          base = kv_cache_base_by_layer[idx];
        torch::Tensor scale;
        if (!kv_scale_base_by_layer.empty()) {
            scale = kv_scale_base_by_layer[idx];
        }

        int group_slot = -1;
        if (!layer_to_group_ids.empty() && layer < layer_to_group_ids.size() && layer_to_group_ids[layer].size() == 1) {
            group_slot = layer_to_group_ids[layer].front();
        } else if (group_tags.size() == 1) {
            group_slot = 0;
        }
        if (group_slot >= 0 && static_cast<size_t>(group_slot) < group_tags.size()) {
            layer_cache.tag = group_tags[static_cast<size_t>(group_slot)];
        }

        const bool is_full = layer_attn_types.empty() ? true :
                                                        (layer < layer_attn_types.size()
                                                         && layer_attn_types[layer] == rtp_llm::CacheGroupType::FULL);

        if (!is_full) {
            // Linear/SSM attention layer: return the raw cache tensor unchanged.
            // Use the physical block size so the layer sees the full per-block storage.
            layer_cache.seq_size_per_block = groupSeqBlockSize(group_slot);
            layer_cache.kv_cache_base      = base;
            layer_cache.kv_scale_base      = scale;
        } else {
            setFullAttentionView(layer_cache, base, scale, group_slot);
        }
        return layer_cache;
    }

private:
    LayerKVCache getLayerCacheBySlot(int idx, int group_slot) {
        const auto layer = static_cast<size_t>(idx);
        if (idx < 0 || layer >= kv_cache_base_by_layer_group.size()) {
            throw std::runtime_error("Invalid layer index: " + std::to_string(idx));
        }
        if (group_slot < 0 || static_cast<size_t>(group_slot) >= kv_cache_base_by_layer_group[layer].size()) {
            throw std::runtime_error("Invalid KV cache topology slot: " + std::to_string(group_slot));
        }
        if (!layer_to_group_ids.empty()) {
            if (layer >= layer_to_group_ids.size()
                || std::find(layer_to_group_ids[layer].begin(), layer_to_group_ids[layer].end(), group_slot)
                       == layer_to_group_ids[layer].end()) {
                throw std::runtime_error("Layer " + std::to_string(idx) + " does not own KV cache topology slot "
                                         + std::to_string(group_slot));
            }
        }

        auto base = kv_cache_base_by_layer_group[layer][static_cast<size_t>(group_slot)];
        if (!base.defined()) {
            throw std::runtime_error("Missing KV cache tensor for layer " + std::to_string(idx) + ", topology slot "
                                     + std::to_string(group_slot));
        }

        LayerKVCache layer_cache;
        layer_cache.layer_id = idx;
        if (static_cast<size_t>(group_slot) < group_tags.size()) {
            layer_cache.tag = group_tags[static_cast<size_t>(group_slot)];
        }
        const bool is_full_group = static_cast<size_t>(group_slot) < group_types.size()
                                   && group_types[static_cast<size_t>(group_slot)] == rtp_llm::CacheGroupType::FULL;
        torch::Tensor scale;
        if (!kv_scale_base_by_layer_group.empty() && layer < kv_scale_base_by_layer_group.size()
            && static_cast<size_t>(group_slot) < kv_scale_base_by_layer_group[layer].size()) {
            scale = kv_scale_base_by_layer_group[layer][static_cast<size_t>(group_slot)];
        }

        if (!is_full_group) {
            layer_cache.seq_size_per_block = groupSeqBlockSize(group_slot);
            layer_cache.kv_cache_base      = base;
            layer_cache.kv_scale_base      = scale;
            return layer_cache;
        }

        setFullAttentionView(layer_cache, base, scale, group_slot);
        return layer_cache;
    }

public:
    LayerKVCache getLayerCache(int idx, const std::string& tag) {
        const auto layer = static_cast<size_t>(idx);
        if (idx < 0 || layer >= layer_tag_to_group_id.size()) {
            throw std::runtime_error("Invalid layer index for cache tag lookup: " + std::to_string(idx));
        }
        const auto it = layer_tag_to_group_id[layer].find(tag);
        if (it == layer_tag_to_group_id[layer].end() || it->second < 0) {
            throw std::runtime_error("Layer " + std::to_string(idx) + " does not own KV cache tag " + tag);
        }
        const int group_slot = it->second;
        if (group_slot < 0 || static_cast<size_t>(group_slot) >= group_tags.size()) {
            throw std::runtime_error("KV cache tag " + tag + " maps to invalid topology slot "
                                     + std::to_string(group_slot));
        }
        return getLayerCacheBySlot(idx, group_slot);
    }

    std::vector<LayerKVCache> getLayerCaches(int idx) {
        if (layer_to_group_ids.empty() || group_tags.empty()) {
            return {getLayerCache(idx)};
        }
        const auto layer = static_cast<size_t>(idx);
        if (idx < 0 || layer >= layer_to_group_ids.size()) {
            throw std::runtime_error("Invalid layer index: " + std::to_string(idx));
        }

        std::vector<LayerKVCache> layer_caches;
        for (int group_slot : layer_to_group_ids[layer]) {
            layer_caches.push_back(getLayerCacheBySlot(idx, group_slot));
        }
        return layer_caches;
    }
};

struct PyModelInitResources {
    std::optional<KVCache> kv_cache;
};

struct PyCacheStoreInputs {
    size_t                                           context_batch_size = 0;
    size_t                                           decoder_batch_size = 0;
    torch::Tensor                                    request_id;
    torch::Tensor                                    request_pd_separation;
    std::map<std::string, rtp_llm::CacheGroupType>   kv_cache_group_types;
    std::map<std::string, rtp_llm::CacheGroupPolicy> kv_cache_group_policies;
    std::map<std::string, size_t>                    tokens_per_block_by_tag;
    std::map<std::string, size_t>                    kv_block_stride_bytes_by_tag;
    std::map<std::string, size_t>                    kv_scale_stride_bytes_by_tag;
    std::vector<std::string>                         cache_keys;  // [context_batch_size]
    size_t                                           tokens_per_block = 0;
    // Physical KV-manager block strides, supplied by CacheConfig rather than inferred from tensor views.
    size_t kv_block_stride_bytes     = 0;
    size_t kv_scale_stride_bytes     = 0;
    bool   pd_separation             = false;
    size_t model_id                  = 0;
    bool   decode_entrance           = false;
    bool   warmup                    = false;
    bool   use_hybrid_kv_cache_store = false;
    bool   use_opaque_kv_cache_store = false;
    bool   mla_kvcache               = false;

    // Cache store reference (C++ only; passes through Python without inspection)
    std::shared_ptr<rtp_llm::CacheStore> cache_store;
    rtp_llm::CacheStoreAsyncWriter*      cache_store_async_writer = nullptr;

    // CP-page-RR sharding context. (1, 0) = no sharding.
    int cp_size = 1;
    int cp_rank = 0;
};

struct PyPrefillCudaGaphCopyParams {
    // for embedding model cuda graph capture, the attenton batch size is padded to max_batch_size,
    // so we can't get the real batch size for `copy kernel` using `input_lengths.size(0)`(which is max_batch_size).
    torch::Tensor cuda_graph_prefill_batch_size = torch::empty(0);
    int           max_seq_len                   = 0;
    int           max_batch_size                = 0;
};

struct PyContextParallelParams {
    torch::Tensor prefill_cp_padding_lengths;
    torch::Tensor prefill_cp_chunk_lengths;
    torch::Tensor prefill_shuffle_indices;
    torch::Tensor prefill_qkv_restore_indice;
    torch::Tensor prefill_qkv_padding_mask;
    torch::Tensor prefill_actual_input_lengths_cpu;
};

// Naming convention: the host (pinned CPU) tensor uses the bare name; its device (CUDA)
// counterpart carries a _device suffix.
struct PyAttentionInputs {
    bool          is_prefill{false};
    bool          is_target_verify{false};
    torch::Tensor prefix_lengths;
    torch::Tensor sequence_lengths;
    torch::Tensor input_lengths;
    // Group-local kernel-granularity block IDs for attention compute.
    // Shape: [batch, max_kernel_blocks].
    torch::Tensor kv_cache_kernel_block_id;
    torch::Tensor kv_cache_kernel_block_id_device;
    // Group-local physical block IDs dedicated for cache store.
    // Shape: [batch, max_blocks].
    torch::Tensor    kv_cache_block_id;
    torch::Tensor    kv_cache_block_id_device;
    caffe2::TypeMeta dtype;
    // Cumulative sequence lengths for attention kernels (e.g. FusedRopeKVCacheDecodeOp).
    // cu_seqlens_device lives on CUDA device; cu_seqlens is its pinned-memory CPU mirror
    // used for CUDA graph replay (write host -> async copy to device, avoiding GPU-side fills).
    torch::Tensor cu_seqlens;
    torch::Tensor cu_seqlens_device;
    torch::Tensor cu_kv_seqlens_device;  // device only (no host mirror needed)
    torch::Tensor decode_cu_seqlens;
    int           context_total_kv_length = 0;
    int           total_tokens            = 0;
    torch::Tensor padding_offset;
    torch::Tensor combo_position_ids;

    // for write cache store
    std::optional<PyCacheStoreInputs> cache_store_inputs;

    std::optional<PyPrefillCudaGaphCopyParams> prefill_cuda_graph_copy_params;
    bool                                       is_s_padded = false;
    // Device-side mirrors of host tensors, managed by C++ for fused D2D copy in CUDA graph.
    torch::Tensor prefix_lengths_device;
    torch::Tensor sequence_lengths_plus_1_device;
    torch::Tensor input_lengths_device;
    torch::Tensor decode_cu_seqlens_device;

    // CUDA Graph mode flags
    bool is_cuda_graph = false;  // True when running in CUDA graph mode (capture or replay)

    std::optional<PyContextParallelParams> context_parallel_info;

    // Headwise attention config (Python dict or None).
    py::object headwise_config{py::none()};
};

struct BertEmbeddingInputs {
    torch::Tensor combo_position_ids;
    torch::Tensor position_encoding;
    torch::Tensor combo_tokens_type_ids;
    torch::Tensor token_type_embedding;
    float         input_embedding_scalar{1.0};
};

struct PyEmbeddingInputs {
    torch::Tensor combo_tokens_type_ids;
    torch::Tensor text_tokens_mask;
};

struct PyMultimodalInputs {
    std::vector<torch::Tensor> multimodal_features;
    torch::Tensor              mm_features_locs;
    std::vector<torch::Tensor> mm_extra_input;
};

using AttentionInputsByTag = std::map<std::string, PyAttentionInputs>;

struct PyModelInputs {
    torch::Tensor      input_ids;
    torch::Tensor      input_hiddens;
    torch::Tensor      combo_position_ids;
    PyEmbeddingInputs  embedding_inputs;
    PyMultimodalInputs multimodal_inputs;
    // C++ common/single-group fast path. Python sees this field through a
    // property which returns either this object or attention_inputs_by_tag.
    PyAttentionInputs    attention_inputs;
    AttentionInputsByTag attention_inputs_by_tag;
    BertEmbeddingInputs  bert_embedding_inputs;

    bool hasAttentionInputsByTag() const {
        return !attention_inputs_by_tag.empty();
    }
};

struct PyModelOutputs {
    torch::Tensor          hidden_states;
    rtp_llm::ParamsBasePtr params_ptr{nullptr};

    PyModelOutputs() = default;

    // Constructor with default hidden_states
    PyModelOutputs(torch::Tensor hidden_states): hidden_states(std::move(hidden_states)), params_ptr(nullptr) {}

    PyModelOutputs(torch::Tensor hidden_states, std::shared_ptr<rtp_llm::ParamsBase> params_ptr):
        hidden_states(std::move(hidden_states)), params_ptr(std::move(params_ptr)) {}
};

void registerPyOpDefs(pybind11::module& m);

}  // namespace torch_ext
