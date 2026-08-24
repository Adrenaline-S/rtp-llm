#include "rtp_llm/cpp/models/CacheGroupAttentionInputs.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/BlockExpression.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_utils.h"
#endif

namespace rtp_llm {

namespace {

void checkBacking(const torch::Tensor& backing, size_t required_numel, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(backing.defined(), "%s packed backing must be defined", name);
    RTP_LLM_CHECK_WITH_INFO(backing.dim() == 1, "%s packed backing must be flat", name);
    RTP_LLM_CHECK_WITH_INFO(
        backing.scalar_type() == torch::kInt32, "%s packed backing must use int32 block identities", name);
    RTP_LLM_CHECK_WITH_INFO(backing.is_contiguous(), "%s packed backing must be contiguous", name);
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(backing.numel()) >= required_numel,
                            "%s packed backing has %ld elements but plan needs %zu",
                            name,
                            backing.numel(),
                            required_numel);
}

}  // namespace

void adoptPackedBlockTables(const CacheBlockTablePackingPlan& plan,
                            const torch::Tensor&              pool_host,
                            const torch::Tensor&              pool_device,
                            const torch::Tensor&              kernel_host,
                            const torch::Tensor&              kernel_device,
                            PackedBlockTableStorage&          storage) {
    checkBacking(pool_host, plan.poolNumel(), "pool host");
    checkBacking(pool_device, plan.poolNumel(), "pool device");
    checkBacking(kernel_host, plan.kernelNumel(), "kernel host");
    checkBacking(kernel_device, plan.kernelNumel(), "kernel device");

    // Adopt references, not copies: reference counting already keeps this storage
    // alive and address-stable for the capture lifetime. The caller must not write
    // its own backing after prepare; structure changes are caught by the signature.
    storage.pool_host     = pool_host;
    storage.pool_device   = pool_device;
    storage.kernel_host   = kernel_host;
    storage.kernel_device = kernel_device;
}

std::map<std::string, torch_ext::PyAttentionInputs>
bindCacheGroupAttentionInputs(const torch_ext::PyAttentionInputs& base,
                              const CacheBlockTablePackingPlan&   plan,
                              const PackedBlockTableStorage&      storage,
                              std::vector<CacheGroupBinding>&     bindings) {
    std::map<std::string, torch_ext::PyAttentionInputs> group_inputs;
    bindings.clear();
    if (plan.groupCount() == 0) {
        return group_inputs;
    }
    RTP_LLM_CHECK_WITH_INFO(storage.defined(), "packed block table storage must be adopted first");
    bindings.reserve(plan.groupCount());

    for (uint32_t ordinal = 0; ordinal < plan.groupCount(); ++ordinal) {
        const auto& region = plan.group(ordinal);
        RTP_LLM_CHECK_WITH_INFO(region.execution_ordinal == ordinal,
                                "packing plan group %u carries ordinal %u; ordinals must be dense",
                                ordinal,
                                region.execution_ordinal);
        RTP_LLM_CHECK_WITH_INFO(!region.tag.empty(), "packing plan group %u has an empty tag", ordinal);

        // Shared fields stay reference-shared; only the four table fields are rebound.
        torch_ext::PyAttentionInputs group    = base;
        group.kv_cache_block_id               = plan.poolView(storage.pool_host, ordinal);
        group.kv_cache_block_id_device        = plan.poolView(storage.pool_device, ordinal);
        group.kv_cache_kernel_block_id        = plan.kernelView(storage.kernel_host, ordinal);
        group.kv_cache_kernel_block_id_device = plan.kernelView(storage.kernel_device, ordinal);

        const auto inserted = group_inputs.emplace(region.tag, std::move(group));
        RTP_LLM_CHECK_WITH_INFO(inserted.second, "packing plan has duplicate tag=%s", region.tag.c_str());

        CacheGroupBinding binding;
        binding.execution_ordinal = ordinal;
        binding.tag               = region.tag;
        bindings.push_back(std::move(binding));
    }

    return group_inputs;
}

size_t normalizeKernelTailFill(const std::map<std::string, torch_ext::PyAttentionInputs>& group_inputs,
                               const std::vector<CacheGroupBinding>&                      bindings,
                               const torch::Tensor&                                       regions_host,
                               const torch::Tensor&                                       regions_device,
                               size_t                                                     region_capacity) {
#if USING_CUDA
    auto*  regions      = regions_host.defined() ?
                              reinterpret_cast<CudaGraphPrepareFillRegion*>(regions_host.data_ptr<uint8_t>()) :
                              nullptr;
    size_t region_count = 0;
#endif

    for (const auto& binding : bindings) {
        const auto it = group_inputs.find(binding.tag);
        RTP_LLM_CHECK_WITH_INFO(
            it != group_inputs.end(), "tail fill has no attention inputs for tag=%s", binding.tag.c_str());
        const auto& kernel_valid_lengths = it->second.kernel_valid_lengths;
        if (!kernel_valid_lengths.defined()) {
            continue;
        }

        const auto& table        = it->second.kv_cache_kernel_block_id;
        const auto& device_table = it->second.kv_cache_kernel_block_id_device;
        const auto* lengths      = kernel_valid_lengths.data_ptr<int32_t>();
        auto*       host_base    = table.data_ptr<int32_t>();
        const auto  row_width    = table.size(1);
#if USING_CUDA
        auto* device_base = device_table.data_ptr<int32_t>();
#endif

        for (int64_t row = 0; row < table.size(0); ++row) {
            const int64_t valid = lengths[row];
            RTP_LLM_CHECK_WITH_INFO(valid >= 0 && valid <= row_width,
                                    "kernel valid length %ld is outside [0, %ld] for tag=%s row %ld",
                                    valid,
                                    row_width,
                                    binding.tag.c_str(),
                                    row);
            if (valid == row_width) {
                continue;
            }
            std::fill(host_base + row * row_width + valid, host_base + (row + 1) * row_width, NULL_BLOCK_IDX);
#if USING_CUDA
            // Device descriptor recording only runs when the caller supplied a
            // descriptor buffer. Host-only callers pass undefined region tensors
            // (capacity 0) to normalize just the host replica.
            if (regions != nullptr) {
                RTP_LLM_CHECK_WITH_INFO(region_count < region_capacity,
                                        "kernel tail-fill region count exceeds prepared capacity %zu",
                                        region_capacity);
                auto& region     = regions[region_count++];
                region.ptr       = device_base + row * row_width + valid;
                region.value_ptr = nullptr;
                region.count     = row_width - valid;
                region.value     = NULL_BLOCK_IDX;
            }
#else
            device_table[row].slice(0, valid).fill_(NULL_BLOCK_IDX);
#endif
        }
    }

#if USING_CUDA
    if (region_count == 0) {
        return 0;
    }
    const auto bytes       = region_count * sizeof(CudaGraphPrepareFillRegion);
    const auto stream      = cuda_graph::graphGetCurrentStream().stream();
    const auto copy_result = cudaMemcpyAsync(
        regions_device.data_ptr<uint8_t>(), regions_host.data_ptr<uint8_t>(), bytes, cudaMemcpyHostToDevice, stream);
    RTP_LLM_CHECK_WITH_INFO(
        copy_result == cudaSuccess, "kernel tail-fill descriptor copy failed: %s", cudaGetErrorString(copy_result));
    invokeCudaGraphPrepareFillRegions(
        reinterpret_cast<const CudaGraphPrepareFillRegion*>(regions_device.data_ptr<uint8_t>()),
        static_cast<int32_t>(region_count),
        stream);
    return 1;
#else
    return 0;
#endif
}

void refreshPackedBlockTableValues(const torch::Tensor&                                 kernel_host,
                                   const torch::Tensor&                                 kernel_device,
                                   const std::vector<torch::Tensor>&                    kernel_valid_lengths,
                                   PackedBlockTableStorage&                             storage,
                                   const std::vector<CacheGroupBinding>&                bindings,
                                   std::map<std::string, torch_ext::PyAttentionInputs>& group_inputs) {
    RTP_LLM_CHECK_WITH_INFO(storage.defined(), "value-only update requires a fully prepared packed snapshot");
    RTP_LLM_CHECK_WITH_INFO(storage.kernel_host.sizes() == kernel_host.sizes(),
                            "value-only update got a kernel backing of a different shape; full prepare required");
    RTP_LLM_CHECK_WITH_INFO(storage.kernel_device.sizes() == kernel_device.sizes(),
                            "value-only update got a device kernel backing of a different shape");
    RTP_LLM_CHECK_WITH_INFO(kernel_valid_lengths.size() == bindings.size(),
                            "value-only update got %zu valid-length tensors for %zu groups",
                            kernel_valid_lengths.size(),
                            bindings.size());

    storage.kernel_host.copy_(kernel_host, /*non_blocking=*/false);
    storage.kernel_device.copy_(kernel_device, /*non_blocking=*/true);

    for (size_t idx = 0; idx < bindings.size(); ++idx) {
        const auto& source = kernel_valid_lengths[idx];
        RTP_LLM_CHECK_WITH_INFO(source.defined(), "valid lengths for group %zu are undefined", idx);
        const auto it = group_inputs.find(bindings[idx].tag);
        RTP_LLM_CHECK_WITH_INFO(it != group_inputs.end(),
                                "value-only update has no attention inputs for tag=%s",
                                bindings[idx].tag.c_str());
        auto& target = it->second.kernel_valid_lengths;
        if (!target.defined() || target.sizes() != source.sizes()) {
            target = torch::empty(source.sizes(), source.options());
        }
        target.copy_(source, /*non_blocking=*/false);
    }
}

}  // namespace rtp_llm
