#pragma once

#include <map>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/CacheBlockTablePacking.h"
#include "rtp_llm/cpp/models/PackedBlockTableStorage.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

// Validates the caller's four flat backings against the plan and adopts their
// references. No copy: reference counting already guarantees the storage stays
// alive and address-stable through capture and replay.
void adoptPackedBlockTables(const CacheBlockTablePackingPlan& plan,
                            const torch::Tensor&              pool_host,
                            const torch::Tensor&              pool_device,
                            const torch::Tensor&              kernel_host,
                            const torch::Tensor&              kernel_device,
                            PackedBlockTableStorage&          storage);

// Pure function: given a base attn_inputs carrying only shared fields, produce one
// PyAttentionInputs per cache group, keyed by tag. Shared fields are reference-shared;
// the four table fields are 2-D alias views of that group's packed segment.
std::map<std::string, torch_ext::PyAttentionInputs>
bindCacheGroupAttentionInputs(const torch_ext::PyAttentionInputs& base,
                              const CacheBlockTablePackingPlan&   plan,
                              const PackedBlockTableStorage&      storage,
                              std::vector<CacheGroupBinding>&     bindings);

// Normalizes each group's kernel table tail to NULL_BLOCK_IDX. Shared by full
// prepare and value-only update so first-forward correctness does not depend on
// the gatherer having already written NULL tails.
// Returns the number of device fill launches issued (0 or 1).
size_t normalizeKernelTailFill(const std::map<std::string, torch_ext::PyAttentionInputs>& group_inputs,
                               const std::vector<CacheGroupBinding>&                      bindings,
                               const torch::Tensor&                                       regions_host,
                               const torch::Tensor&                                       regions_device,
                               size_t                                                     region_capacity);

// Value-only refresh: copies new values into the storage we already hold. Never
// reallocates and never rebuilds views, so CUDA Graph capture stays valid. New
// per-group valid lengths are copied in place into each per-group PyAttentionInputs
// (keyed by binding tag).
void refreshPackedBlockTableValues(const torch::Tensor&                                 kernel_host,
                                   const torch::Tensor&                                 kernel_device,
                                   const std::vector<torch::Tensor>&                    kernel_valid_lengths,
                                   PackedBlockTableStorage&                             storage,
                                   const std::vector<CacheGroupBinding>&                bindings,
                                   std::map<std::string, torch_ext::PyAttentionInputs>& group_inputs);

}  // namespace rtp_llm
