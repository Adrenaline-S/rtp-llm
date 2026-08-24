#pragma once

#include <cstdint>
#include <string>

#include <torch/extension.h>

namespace rtp_llm {

// References to the caller's four flat packed backings, adopted at full prepare.
// Tensor reference counting keeps this storage alive and address-stable for the
// whole CUDA Graph capture lifetime, so no copy is needed here; value-only update
// writes into these same tensors via copy_.
struct PackedBlockTableStorage {
    torch::Tensor pool_host;
    torch::Tensor pool_device;
    torch::Tensor kernel_host;
    torch::Tensor kernel_device;

    bool defined() const {
        return pool_host.defined() && pool_device.defined() && kernel_host.defined() && kernel_device.defined();
    }
};

// Per-group bookkeeping that never crosses the pybind boundary. execution_ordinal
// lives here because it is a packed-layout detail, not a business identity. Per-group
// valid lengths live on each per-group PyAttentionInputs, not here.
struct CacheGroupBinding {
    uint32_t    execution_ordinal = 0;
    std::string tag;
};

}  // namespace rtp_llm
