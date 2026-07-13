from typing import Any

from rtp_llm.ops.compute_ops import PyAttentionInputs


def get_attn_inputs_list(inputs: Any) -> list[PyAttentionInputs]:
    attn_inputs_list = getattr(inputs, "attn_inputs_list", None)
    if attn_inputs_list is not None and len(attn_inputs_list) > 0:
        return attn_inputs_list

    raise RuntimeError("PyModelInputs.attn_inputs_list must not be empty")


def select_block_map_for_layer(
    attention_inputs: PyAttentionInputs, layer_idx: int
) -> int:
    if attention_inputs.kv_cache_kernel_block_id_device_by_group is None:
        return

    gid = 0
    if attention_inputs.kv_cache_layer_to_group is not None:
        gid = int(attention_inputs.kv_cache_layer_to_group[layer_idx].item())

    if attention_inputs.kv_cache_kernel_block_id_device_by_group is not None and len(
        attention_inputs.kv_cache_kernel_block_id_device_by_group
    ):
        attention_inputs.kv_cache_kernel_block_id_device = (
            attention_inputs.kv_cache_kernel_block_id_device_by_group[gid]
        )

    if attention_inputs.kv_cache_kernel_block_id_by_group is not None and len(
        attention_inputs.kv_cache_kernel_block_id_by_group
    ):
        attention_inputs.kv_cache_kernel_block_id = (
            attention_inputs.kv_cache_kernel_block_id_by_group[gid]
        )
    return gid


def _layer_group_ids(kv_cache: Any, local_layer_idx: int) -> list[int]:
    if kv_cache is None or not getattr(kv_cache, "layer_to_group_ids", None):
        return []
    if local_layer_idx < 0 or local_layer_idx >= len(kv_cache.layer_to_group_ids):
        raise RuntimeError(
            f"local layer {local_layer_idx} is out of layer_to_group_ids range "
            f"{len(kv_cache.layer_to_group_ids)}"
        )
    return [int(gid) for gid in kv_cache.layer_to_group_ids[local_layer_idx]]


def _layer_group_id(
    kv_cache: Any, local_layer_idx: int, attention_inputs: Any | None = None
) -> int:
    group_ids = _layer_group_ids(kv_cache, local_layer_idx)
    if not group_ids:
        return 0
    if len(group_ids) == 1:
        return group_ids[0]
    if attention_inputs is not None:
        group_id = int(getattr(attention_inputs, "cache_group_id", -1))
        if group_id in group_ids:
            return group_id
    raise RuntimeError(
        f"local layer {local_layer_idx} maps to multiple attention input groups "
        f"{group_ids}; Python model_desc supports one group per layer"
    )


def select_attention_inputs_for_layer(
    inputs: Any, kv_cache: Any, local_layer_idx: int
) -> PyAttentionInputs:
    """Return the group-specific PyAttentionInputs used by a model-local layer."""
    attn_inputs_list = get_attn_inputs_list(inputs)
    group_id = _layer_group_id(kv_cache, local_layer_idx)
    if group_id < 0 or group_id >= len(attn_inputs_list):
        raise RuntimeError(
            f"local layer {local_layer_idx} maps to invalid attention_inputs group "
            f"{group_id}; available groups={len(attn_inputs_list)}"
        )
    return attn_inputs_list[group_id]


def select_fmha_impl_for_attention_inputs(
    fmha_impl: Any, attention_inputs: PyAttentionInputs
) -> Any:
    if not isinstance(fmha_impl, (list, tuple)):
        return fmha_impl

    group_id = getattr(attention_inputs, "cache_group_id", 0)
    group_id = 0 if group_id is None or group_id < 0 else int(group_id)
    if group_id >= len(fmha_impl):
        raise RuntimeError(
            f"attention_inputs group {group_id} is out of fmha_impl range "
            f"{len(fmha_impl)}"
        )
    return fmha_impl[group_id]
