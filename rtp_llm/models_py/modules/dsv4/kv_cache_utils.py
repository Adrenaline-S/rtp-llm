"""DSV4 KV-cache tags and lookup utilities.

This module owns two things:

1. The canonical DSV4 cache-group **tags**. The framework KV cache API is
   tag-driven: a cache group is identified by a string tag ("swa_kv",
   "csa_kv", ...) which is the same string used by ``CacheConfig`` /
   resolved ``CacheConfig`` groups on the C++ side, by ``KVCache.get_layer_cache(layer, tag)``
   / ``KVCache.get_seq_size_per_block(tag)``, and as the key of
   ``PyModelInputs.attention_inputs`` when the model owns several groups.
   These constants replace the old int ``attn_type`` ids that mirrored the
   deleted C++ ``KVCacheRegionName`` enum.

2. Generic ``tag -> block_table`` helpers shared between prefill and decode.
   They translate the dense ordinal-indexed alias views only for DSV4 backend
   APIs that still require tag-keyed dictionaries; common request metadata
   remains in the single ``PyModelInputs.attention_inputs`` object.

Path-specific forward helpers live in :mod:`prefill.forward` /
:mod:`decode.forward`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Canonical cache-group tags. These are the *consumer* side of the tags that
# ``rtp_llm/models/dsv4_kv_cache.py`` (``CSA_KV_TAG`` ... ``SWA_KV_TAG``) hands
# to ``ModelConfig.kv_cache_spec_descs`` and that CacheConfig then
# publishes as ``KVCache.group_tags``. They are duplicated rather than imported
# on purpose: ``rtp_llm.models.dsv4_kv_cache`` pulls in ``rtp_llm.ops`` (the
# compiled .so), while this module must stay importable from kernel-level code
# and unit tests that only need torch. Keep both lists in sync.
# ---------------------------------------------------------------------------
SWA_KV = "swa_kv"
CSA_KV = "csa_kv"
HCA_KV = "hca_kv"
INDEXER_KV = "indexer_kv"
INDEXER_STATE = "indexer_state"
CSA_STATE = "csa_state"
HCA_STATE = "hca_state"

# Single-group models expose exactly one group under this tag.
DEFAULT_TAG = "default"

# Paged (FULL) KV pools — a block-table row covers ``kernel_seq_size_per_block``
# raw tokens.
DSV4_KERNEL_ROW_TAGS: Tuple[str, ...] = (CSA_KV, HCA_KV, INDEXER_KV)
# Fixed / ring pools — a block-table row covers ``seq_size_per_block`` raw
# tokens.
DSV4_PHYSICAL_ROW_TAGS: Tuple[str, ...] = (
    SWA_KV,
    CSA_STATE,
    HCA_STATE,
    INDEXER_STATE,
)
DSV4_TAGS: Tuple[str, ...] = DSV4_KERNEL_ROW_TAGS + DSV4_PHYSICAL_ROW_TAGS


def kv_tag_for_compress_ratio(ratio: int) -> Optional[str]:
    """Compressed KV pool tag for a layer's compression ratio (``None`` = SWA-only)."""
    if int(ratio) == 4:
        return CSA_KV
    if int(ratio) == 128:
        return HCA_KV
    return None


def select_cache_group_attn_inputs(
    model_inputs: Any, cache_tags: Sequence[str]
) -> Dict[str, Any]:
    """Select this backend's per-tag attention inputs from the model inputs."""
    if model_inputs is None:
        return {}
    group_attn_inputs = getattr(model_inputs, "cache_group_attn_inputs", None)
    if not group_attn_inputs:
        return {}
    result: Dict[str, Any] = {}
    for tag in cache_tags:
        if tag not in group_attn_inputs:
            raise RuntimeError(
                f"cache tag {tag!r} has no attention inputs; "
                f"available tags: {sorted(group_attn_inputs)}"
            )
        result[tag] = group_attn_inputs[tag]
    return result


def primary_attention_inputs(
    attention_inputs: Any,
    kv_cache: Optional[Any] = None,
) -> Optional[Any]:
    """Return the per-forward inputs carrying the group-invariant fields.

    The framework publishes exactly one ``PyAttentionInputs`` carrying
    ``cu_seqlens`` / ``input_lengths`` / ``sequence_lengths`` /
    ``prefix_lengths`` / ``cache_store_inputs`` / ``context_parallel_info``.
    Only block tables are group-local alias views — read those through
    :func:`build_block_tables` / :func:`build_block_tables_batched`.
    """
    if attention_inputs is None:
        return None
    if hasattr(attention_inputs, "attention_inputs"):
        attention_inputs = attention_inputs.attention_inputs
        if attention_inputs is None:
            return None
    return attention_inputs


def _block_table_for_tag(tagged_inputs: Any) -> Optional[torch.Tensor]:
    if tagged_inputs is None:
        return None
    block_table = getattr(tagged_inputs, "kv_cache_kernel_block_id_device", None)
    if block_table is None or block_table.numel() == 0:
        return None
    return block_table


def _build_block_tables(
    model_inputs: Any,
    group_bindings: Sequence[str],
    batch_slice: Optional[slice],
    keep_tags: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    by_tag = select_cache_group_attn_inputs(model_inputs, group_bindings)
    if not by_tag:
        return None
    wanted = None if keep_tags is None else set(keep_tags)
    block_tables: Dict[str, torch.Tensor] = {}
    for tag, tagged_inputs in by_tag.items():
        if wanted is not None and tag not in wanted:
            continue
        block_table = _block_table_for_tag(tagged_inputs)
        if block_table is None:
            continue
        block_tables[tag] = (
            block_table if batch_slice is None else block_table[batch_slice]
        )
    return block_tables or None


def build_block_tables(
    model_inputs: Any,
    group_bindings: Sequence[str],
    batch_offset: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build the per-tag block-table dict for one prefill request.

    The framework hands each cache group a tag-keyed zero-copy attention-inputs
    entry in ``PyModelInputs.cache_group_attn_inputs``. This helper translates the
    selected kernel tables into the tag-keyed dictionary required by DSV4 kernels.

    The ``batch_offset`` arg slices out a single-request row
    ``[batch_offset : batch_offset + 1]`` so the returned block table is
    per-request, matching how ``DeepSeekV4Model.forward`` unrolls batched
    prefill into one-request-at-a-time layer calls.

    Returns ``None`` when no block tables are available (warmup / paged-KV
    disabled / missing framework state).
    """
    return _build_block_tables(
        model_inputs,
        group_bindings,
        slice(batch_offset, batch_offset + 1),
    )


def build_block_tables_batched(
    model_inputs: Any,
    group_bindings: Sequence[str],
) -> Optional[Dict[str, torch.Tensor]]:
    """Build the per-tag block-table dict for an entire prefill batch.

    Same semantics as :func:`build_block_tables` but returns the full
    ``[B, max_blocks]`` block table per tag (no ``batch_offset`` slice).
    Used by the batched ``forward_prefill`` main path so a single ``v4()`` call
    can cover the whole batch.

    Returns ``None`` when no block tables are available (warmup / paged-KV
    disabled / missing framework state).
    """
    return _build_block_tables(model_inputs, group_bindings, None)


def build_block_tables_for_tags(
    model_inputs: Any,
    group_bindings: Sequence[str],
    tags: Iterable[str],
) -> Optional[Dict[str, torch.Tensor]]:
    """Batched block tables restricted to ``tags`` (decode's paged-pool set)."""
    return _build_block_tables(model_inputs, group_bindings, None, keep_tags=tags)
