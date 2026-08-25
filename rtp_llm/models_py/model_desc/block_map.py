from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, TypeVar

from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, PyModelInputs

T = TypeVar("T")


class LayeredKVCache(Protocol):
    def get_layer_cache_groups(
        self, local_layer_idx: int
    ) -> Sequence[LayerKVCache]: ...


def get_attention_inputs_value(inputs: PyModelInputs) -> PyAttentionInputs:
    value = inputs.attention_inputs
    if isinstance(value, PyAttentionInputs):
        return value
    raise RuntimeError("PyModelInputs.attention_inputs must be PyAttentionInputs")


def get_primary_attention_inputs(
    inputs: PyModelInputs, kv_cache: LayeredKVCache | None = None
) -> PyAttentionInputs:
    """Return the common/single fast-path value without interpreting tag names."""
    return get_attention_inputs_value(inputs)


def get_layer_tags(kv_cache: LayeredKVCache | None, local_layer_idx: int) -> list[str]:
    if kv_cache is None:
        return []
    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    tags = [str(cache.tag) for cache in layer_caches]
    if not tags or any(not tag for tag in tags):
        raise RuntimeError(f"local layer {local_layer_idx} has no cache group tag")
    if len(tags) != len(set(tags)):
        raise RuntimeError(
            f"local layer {local_layer_idx} has duplicate KV cache tag; tags={tags}"
        )
    return tags


def get_layer_cache_for_tag(
    kv_cache: LayeredKVCache | None, local_layer_idx: int, tag: str
) -> LayerKVCache:
    if kv_cache is None:
        raise RuntimeError(
            f"KV cache tag {tag!r} is missing for local layer {local_layer_idx}: "
            "KV cache is not initialized"
        )
    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    matches = [cache for cache in layer_caches if str(cache.tag) == tag]
    if len(matches) > 1:
        raise RuntimeError(
            f"local layer {local_layer_idx} has duplicate KV cache tag {tag!r}"
        )
    if not matches:
        available_tags = [str(cache.tag) for cache in layer_caches]
        raise RuntimeError(
            f"KV cache tag {tag!r} is missing for local layer {local_layer_idx}; "
            f"available tags={available_tags}"
        )
    return matches[0]


def get_layer_caches_for_tags(
    kv_cache: LayeredKVCache | None,
    local_layer_idx: int,
    tags: Sequence[str],
) -> dict[str, LayerKVCache]:
    required_tags = list(tags)
    if (
        not required_tags
        or any(not tag for tag in required_tags)
        or len(required_tags) != len(set(required_tags))
    ):
        raise RuntimeError(
            f"required KV cache tags must be unique and non-empty: {tags}"
        )
    if kv_cache is None:
        raise RuntimeError(
            f"KV cache is not initialized for local layer {local_layer_idx}; "
            f"required tags={required_tags}"
        )

    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    by_tag: dict[str, LayerKVCache] = {}
    for cache in layer_caches:
        cache_tag = str(cache.tag)
        if not cache_tag:
            raise RuntimeError(f"local layer {local_layer_idx} has no cache group tag")
        if cache_tag in by_tag:
            raise RuntimeError(
                f"local layer {local_layer_idx} has duplicate KV cache tag {cache_tag!r}"
            )
        by_tag[cache_tag] = cache

    if set(by_tag) != set(required_tags):
        raise RuntimeError(
            f"local layer {local_layer_idx} requires exactly KV cache tags "
            f"{required_tags}; available tags={list(by_tag)}"
        )
    return {tag: by_tag[tag] for tag in required_tags}


def get_group_tags_for_layers(
    kv_cache: LayeredKVCache | None, local_layer_indices: Iterable[int]
) -> list[str]:
    """Return topology tags for model-selected layers, preserving topology order."""
    tags: list[str] = []
    seen: set[str] = set()
    for local_layer_idx in local_layer_indices:
        for tag in get_layer_tags(kv_cache, local_layer_idx):
            if tag not in seen:
                tags.append(tag)
                seen.add(tag)
    return tags


def select_cache_group_attn_inputs_for_layer(
    inputs: PyModelInputs,
    kv_cache: LayeredKVCache | None,
    local_layer_idx: int,
) -> PyAttentionInputs | list[PyAttentionInputs]:
    """Return this layer's per-group attention inputs, keyed by cache tag."""
    if kv_cache is None:
        raise RuntimeError(
            f"KV cache is not initialized for local layer {local_layer_idx}"
        )
    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    if not layer_caches:
        raise RuntimeError(f"local layer {local_layer_idx} has no KV cache group")
    group_attn_inputs = inputs.cache_group_attn_inputs
    selected = []
    for layer_cache in layer_caches:
        tag = str(layer_cache.tag)
        if tag not in group_attn_inputs:
            raise RuntimeError(
                f"local layer {local_layer_idx} cache tag {tag!r} has no attention "
                f"inputs; available tags: {sorted(group_attn_inputs)}"
            )
        selected.append(group_attn_inputs[tag])
    return selected[0] if len(selected) == 1 else selected


def select_fmha_impl_for_layer(
    fmha_impl: T | Mapping[str, T],
    kv_cache: LayeredKVCache | None,
    local_layer_idx: int,
) -> T | list[T]:
    if not isinstance(fmha_impl, Mapping):
        return fmha_impl
    if kv_cache is None:
        raise RuntimeError(
            f"KV cache is not initialized for local layer {local_layer_idx}"
        )
    layer_caches = kv_cache.get_layer_cache_groups(local_layer_idx)
    if not layer_caches:
        raise RuntimeError(f"local layer {local_layer_idx} has no KV cache group")
    selected = [
        select_fmha_impl_for_tag(fmha_impl, str(layer_cache.tag))
        for layer_cache in layer_caches
    ]
    return selected[0] if len(selected) == 1 else selected


def select_fmha_impl_for_tag(fmha_impl: Mapping[str, T], tag: str) -> T:
    if not isinstance(fmha_impl, Mapping):
        raise RuntimeError(
            f"sparse MLA requires tagged FMHA implementations, got {type(fmha_impl)!r}"
        )
    try:
        return fmha_impl[tag]
    except KeyError as error:
        raise RuntimeError(
            f"FMHA tag {tag!r} is missing; available tags={list(fmha_impl)}"
        ) from error
