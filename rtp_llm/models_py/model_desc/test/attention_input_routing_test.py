import unittest
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.block_map import get_group_tags_for_layers
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_next import (
    Qwen3NextGatedDeltaNetDecode,
    Qwen3NextMetadata,
    _maybe_write_cp_cache_store,
    _write_cp_cache_store,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import select_cache_group_attn_inputs
from rtp_llm.ops.compute_ops import PyAttentionInputs


def _make_group_view(ordinal: int) -> PyAttentionInputs:
    view = PyAttentionInputs()
    view.kv_cache_block_id = torch.tensor([[ordinal]], dtype=torch.int32)
    view.kv_cache_block_id_device = torch.tensor([[ordinal]], dtype=torch.int32)
    view.kv_cache_kernel_block_id = torch.tensor([[ordinal + 10]], dtype=torch.int32)
    view.kv_cache_kernel_block_id_device = torch.tensor(
        [[ordinal + 10]], dtype=torch.int32
    )
    return view


def make_group_attn_inputs(tags: list[str]) -> dict[str, PyAttentionInputs]:
    return {tag: _make_group_view(ordinal) for ordinal, tag in enumerate(tags)}


class FakeKVCache:
    def __init__(self, layer_tags: list[list[str]]):
        self.layer_tags = layer_tags
        self.ordinals = {
            tag: ordinal
            for ordinal, tag in enumerate(
                sorted({tag for tags in layer_tags for tag in tags})
            )
        }
        self.layer_count = len(layer_tags)
        self.group_tags = sorted(self.ordinals)

    def get_layer_cache_groups(self, layer_idx: int):
        return [
            SimpleNamespace(
                tag=tag,
                execution_ordinal=self.ordinals[tag],
                kv_cache_base=torch.empty(0),
                kv_scale_base=None,
            )
            for tag in self.layer_tags[layer_idx]
        ]


class DuplicateSparseTagMapping(Mapping[str, object]):
    def __init__(self):
        self.values = {"default": object(), "indexer_kv": object()}

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("default", "indexer_kv", "indexer_kv"))

    def __len__(self) -> int:
        return 3


class RoutingModel(GptModelBase):
    def __init__(self, fmha_group_tags: list[str] | None):
        nn.Module.__init__(self)
        self.config = object()
        self.parallelism_config = object()
        self.weight = object()
        self.fmha_config = object()
        self.fmha_group_tags = fmha_group_tags
        selected_tags = fmha_group_tags or ["group0", "group1"]
        self._fmha_cache_tags = tuple(selected_tags)

    def _get_fmha_group_tags(self) -> list[str] | None:
        return self.fmha_group_tags


class AttentionInputRoutingTest(unittest.TestCase):
    def test_xqa_binding_boundary_receives_selected_group_table(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import XQAImpl

        common = PyAttentionInputs()
        common.sequence_lengths = torch.tensor([3], dtype=torch.int32)
        view = _make_group_view(1)
        fmha_op = Mock()
        fmha_op.prepare.return_value = SimpleNamespace()
        rope_op = Mock()
        rope_op.prepare.return_value = SimpleNamespace()

        with (
            patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa.XQAAttnOp",
                return_value=fmha_op,
            ),
            patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa.FusedRopeKVCacheDecodeOp",
                return_value=rope_op,
            ),
            patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa.common.create_write_cache_store_impl",
                return_value=None,
            ),
        ):
            impl = XQAImpl(SimpleNamespace(need_rope_kv_cache=True), common)

        self.assertIs(impl.attn_inputs, common)
        self.assertIs(impl.attn_inputs, common)
        fmha_op.prepare.assert_called_once_with(common)
        rope_op.prepare.assert_called_once_with(common)

    def test_sparse_mla_boundary_receives_narrowed_common_metadata(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaImpl,
        )

        common = PyAttentionInputs()
        common.input_lengths = torch.tensor([2], dtype=torch.int32)
        common.prefix_lengths = torch.tensor([3], dtype=torch.int32)
        common.sequence_lengths = torch.tensor([5], dtype=torch.int32)
        common.is_prefill = True
        view = _make_group_view(1)
        impl = object.__new__(SparseMlaImpl)
        impl.fmha_params = Mock()
        impl.fmha_impl = Mock()
        impl.attn_inputs = common
        impl.seq_size_per_block = 64

        impl.prepare(common, forbid_realloc=True)

        impl.fmha_params.fill_params.assert_called_once_with(
            common.input_lengths,
            common.prefix_lengths,
            common.sequence_lengths,
            common.kv_cache_kernel_block_id,
            common.is_prefill,
            64,
            True,
        )
        impl.fmha_impl.plan.assert_called_once_with(
            impl.fmha_params,
            view.kv_cache_kernel_block_id_device,
            attn_inputs=common,
        )

    def test_generic_sparse_mla_prepares_only_exact_semantic_groups(self):
        model = object.__new__(GenericMoeModel)
        model.__dict__["config"] = SimpleNamespace(
            attn_config=SimpleNamespace(is_sparse=True, use_mla=True)
        )

        self.assertEqual(model._get_fmha_group_tags(), ["default", "indexer_kv"])

    def test_generic_dense_mla_keeps_scalar_group_selection(self):
        model = object.__new__(GenericMoeModel)
        model.__dict__["config"] = SimpleNamespace(
            attn_config=SimpleNamespace(is_sparse=False, use_mla=True)
        )

        self.assertIsNone(model._get_fmha_group_tags())

    def test_generic_sparse_non_mla_keeps_scalar_group_selection(self):
        model = object.__new__(GenericMoeModel)
        model.__dict__["config"] = SimpleNamespace(
            attn_config=SimpleNamespace(is_sparse=True, use_mla=False)
        )

        self.assertIsNone(model._get_fmha_group_tags())

    def test_generic_sparse_mla_rejects_invalid_semantic_tags_during_initialize(self):
        model = object.__new__(GenericMoeModel)
        model.__dict__.update(
            config=SimpleNamespace(
                attn_config=SimpleNamespace(is_sparse=True, use_mla=True)
            ),
            parallelism_config=object(),
            weight=object(),
            fmha_config=object(),
            layer_num=1,
            _fmha_cache_tags=None,
            _sparse_layer_kv_caches=None,
        )
        model._get_fmha_group_tags = Mock(
            return_value=["default", "indexer_kv", "extra"]
        )
        init_resource = SimpleNamespace(
            kv_cache=FakeKVCache([["default", "indexer_kv", "extra"]])
        )

        with self.assertRaisesRegex(RuntimeError, "exactly.*tags"):
            model.initialize(init_resource)

    def test_fmha_impls_are_bound_by_tag(self):
        model = RoutingModel(["group1", "group0"])
        GptModelBase.initialize(
            model, SimpleNamespace(kv_cache=FakeKVCache([["group0", "group1"]]))
        )
        self.assertEqual(model._fmha_cache_tags, ("group1", "group0"))
        model._get_fmha_group_tags = Mock(
            side_effect=AssertionError("hot path must not resolve tags")
        )
        factory_impls = [object(), object()]
        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=factory_impls,
        ):
            result = model.prepare_fmha_impl(
                SimpleNamespace(
                    attention_inputs=PyAttentionInputs(),
                    cache_group_attn_inputs=make_group_attn_inputs(
                        ["group0", "group1"]
                    ),
                )
            )
        self.assertEqual(set(result), {"group0", "group1"})
        # _fmha_cache_tags iterates ("group1", "group0"), so group1 binds first.
        self.assertIs(result["group1"], factory_impls[0])
        self.assertIs(result["group0"], factory_impls[1])

    def test_dsv4_backend_dict_selects_requested_tags(self):
        views = make_group_attn_inputs(["extra", "indexer_kv", "swa_kv"])
        model_inputs = SimpleNamespace(cache_group_attn_inputs=views)

        by_tag = select_cache_group_attn_inputs(model_inputs, ("indexer_kv", "swa_kv"))

        self.assertEqual(tuple(by_tag), ("indexer_kv", "swa_kv"))
        self.assertIs(by_tag["indexer_kv"], views["indexer_kv"])
        self.assertIs(by_tag["swa_kv"], views["swa_kv"])

    def test_qwen3_next_cuda_graph_uses_narrow_block_map_view(self):
        block_map = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=True,
            kv_cache_kernel_block_id_device=block_map,
        )
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)

        narrowed = decode._get_fla_block_map(attention_inputs, block_map)

        self.assertEqual(narrowed.shape, (3, 1))
        self.assertEqual(narrowed.stride(0), block_map.stride(0))
        self.assertEqual(narrowed[:, 0].tolist(), [0, 4, 8])

    def test_qwen3_next_non_graph_keeps_full_block_map(self):
        block_map = torch.arange(12, dtype=torch.int32).reshape(3, 4)
        attention_inputs = SimpleNamespace(
            is_cuda_graph=False,
            kv_cache_kernel_block_id_device=block_map,
        )
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)

        self.assertIs(decode._get_fla_block_map(attention_inputs, block_map), block_map)

    def test_cp_cache_store_uses_each_layer_tag_metadata(self):
        layer_inputs = {}
        for tag in ("full", "linear0", "linear1"):
            cache_store_inputs = SimpleNamespace(tag=tag)
            kv_cache = SimpleNamespace(tag=tag)
            cache_store_writer = Mock()
            layer_inputs[tag] = (
                SimpleNamespace(
                    cache_store_inputs=cache_store_inputs,
                    cache_store_writer=cache_store_writer,
                ),
                kv_cache,
            )

        for tag in ("full", "linear0", "linear1"):
            attention_inputs, kv_cache = layer_inputs[tag]
            pool_block_table = torch.tensor([[1]], dtype=torch.int32)
            _write_cp_cache_store(attention_inputs, kv_cache, pool_block_table)
            attention_inputs.cache_store_writer.write.assert_called_once_with(
                attention_inputs.cache_store_inputs, kv_cache, pool_block_table
            )

    def test_cp_cache_store_skips_layer_without_store_inputs(self):
        cache_store_writer = Mock()
        attention_inputs = SimpleNamespace(
            cache_store_inputs=None, cache_store_writer=cache_store_writer
        )

        _write_cp_cache_store(
            attention_inputs,
            SimpleNamespace(tag="linear0"),
            torch.tensor([[1]], dtype=torch.int32),
        )

        cache_store_writer.write.assert_not_called()

    def test_cp_cache_store_skips_layer_without_writer(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            cache_store_writer=None,
        )

        _write_cp_cache_store(
            attention_inputs,
            SimpleNamespace(tag="linear0"),
            torch.tensor([[1]], dtype=torch.int32),
        )

    def test_non_cp_linear_attention_does_not_write_cache_store(self):
        attention_inputs = SimpleNamespace(
            cache_store_inputs=SimpleNamespace(tag="linear0"),
            cache_store_writer=Mock(),
            context_parallel_info=SimpleNamespace(
                prefill_actual_input_lengths_cpu=torch.tensor([1], dtype=torch.int32)
            ),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            kv_cache_block_id=torch.tensor([[1]], dtype=torch.int32),
        )

        _maybe_write_cp_cache_store(
            attention_inputs,
            SimpleNamespace(tag="linear0"),
            Qwen3NextMetadata(),
            torch.tensor([[1]], dtype=torch.int32),
        )

        attention_inputs.cache_store_writer.write.assert_not_called()

    def test_get_group_tags_for_model_selected_layers(self):
        kv_cache = FakeKVCache([["full"], ["linear0"], ["linear1"], ["full", "aux"]])

        self.assertEqual(get_group_tags_for_layers(kv_cache, [0, 3]), ["full", "aux"])

    def test_prepare_fmha_impl_only_for_model_selected_tags(self):
        common = PyAttentionInputs()
        common.input_lengths = torch.tensor([7], dtype=torch.int32)
        views = make_group_attn_inputs(["full", "linear0", "linear1"])
        inputs = SimpleNamespace(attention_inputs=common, cache_group_attn_inputs=views)
        model = RoutingModel(["full"])

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, group_view, _fmha_config, _is_cuda_graph: (
                SimpleNamespace(common=group_inputs, view=group_view)
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertEqual(set(fmha_impl), {"full"})
        self.assertIs(fmha_impl["full"].common, views["full"])
        self.assertIsNone(fmha_impl["full"].view)
        factory.assert_called_once()

    def test_default_model_prepares_every_tag(self):
        common = PyAttentionInputs()
        common.input_lengths = torch.tensor([7], dtype=torch.int32)
        views = make_group_attn_inputs(["group0", "group1"])
        inputs = SimpleNamespace(attention_inputs=common, cache_group_attn_inputs=views)
        model = RoutingModel(None)

        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
            side_effect=lambda _config, _parallelism_config, _weight, group_inputs, group_view, _fmha_config, _is_cuda_graph: (
                SimpleNamespace(common=group_inputs, view=group_view)
            ),
        ) as factory:
            fmha_impl = model.prepare_fmha_impl(inputs)

        self.assertEqual(set(fmha_impl), {"group0", "group1"})
        self.assertEqual(
            {tag: impl.common for tag, impl in fmha_impl.items()},
            {"group0": views["group0"], "group1": views["group1"]},
        )
        self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
