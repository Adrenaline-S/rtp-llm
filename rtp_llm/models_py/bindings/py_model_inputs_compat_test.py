import ast
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.model_desc.block_map import (
    get_layer_cache_for_tag,
    get_layer_caches_for_tags,
    select_cache_group_attn_inputs_for_layer,
    select_fmha_impl_for_tag,
)
from rtp_llm.models_py.utils.kvcache import SingleGroupKVCacheAdapter
from rtp_llm.ops import HybridAttentionConfig, HybridAttentionType
from rtp_llm.ops.compute_ops import (
    CacheStoreWriter,
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)


class _RoutingCache:
    def __init__(self, layer_tags: list[list[str]]) -> None:
        self._layer_tags = layer_tags

    def get_layer_cache_groups(self, layer_id: int) -> list[LayerKVCache]:
        return [
            LayerKVCache(
                torch.ones(1),
                1,
                layer_id,
                tag,
                execution_ordinal=ordinal,
            )
            for ordinal, tag in enumerate(self._layer_tags[layer_id])
        ]


class _ConcreteRoutingCache:
    def __init__(self, caches: list[LayerKVCache]) -> None:
        self._caches = caches

    def get_layer_cache_groups(self, layer_id: int) -> list[LayerKVCache]:
        if layer_id != 0:
            raise RuntimeError(f"invalid layer {layer_id}")
        return self._caches


class PyModelInputsCompatTest(unittest.TestCase):
    def test_hybrid_attention_config_has_explicit_constructors(self) -> None:
        default_config = HybridAttentionConfig()
        self.assertFalse(default_config.enable_hybrid_attention)
        self.assertFalse(default_config.enable_independent_kv_cache_pools)
        self.assertEqual(default_config.hybrid_attention_types, [])

        attention_types = [HybridAttentionType.NONE, HybridAttentionType.LINEAR]
        config = HybridAttentionConfig(True, True, attention_types)
        self.assertTrue(config.enable_hybrid_attention)
        self.assertTrue(config.enable_independent_kv_cache_pools)
        self.assertEqual(config.hybrid_attention_types, attention_types)

        with self.assertRaises(TypeError):
            HybridAttentionConfig(True, True)

    def test_sparse_routes_select_exact_tags_independent_of_topology_order(
        self,
    ) -> None:
        default_cache = LayerKVCache(torch.ones(1), 64, layer_id=0, tag="default")
        indexer_cache = LayerKVCache(
            torch.ones(1) * 2, 64, layer_id=0, tag="indexer_kv"
        )
        cache = _ConcreteRoutingCache([indexer_cache, default_cache])

        self.assertIs(get_layer_cache_for_tag(cache, 0, "default"), default_cache)
        self.assertIs(get_layer_cache_for_tag(cache, 0, "indexer_kv"), indexer_cache)
        self.assertEqual(
            get_layer_caches_for_tags(cache, 0, ("default", "indexer_kv")),
            {"default": default_cache, "indexer_kv": indexer_cache},
        )
        routes = {"indexer_kv": object(), "default": object()}
        self.assertIs(select_fmha_impl_for_tag(routes, "default"), routes["default"])
        self.assertIs(
            select_fmha_impl_for_tag(routes, "indexer_kv"), routes["indexer_kv"]
        )

    def test_sparse_routes_reject_absent_duplicate_and_wrong_tags(self) -> None:
        absent = _ConcreteRoutingCache([LayerKVCache(torch.ones(1), 64, 0, "default")])
        with self.assertRaisesRegex(RuntimeError, "indexer_kv"):
            get_layer_cache_for_tag(absent, 0, "indexer_kv")

        duplicate = _ConcreteRoutingCache(
            [
                LayerKVCache(torch.ones(1), 64, 0, "indexer_kv"),
                LayerKVCache(torch.ones(1), 64, 0, "indexer_kv"),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate KV cache tag"):
            get_layer_cache_for_tag(duplicate, 0, "indexer_kv")

        with self.assertRaisesRegex(RuntimeError, "indexer_kv"):
            select_fmha_impl_for_tag({"wrong": object()}, "indexer_kv")

    def test_cache_binding_stubs_match_runtime_members(self) -> None:
        stub_path = (
            Path(__file__).resolve().parents[2]
            / "ops"
            / "librtp_compute_ops"
            / "__init__.pyi"
        )
        module = ast.parse(stub_path.read_text())
        stub_classes = {
            node.name: node for node in module.body if isinstance(node, ast.ClassDef)
        }

        for class_name, runtime_class in (
            ("CacheStoreWriter", CacheStoreWriter),
            ("KVCache", KVCache),
            ("LayerKVCache", LayerKVCache),
            ("PyAttentionInputs", PyAttentionInputs),
        ):
            with self.subTest(class_name=class_name):
                stub_members = set()
                for node in stub_classes[class_name].body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        stub_members.add(node.name)
                    elif isinstance(node, ast.AnnAssign) and isinstance(
                        node.target, ast.Name
                    ):
                        stub_members.add(node.target.id)

                stub_members = {
                    name for name in stub_members if not name.startswith("_")
                }
                runtime_members = {
                    name for name in vars(runtime_class) if not name.startswith("_")
                }
                self.assertEqual(stub_members, runtime_members)

    def _attn_inputs(self, is_prefill: bool, input_length: int) -> PyAttentionInputs:
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = is_prefill
        attn_inputs.input_lengths = torch.tensor([input_length], dtype=torch.int32)
        return attn_inputs

    def test_default_constructor_then_assign_attention_inputs(self) -> None:
        attn_inputs = self._attn_inputs(is_prefill=True, input_length=3)

        inputs = PyModelInputs()
        inputs.attention_inputs = attn_inputs

        self.assertTrue(inputs.attention_inputs.is_prefill)
        self.assertEqual(3, inputs.attention_inputs.input_lengths.item())

    def test_model_outputs_only_exposes_hidden_states(self) -> None:
        hidden_states = torch.empty(0)
        outputs = PyModelOutputs(hidden_states)

        self.assertEqual(hidden_states.data_ptr(), outputs.hidden_states.data_ptr())
        self.assertFalse(hasattr(outputs, "params_ptr"))
        with self.assertRaises(TypeError):
            PyModelOutputs(hidden_states, {"full": None})

    def test_attention_inputs_field_updates_directly(self) -> None:
        inputs = PyModelInputs()
        inputs.attention_inputs = self._attn_inputs(is_prefill=False, input_length=2)

        self.assertFalse(inputs.attention_inputs.is_prefill)
        inputs.attention_inputs.is_prefill = True
        inputs.attention_inputs.input_lengths = torch.tensor([5], dtype=torch.int32)
        self.assertTrue(inputs.attention_inputs.is_prefill)
        self.assertEqual(5, inputs.attention_inputs.input_lengths.item())

    def test_multi_group_inputs_are_tag_keyed_alias_views(self) -> None:
        common = self._attn_inputs(is_prefill=False, input_length=2)
        common.kv_cache_block_id = torch.arange(10, dtype=torch.int32)
        common.kv_cache_kernel_block_id = torch.arange(16, dtype=torch.int32)
        full = PyAttentionInputs()
        full.kv_cache_block_id = common.kv_cache_block_id.narrow(0, 0, 6).view(2, 3)
        full.kv_cache_kernel_block_id = common.kv_cache_kernel_block_id.narrow(
            0, 0, 12
        ).view(2, 6)
        swa = PyAttentionInputs()
        swa.kv_cache_block_id = common.kv_cache_block_id.narrow(0, 6, 4).view(2, 2)
        swa.kv_cache_kernel_block_id = common.kv_cache_kernel_block_id.narrow(
            0, 12, 4
        ).view(2, 2)

        inputs = PyModelInputs()
        inputs.attention_inputs = common
        inputs.cache_group_attn_inputs = {"default": full, "swa": swa}

        self.assertIsInstance(inputs.attention_inputs, PyAttentionInputs)
        group_attn_inputs = inputs.cache_group_attn_inputs
        self.assertIsInstance(group_attn_inputs, dict)
        self.assertEqual({"default", "swa"}, set(group_attn_inputs))
        for tag, group in group_attn_inputs.items():
            self.assertIsInstance(tag, str)
            self.assertIsInstance(group, PyAttentionInputs)
            self.assertEqual(2, group.kv_cache_kernel_block_id.dim())
        self.assertEqual(
            (2, 6),
            tuple(group_attn_inputs["default"].kv_cache_kernel_block_id.shape),
        )
        self.assertEqual(
            common.kv_cache_block_id.data_ptr(),
            group_attn_inputs["default"].kv_cache_block_id.data_ptr(),
        )
        self.assertFalse(hasattr(inputs, "attention_inputs_by_tag"))

    def test_kv_cache_is_runtime_constructed_and_read_only(self) -> None:
        with self.assertRaises(TypeError):
            KVCache()

        for old_name in (
            "kv_cache_base_by_layer",
            "kv_scale_base_by_layer",
            "seq_size_per_block",
            "kernel_seq_size_per_block",
            "num_kv_heads",
            "head_dim",
            "use_mla",
            "kv_lora_rank",
            "rope_head_dim",
            "layer_attn_types",
            "group_types",
            "group_seq_block_sizes",
            "group_kernel_seq_block_sizes",
            "layer_to_group_ids",
            "layer_tag_to_group_id",
            "get_layer_caches",
        ):
            self.assertFalse(hasattr(KVCache, old_name), old_name)

        for new_name in (
            "group_tags",
            "layer_count",
            "get_layer_cache",
            "get_layer_cache_groups",
            "get_seq_size_per_block",
            "get_kernel_seq_size_per_block",
        ):
            self.assertTrue(hasattr(KVCache, new_name), new_name)

    def test_layer_kv_cache_parameterized_constructor(self) -> None:
        base = torch.arange(8, dtype=torch.float16).reshape(2, 4)
        scale = torch.ones((2, 1), dtype=torch.float32)

        layer = LayerKVCache(
            base,
            16,
            layer_id=3,
            tag="full",
            kv_scale_base=scale,
        )

        self.assertEqual(base.data_ptr(), layer.kv_cache_base.data_ptr())
        self.assertEqual(scale.data_ptr(), layer.kv_scale_base.data_ptr())
        self.assertEqual(16, layer.seq_size_per_block)
        self.assertEqual(3, layer.layer_id)
        self.assertEqual("full", layer.tag)

    def test_layer_kv_cache_exposes_execution_ordinal_not_business_group_id(
        self,
    ) -> None:
        self.assertFalse(hasattr(LayerKVCache, "group_id"))
        layer = LayerKVCache(
            torch.ones(1), 8, layer_id=0, tag="full", execution_ordinal=1
        )
        self.assertFalse(hasattr(layer, "group_id"))
        self.assertEqual(1, layer.execution_ordinal)
        with self.assertRaises(TypeError):
            LayerKVCache(torch.ones(1), 8, layer_id=0, group_id=0, tag="full")
        # The 4th positional argument is the tag, not an ordinal.
        self.assertEqual("linear", LayerKVCache(torch.ones(1), 8, 0, "linear").tag)

    def test_cache_group_attn_inputs_is_selected_by_layer_tag(self) -> None:
        inputs = PyModelInputs()
        full = PyAttentionInputs()
        full.kv_cache_kernel_block_id = torch.tensor([[10]], dtype=torch.int32)
        linear = PyAttentionInputs()
        linear.kv_cache_kernel_block_id = torch.tensor([[20]], dtype=torch.int32)
        inputs.cache_group_attn_inputs = {"full": full, "linear": linear}
        cache = _RoutingCache([["full", "linear"]])
        selected = select_cache_group_attn_inputs_for_layer(inputs, cache, 0)

        self.assertEqual(
            [10, 20], [view.kv_cache_kernel_block_id.item() for view in selected]
        )

    def test_missing_cache_tag_reports_available_tags(self) -> None:
        inputs = PyModelInputs()
        inputs.cache_group_attn_inputs = {"default": PyAttentionInputs()}

        def select(cache_tags):
            group_attn_inputs = inputs.cache_group_attn_inputs
            for tag in cache_tags:
                if tag not in group_attn_inputs:
                    raise RuntimeError(
                        f"FMHA cache tag {tag!r} has no attention inputs; "
                        f"available tags: {sorted(group_attn_inputs)}"
                    )

        with self.assertRaises(RuntimeError) as ctx:
            select(("default", "linear"))
        message = str(ctx.exception)
        self.assertIn("linear", message)
        self.assertIn("default", message)

    def test_single_group_adapter_returns_native_layer_cache(self) -> None:
        tensors = [torch.zeros((2, 2, 1, 8, 4), dtype=torch.float16)]
        cache = SingleGroupKVCacheAdapter(tensors, 8)

        layer = cache.get_layer_cache(0)
        self.assertIsInstance(layer, LayerKVCache)
        self.assertEqual(tensors[0].data_ptr(), layer.kv_cache_base.data_ptr())
        self.assertEqual(["default"], cache.group_tags)
        self.assertEqual(1, cache.layer_count)
        self.assertEqual(8, cache.get_seq_size_per_block("default"))
        self.assertEqual(
            ["default"], [item.tag for item in cache.get_layer_cache_groups(0)]
        )
        self.assertFalse(hasattr(cache, "get_layer_caches"))


if __name__ == "__main__":
    unittest.main()
