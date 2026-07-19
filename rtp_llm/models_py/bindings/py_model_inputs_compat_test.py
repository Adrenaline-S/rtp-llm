import ast
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.model_desc.block_map import select_attention_inputs_for_layer
from rtp_llm.models_py.utils.kvcache import SingleGroupKVCacheAdapter
from rtp_llm.ops.compute_ops import (
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
            LayerKVCache(torch.ones(1), 1, layer_id, group_id, tag)
            for group_id, tag in enumerate(self._layer_tags[layer_id])
        ]


class PyModelInputsCompatTest(unittest.TestCase):
    def _stub_public_members(self, class_name: str) -> set[str]:
        stub_path = (
            Path(__file__).parents[2] / "ops/librtp_compute_ops/__init__.pyi"
        )
        module = ast.parse(stub_path.read_text(encoding="utf-8"))
        class_node = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )

        members: set[str] = set()
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                members.add(node.target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("__"):
                    members.add(node.name)
        return members

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

    def test_model_outputs_discards_python_only_params(self) -> None:
        outputs = PyModelOutputs(torch.empty(0), {"full": None})
        self.assertIsNone(outputs.params_ptr)

    def test_attention_inputs_field_updates_directly(self) -> None:
        inputs = PyModelInputs()
        inputs.attention_inputs = self._attn_inputs(is_prefill=False, input_length=2)

        self.assertFalse(inputs.attention_inputs.is_prefill)
        inputs.attention_inputs.is_prefill = True
        inputs.attention_inputs.input_lengths = torch.tensor([5], dtype=torch.int32)
        self.assertTrue(inputs.attention_inputs.is_prefill)
        self.assertEqual(5, inputs.attention_inputs.input_lengths.item())

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
            group_id=2,
            tag="full",
            kv_scale_base=scale,
        )

        self.assertEqual(base.data_ptr(), layer.kv_cache_base.data_ptr())
        self.assertEqual(scale.data_ptr(), layer.kv_scale_base.data_ptr())
        self.assertEqual(16, layer.seq_size_per_block)
        self.assertEqual(3, layer.layer_id)
        self.assertEqual(2, layer.group_id)
        self.assertEqual("full", layer.tag)

    def test_attention_inputs_mapping_is_selected_by_layer_tag(self) -> None:
        full = self._attn_inputs(is_prefill=False, input_length=1)
        linear = self._attn_inputs(is_prefill=False, input_length=1)
        full.kv_cache_kernel_block_id = torch.tensor([[10]], dtype=torch.int32)
        linear.kv_cache_kernel_block_id = torch.tensor([[20]], dtype=torch.int32)

        inputs = PyModelInputs()
        inputs.attention_inputs = {"full": full, "linear": linear}
        selected = select_attention_inputs_for_layer(
            inputs, _RoutingCache([["linear"]]), 0
        )

        self.assertEqual(20, selected.kv_cache_kernel_block_id.item())
        self.assertFalse(hasattr(selected, "kv_cache_kernel_block_id_by_group"))
        self.assertFalse(hasattr(selected, "kv_cache_layer_to_group"))

    def test_cache_binding_stub_members_exist_at_runtime(self) -> None:
        for bound_type in (KVCache, LayerKVCache, PyAttentionInputs):
            with self.subTest(class_name=bound_type.__name__):
                for member in self._stub_public_members(bound_type.__name__):
                    self.assertTrue(
                        hasattr(bound_type, member),
                        f"{bound_type.__name__}.{member} is declared in the stub "
                        "but missing from the runtime binding",
                    )

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
