import unittest

import torch

from rtp_llm.cpp.models.test.libth_pywrapped_model_cache_store_integration_test import (
    PyModelInputs,
    PyModelOutputs,
    run_cache_free_graph_lifecycle,
    run_kernel_block_table_update_lifecycle,
    run_large_tail_update_lifecycle,
    run_scenario,
)


class CacheStoreForwardModel:
    """Test model that replaces attention math but keeps the real cache-store call."""

    def __init__(self) -> None:
        self.kv_cache = None
        self.forward_calls = 0
        self.micro_batch_calls = 0
        self.seen_input_lengths: list[list[int]] = []

    def initialize(self, resources) -> bool:
        self.kv_cache = resources.kv_cache
        return True

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        self.seen_input_lengths.append(attention_inputs.input_lengths.tolist())

        assert self.kv_cache is not None
        for layer_cache in self.kv_cache.get_layer_cache_groups(0):
            group_view = inputs.cache_group_attn_inputs[str(layer_cache.tag)]
            if (
                attention_inputs.cache_store_inputs is not None
                and attention_inputs.cache_store_writer is not None
            ):
                attention_inputs.cache_store_writer.write(
                    attention_inputs.cache_store_inputs,
                    layer_cache,
                    group_view.kv_cache_block_id,
                )

        hidden_states = torch.zeros(
            (inputs.input_ids.numel(), 1),
            dtype=torch.float16,
            device=inputs.input_ids.device,
        )
        return PyModelOutputs(hidden_states)

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        return self._forward_one(inputs)

    def forward_micro_batch(self, inputs: list[PyModelInputs]) -> list[PyModelOutputs]:
        self.micro_batch_calls += 1
        return [self._forward_one(model_inputs) for model_inputs in inputs]


class CacheFreeBoundaryModel(CacheStoreForwardModel):
    def initialize(self, resources) -> bool:
        self.kv_cache = None
        return True

    def _forward_one(self, inputs: PyModelInputs) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        self.seen_input_lengths.append(attention_inputs.input_lengths.tolist())
        for table in (
            attention_inputs.kv_cache_block_id,
            attention_inputs.kv_cache_block_id_device,
            attention_inputs.kv_cache_kernel_block_id,
            attention_inputs.kv_cache_kernel_block_id_device,
        ):
            assert table is not None
            assert tuple(table.shape) == (3, 0)
        return PyModelOutputs(
            torch.zeros(
                (inputs.input_ids.numel(), 1),
                dtype=torch.float16,
                device=inputs.input_ids.device,
            )
        )


def _blocks_by_key(result: dict) -> dict[str, dict]:
    return {
        block["key"]: block
        for record in result["records"]
        for block in record["blocks"]
    }


def _record_for_request(result: dict, request_id: int) -> dict:
    matches = [
        record
        for record in result["records"]
        if record["request_id"] == str(request_id)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one record for request {request_id}, got {len(matches)}"
        )
    return matches[0]


def _offsets_by_tag(result: dict) -> dict:
    blocks = _blocks_by_key(result)
    offsets = {}
    for tag in ("full", "linear"):
        offsets[tag] = sorted(
            block["address"] - result["base_addresses"][tag]
            for key, block in blocks.items()
            if ("_tag_" + tag) in key
        )
    return offsets


class PyWrappedModelCacheStoreIntegrationTest(unittest.TestCase):
    def test_cache_free_prepare_publishes_canonical_tables_to_graph(self) -> None:
        model = CacheFreeBoundaryModel()
        result = run_cache_free_graph_lifecycle(model)
        self.assertEqual(result["table_shapes"], ([3, 0],) * 4)
        self.assertEqual(result["table_device_flags"], (False, True, False, True))
        self.assertEqual(result["group_view_count"], 0)
        self.assertEqual(result["graph_calls"], (2, 1, 1))
        self.assertEqual(model.seen_input_lengths, [[4, 3, 2]])

    def test_large_heterogeneous_tail_update_uses_one_cuda_launch(self) -> None:
        result = run_large_tail_update_lifecycle(CacheStoreForwardModel())
        self.assertTrue(result["backing_stable"])
        self.assertEqual(result["short_row_count"], 65)
        self.assertEqual(result["device_tail_fill_launches"], 1)
        self.assertTrue(result["host_tails_cleared"])
        self.assertTrue(result["device_tails_cleared"])

    def test_kernel_value_update_preserves_prepared_packed_storage(self) -> None:
        result = run_kernel_block_table_update_lifecycle(CacheStoreForwardModel())
        self.assertTrue(result["backings_stable"])
        self.assertTrue(result["views_stable"])
        self.assertTrue(result["pool_unchanged"])
        expected_kernel_values = [
            101,
            -1,
            -1,
            -1,
            105,
            106,
            107,
            108,
            -1,
            -1,
            111,
            112,
            113,
            -1,
            115,
            -1,
            -1,
            -1,
        ]
        self.assertEqual(result["kernel_host_values"].tolist(), expected_kernel_values)
        self.assertEqual(
            result["kernel_device_values"].tolist(), expected_kernel_values
        )
        self.assertEqual(
            tuple(value.tolist() for value in result["kernel_valid_lengths"]),
            ([1, 0, 2], [2, 3, 1]),
        )
        self.assertEqual(result["device_tail_fill_launches"], 1)
        self.assertIn("full prepare", result["structural_rejection"])

    def test_multi_tag_binding_ignores_cache_group_declaration_order(self) -> None:
        # Dense execution ordinals are assigned from sorted tags during plan
        # construction, independent of CacheConfig declaration order.
        unsorted_result = run_scenario(CacheStoreForwardModel(), "multi_tag")
        sorted_result = run_scenario(
            CacheStoreForwardModel(), "multi_tag_sorted_declaration"
        )

        self.assertEqual(
            _offsets_by_tag(unsorted_result), _offsets_by_tag(sorted_result)
        )
        self.assertEqual(
            _offsets_by_tag(sorted_result),
            {"full": [16, 32], "linear": [72, 96, 120, 144]},
        )

    def test_tp_non_root_reconstructs_single_tag_after_tensor_sync(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "tp_non_root_single_tag")
        self.assertEqual(model.forward_calls, 1)
        record = _record_for_request(result, 351)
        base = result["base_addresses"]["default"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [16, 32],
        )

    def test_tp_non_root_reconstructs_reordered_multi_group_parallel_rows(self) -> None:
        canonical = run_scenario(CacheStoreForwardModel(), "multi_tag")
        reconstructed = run_scenario(CacheStoreForwardModel(), "tp_non_root_multi_tag")
        self.assertEqual(_offsets_by_tag(reconstructed), _offsets_by_tag(canonical))

    def test_multi_tag_uses_each_tag_local_physical_block_table(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "multi_tag")

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(len(result["records"]), 2)
        blocks = _blocks_by_key(result)

        full_blocks = {
            key: block for key, block in blocks.items() if "_tag_full" in key
        }
        linear_blocks = {
            key: block for key, block in blocks.items() if "_tag_linear" in key
        }
        self.assertEqual(len(full_blocks), 2)
        self.assertEqual(len(linear_blocks), 4)
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["full"]
                for block in full_blocks.values()
            ),
            [16, 32],
        )
        self.assertEqual(
            sorted(
                block["address"] - result["base_addresses"]["linear"]
                for block in linear_blocks.values()
            ),
            [72, 96, 120, 144],
        )
        self.assertEqual({block["length"] for block in full_blocks.values()}, {16})
        self.assertEqual({block["length"] for block in linear_blocks.values()}, {24})

    def test_micro_batch_slices_request_metadata_with_block_rows(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "micro_batch")

        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(model.micro_batch_calls, 1)
        self.assertEqual(model.seen_input_lengths, [[2, 4], [2]])
        self.assertEqual(len(result["records"]), 3)

        expected = {
            201: ([2101], [16]),
            202: ([2201, 2202], [32, 48]),
            203: ([2301], [64]),
        }
        base = result["base_addresses"]["default"]
        for request_id, (token_keys, offsets) in expected.items():
            record = _record_for_request(result, request_id)
            self.assertEqual(len(record["blocks"]), len(token_keys))
            self.assertEqual(
                sorted(block["address"] - base for block in record["blocks"]),
                offsets,
            )
            for token_key in token_keys:
                self.assertTrue(
                    any(
                        f"_token_id_str_{token_key}_" in block["key"]
                        for block in record["blocks"]
                    )
                )

    def test_context_parallel_publishes_original_lengths_not_local_chunk(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "cp_actual_lengths")

        # CP turns the six-token request into a four-token rank-local chunk for
        # attention, while CacheStore must still publish three two-token blocks.
        self.assertEqual(model.seen_input_lengths, [[4]])
        record = _record_for_request(result, 301)
        self.assertEqual(len(record["blocks"]), 3)
        base = result["base_addresses"]["default"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [16, 32, 48],
        )
        self.assertEqual(
            sorted(
                token_key
                for token_key in (3102, 3104, 3106)
                if any(
                    f"_token_id_str_{token_key}_" in block["key"]
                    for block in record["blocks"]
                )
            ),
            [3102, 3104, 3106],
        )

    def test_mtp_writer_uses_selected_sub_config_for_real_write(self) -> None:
        model = CacheStoreForwardModel()
        result = run_scenario(model, "mtp_sub_config")

        record = _record_for_request(result, 401)
        self.assertEqual(len(record["blocks"]), 2)
        base = result["base_addresses"]["draft"]
        self.assertEqual(
            sorted(block["address"] - base for block in record["blocks"]),
            [32, 64],
        )
        self.assertEqual({block["length"] for block in record["blocks"]}, {32})
        self.assertTrue(
            all("model_id_7_" in block["key"] for block in record["blocks"])
        )
        self.assertTrue(all("_tag_draft" in block["key"] for block in record["blocks"]))


if __name__ == "__main__":
    unittest.main()
