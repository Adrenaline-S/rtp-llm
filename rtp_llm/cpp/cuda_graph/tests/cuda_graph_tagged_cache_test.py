import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)

GROUP_TAGS = ["aux", "full"]
MIXED_ATTENTION_TAGS = ["full", "linear"]
HIDDEN_SIZE = 4
TOKENS_PER_BLOCK = 8


class TaggedBlockTableModel:
    """Small graph-safe model whose output exposes both tag-local block tables."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        common = inputs.attention_inputs
        for group_inputs in inputs.cache_group_attn_inputs.values():
            for field in (
                "prefix_lengths",
                "sequence_lengths",
                "input_lengths",
                "cu_seqlens",
                "cu_seqlens_device",
                "cu_kv_seqlens_device",
                "padding_offset",
                "input_lengths_device",
                "sequence_lengths_plus_1_device",
                "decode_cu_seqlens_device",
            ):
                common_tensor = getattr(common, field)
                group_tensor = getattr(group_inputs, field)
                assert group_tensor is not None, field
                assert group_tensor.data_ptr() == common_tensor.data_ptr()
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        aux_id = inputs.cache_group_attn_inputs["aux"].kv_cache_kernel_block_id_device[
            0, 0
        ]
        full_id = inputs.cache_group_attn_inputs[
            "full"
        ].kv_cache_kernel_block_id_device[0, 0]
        signature = (full_id + 16 * aux_id).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class _FullCudaGraphImpl:
    def __init__(self, common: PyAttentionInputs, full: PyAttentionInputs) -> None:
        self.common_input_lengths_ptr = common.input_lengths.data_ptr()
        self.full_table_ptr = full.kv_cache_kernel_block_id_device.data_ptr()

    def prepare_cuda_graph(self, attention_inputs: PyAttentionInputs) -> None:
        assert (
            attention_inputs.input_lengths.data_ptr() == self.common_input_lengths_ptr
        )
        assert (
            attention_inputs.kv_cache_kernel_block_id_device.data_ptr()
            == self.full_table_ptr
        )


class TaggedSequenceLengthModel:
    """Expose the cumulative lengths used by a tagged captured graph."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        common = inputs.attention_inputs
        signature = torch.stack(
            (
                common.cu_seqlens_device[-1],
                common.cu_kv_seqlens_device[-1],
                common.input_lengths_device.sum(),
                common.prefix_lengths_device.sum(),
            )
        ).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class MixedFullLinearModel:
    """Hybrid model where only the full cache group owns an FMHA impl."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return {
            "full": _FullCudaGraphImpl(
                inputs.attention_inputs, inputs.cache_group_attn_inputs["full"]
            )
        }

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        full_id = inputs.cache_group_attn_inputs[
            "full"
        ].kv_cache_kernel_block_id_device[0, 0]
        linear_id = inputs.cache_group_attn_inputs[
            "linear"
        ].kv_cache_kernel_block_id_device[0, 0]
        signature = (full_id + 16 * linear_id).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class CacheFreeModel:
    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        return PyModelOutputs(inputs.input_hiddens)


def _build_common_inputs(
    attention_inputs: PyAttentionInputs,
    tags: list[str],
    values: dict[str, int],
    batch_size: int,
    token_count: int,
    block_count: int,
) -> PyModelInputs:
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(token_count, dtype=torch.int32, device="cuda")
    inputs.input_hiddens = torch.zeros(
        (token_count, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )

    attention_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    attention_inputs.padding_offset = torch.zeros(
        token_count, dtype=torch.int32, device="cuda"
    )
    attention_inputs.total_tokens = token_count
    canonical_tags = sorted(tags)
    host_segments = [
        torch.full(
            (batch_size, block_count), values[tag], dtype=torch.int32
        ).pin_memory()
        for tag in canonical_tags
    ]
    attention_inputs.kv_cache_kernel_block_id = torch.cat(
        [segment.reshape(-1) for segment in host_segments]
    ).pin_memory()
    attention_inputs.kv_cache_kernel_block_id_device = (
        attention_inputs.kv_cache_kernel_block_id.cuda()
    )
    attention_inputs.kv_cache_block_id = (
        attention_inputs.kv_cache_kernel_block_id.clone().pin_memory()
    )
    attention_inputs.kv_cache_block_id_device = (
        attention_inputs.kv_cache_block_id.cuda()
    )
    inputs.attention_inputs = attention_inputs
    group_attn_inputs = {}
    offset = 0
    segment_numel = batch_size * block_count
    for _ordinal, tag in enumerate(canonical_tags):
        view = PyAttentionInputs()
        view.kv_cache_kernel_block_id = (
            attention_inputs.kv_cache_kernel_block_id.narrow(
                0, offset, segment_numel
            ).view(batch_size, block_count)
        )
        view.kv_cache_kernel_block_id_device = (
            attention_inputs.kv_cache_kernel_block_id_device.narrow(
                0, offset, segment_numel
            ).view(batch_size, block_count)
        )
        view.kv_cache_block_id = attention_inputs.kv_cache_block_id.narrow(
            0, offset, segment_numel
        ).view(batch_size, block_count)
        view.kv_cache_block_id_device = (
            attention_inputs.kv_cache_block_id_device.narrow(
                0, offset, segment_numel
            ).view(batch_size, block_count)
        )
        view.pool_valid_lengths = torch.full(
            (batch_size,), block_count, dtype=torch.int32
        ).pin_memory()
        view.kernel_valid_lengths = torch.full(
            (batch_size,), block_count, dtype=torch.int32
        ).pin_memory()
        group_attn_inputs[tag] = view
        offset += segment_numel
    inputs.cache_group_attn_inputs = group_attn_inputs
    return inputs


def _build_decode_inputs(
    tags: list[str],
    values: dict[str, int],
    batch_size: int = 2,
) -> PyModelInputs:
    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = False
    attention_inputs.is_target_verify = False
    attention_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
    attention_inputs.input_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths_plus_1_device = torch.full(
        (batch_size,), 2, dtype=torch.int32, device="cuda"
    )
    attention_inputs.decode_cu_seqlens_device = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cuda"
    )
    attention_inputs.cu_seqlens = torch.zeros(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = torch.zeros_like(
        attention_inputs.cu_seqlens_device
    )
    attention_inputs.context_total_kv_length = batch_size
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=batch_size,
        token_count=batch_size,
        block_count=1,
    )


def _build_cache_free_decode_inputs(batch_size: int = 2) -> PyModelInputs:
    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = False
    attention_inputs.is_target_verify = False
    attention_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
    attention_inputs.input_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths_plus_1_device = torch.full(
        (batch_size,), 2, dtype=torch.int32, device="cuda"
    )
    attention_inputs.decode_cu_seqlens_device = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cuda"
    )
    attention_inputs.cu_seqlens = torch.zeros(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = torch.zeros_like(
        attention_inputs.cu_seqlens_device
    )
    attention_inputs.context_total_kv_length = batch_size
    attention_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    attention_inputs.padding_offset = torch.zeros(
        batch_size, dtype=torch.int32, device="cuda"
    )
    attention_inputs.total_tokens = batch_size
    attention_inputs.kv_cache_block_id = torch.empty(
        (batch_size, 0), dtype=torch.int32
    ).pin_memory()
    attention_inputs.kv_cache_block_id_device = torch.empty(
        (batch_size, 0), dtype=torch.int32, device="cuda"
    )
    attention_inputs.kv_cache_kernel_block_id = torch.empty(
        (batch_size, 0), dtype=torch.int32
    ).pin_memory()
    attention_inputs.kv_cache_kernel_block_id_device = torch.empty(
        (batch_size, 0), dtype=torch.int32, device="cuda"
    )

    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    inputs.input_hiddens = torch.zeros(
        (batch_size, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    inputs.attention_inputs = attention_inputs
    inputs.cache_group_attn_inputs = {}
    return inputs


def _build_heterogeneous_decode_inputs(
    tags: list[str],
    widths: dict[str, tuple[int, int]],
    values: dict[str, int],
    batch_size: int,
) -> PyModelInputs:
    inputs = _build_decode_inputs(tags, values, batch_size=batch_size)
    canonical_tags = sorted(tags)
    pool_segments = []
    kernel_segments = []
    for tag in canonical_tags:
        pool_width, kernel_width = widths[tag]
        pool_segments.append(
            torch.full(
                (batch_size, pool_width), values[tag], dtype=torch.int32
            ).pin_memory()
        )
        kernel_segments.append(
            torch.full(
                (batch_size, kernel_width), values[tag], dtype=torch.int32
            ).pin_memory()
        )
    pool_host = torch.cat(
        [segment.reshape(-1) for segment in pool_segments]
    ).pin_memory()
    kernel_host = torch.cat(
        [segment.reshape(-1) for segment in kernel_segments]
    ).pin_memory()
    inputs.attention_inputs.kv_cache_block_id = pool_host
    inputs.attention_inputs.kv_cache_block_id_device = pool_host.cuda()
    inputs.attention_inputs.kv_cache_kernel_block_id = kernel_host
    inputs.attention_inputs.kv_cache_kernel_block_id_device = kernel_host.cuda()
    group_attn_inputs = {}
    pool_offset = 0
    kernel_offset = 0
    for _ordinal, tag in enumerate(canonical_tags):
        pool_width, kernel_width = widths[tag]
        view = PyAttentionInputs()
        pool_numel = batch_size * pool_width
        kernel_numel = batch_size * kernel_width
        view.kv_cache_block_id = pool_host.narrow(0, pool_offset, pool_numel).view(
            batch_size, pool_width
        )
        view.kv_cache_block_id_device = (
            inputs.attention_inputs.kv_cache_block_id_device.narrow(
                0, pool_offset, pool_numel
            ).view(batch_size, pool_width)
        )
        view.kv_cache_kernel_block_id = kernel_host.narrow(
            0, kernel_offset, kernel_numel
        ).view(batch_size, kernel_width)
        view.kv_cache_kernel_block_id_device = (
            inputs.attention_inputs.kv_cache_kernel_block_id_device.narrow(
                0, kernel_offset, kernel_numel
            ).view(batch_size, kernel_width)
        )
        view.pool_valid_lengths = torch.full(
            (batch_size,), pool_width, dtype=torch.int32
        ).pin_memory()
        view.kernel_valid_lengths = torch.full(
            (batch_size,), kernel_width, dtype=torch.int32
        ).pin_memory()
        group_attn_inputs[tag] = view
        pool_offset += pool_numel
        kernel_offset += kernel_numel
    inputs.cache_group_attn_inputs = group_attn_inputs
    return inputs


def _build_prefill_inputs(
    tags: list[str], values: dict[str, int], seq_len: int = 4
) -> PyModelInputs:
    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = True
    attention_inputs.is_target_verify = False
    attention_inputs.input_lengths = torch.tensor(
        [seq_len], dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32).pin_memory()
    attention_inputs.cu_seqlens = torch.tensor(
        [0, seq_len], dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = attention_inputs.cu_seqlens_device.clone()
    attention_inputs.context_total_kv_length = seq_len
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=1,
        token_count=seq_len,
        block_count=1,
    )


def _build_target_verify_inputs(
    tags: list[str],
    values: dict[str, int],
    batch_size: int = 1,
    query_len: int = 5,
    prefix_len: int = 11,
    is_prefill: bool = True,
) -> PyModelInputs:
    token_count = batch_size * query_len

    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = is_prefill
    attention_inputs.is_target_verify = True
    attention_inputs.input_lengths = torch.full(
        (batch_size,), query_len, dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.full(
        (batch_size,), prefix_len, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
    attention_inputs.sequence_lengths_plus_1_device = (
        attention_inputs.prefix_lengths.cuda() + 1
    )

    cu_q = torch.arange(0, token_count + 1, query_len, dtype=torch.int32).pin_memory()
    attention_inputs.cu_seqlens = cu_q
    attention_inputs.cu_seqlens_device = cu_q.cuda()
    attention_inputs.cu_kv_seqlens_device = torch.arange(
        0,
        batch_size * (query_len + prefix_len) + 1,
        query_len + prefix_len,
        dtype=torch.int32,
        device="cuda",
    )
    attention_inputs.decode_cu_seqlens = torch.arange(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attention_inputs.decode_cu_seqlens_device = (
        attention_inputs.decode_cu_seqlens.cuda()
    )

    attention_inputs.context_total_kv_length = batch_size * (query_len + prefix_len)

    block_count = (prefix_len + query_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=batch_size,
        token_count=token_count,
        block_count=block_count,
    )


class TestCudaGraphTaggedCache(unittest.TestCase):
    def test_cache_free_capture_owns_four_fixed_empty_tables(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            CacheFreeModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            [],
        )
        inputs = _build_cache_free_decode_inputs(batch_size=2)
        self.assertTrue(runner.canRun(inputs))
        self.assertEqual(
            runner.packed_cache_descriptors(),
            [
                [1, 4, 0, 0],
                [1, 4, 0, 1],
                [1, 4, 0, 0],
                [1, 4, 0, 1],
            ],
        )
        runner.update_kernel_tables(inputs)
        self.assertEqual(
            runner.packed_cache_descriptors(),
            [
                [1, 4, 0, 0],
                [1, 4, 0, 1],
                [1, 4, 0, 0],
                [1, 4, 0, 1],
            ],
        )

    def test_heterogeneous_packed_replay_keeps_four_backings_and_views_stable(
        self,
    ) -> None:
        widths = {"aux": (3, 6), "full": (1, 2)}
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            GROUP_TAGS,
            False,
            1,
            [3, 1],
            [6, 2],
        )
        initial = _build_heterogeneous_decode_inputs(
            GROUP_TAGS, widths, {"aux": 3, "full": 5}, batch_size=2
        )
        self.assertTrue(runner.canRun(initial))
        runner.update_kernel_tables(initial)
        pointers_before = runner.packed_cache_pointers()
        self.assertEqual(runner.packed_cache_numel(), [16, 16, 32, 32])

        smaller = _build_heterogeneous_decode_inputs(
            GROUP_TAGS, widths, {"aux": 7, "full": 11}, batch_size=1
        )
        smaller.cache_group_attn_inputs["aux"].kernel_valid_lengths.fill_(2)
        smaller.cache_group_attn_inputs["aux"].kv_cache_kernel_block_id[:, 2:].fill_(99)
        smaller.cache_group_attn_inputs["aux"].kv_cache_kernel_block_id_device.copy_(
            smaller.cache_group_attn_inputs["aux"].kv_cache_kernel_block_id
        )
        self.assertTrue(runner.canRun(smaller))
        runner.update_kernel_tables(smaller)
        torch.cuda.synchronize()

        self.assertEqual(runner.packed_cache_pointers(), pointers_before)
        captured = [table.cpu() for table in runner.captured_kernel_tables()]
        self.assertEqual(captured[0][0].tolist(), [7, 7, -1, -1, -1, -1])
        self.assertTrue(torch.equal(captured[0][1:], torch.zeros_like(captured[0][1:])))
        self.assertEqual(captured[1][0].tolist(), [11, 11])
        self.assertTrue(torch.equal(captured[1][1:], torch.zeros_like(captured[1][1:])))

    def _assert_replay_signature(
        self, runner: CudaGraphRunner, inputs: PyModelInputs, expected: int
    ) -> None:
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected_output = torch.full_like(output.hidden_states, expected)
        torch.testing.assert_close(output.hidden_states, expected_output)

    def test_decode_tag_validation_and_replay_updates(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
        )

        self._assert_replay_signature(
            runner,
            _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1}),
            18,
        )
        self._assert_replay_signature(
            runner,
            _build_decode_inputs(GROUP_TAGS, {"full": 5, "aux": 3}),
            53,
        )

        self.assertFalse(runner.canRun(_build_decode_inputs(["full"], {"full": 2})))
        self.assertFalse(
            runner.canRun(
                _build_decode_inputs(
                    ["full", "aux", "extra"],
                    {"full": 2, "aux": 1, "extra": 9},
                )
            )
        )
        wrong_tag = _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1})
        remapped = dict(wrong_tag.cache_group_attn_inputs)
        remapped["unexpected"] = remapped.pop("full")
        wrong_tag.cache_group_attn_inputs = remapped
        self.assertFalse(runner.canRun(wrong_tag))

        oversized = _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1})
        oversized.cache_group_attn_inputs["aux"].kv_cache_kernel_block_id = torch.zeros(
            (2, 2), dtype=torch.int32
        ).pin_memory()
        oversized.cache_group_attn_inputs["aux"].kv_cache_kernel_block_id_device = (
            oversized.cache_group_attn_inputs["aux"].kv_cache_kernel_block_id.cuda()
        )
        self.assertFalse(runner.canRun(oversized))

    def test_mixed_full_linear_only_prepares_full_fmha_tag(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            MixedFullLinearModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            MIXED_ATTENTION_TAGS,
        )

        self._assert_replay_signature(
            runner,
            _build_decode_inputs(MIXED_ATTENTION_TAGS, {"full": 5, "linear": 3}),
            53,
        )

    def test_focused_kernel_update_refreshes_heterogeneous_valid_tails(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            GROUP_TAGS,
        )
        inputs = _build_decode_inputs(GROUP_TAGS, {"full": 5, "aux": 3}, batch_size=2)
        inputs.cache_group_attn_inputs["aux"].kernel_valid_lengths.copy_(
            torch.tensor([1, 0], dtype=torch.int32)
        )
        inputs.cache_group_attn_inputs["full"].kernel_valid_lengths.copy_(
            torch.tensor([0, 1], dtype=torch.int32)
        )
        self.assertTrue(runner.canRun(inputs))

        captured_valid_lengths = runner.update_kernel_tables(inputs)

        self.assertEqual(captured_valid_lengths[0].tolist(), [1, 0, 0, 0])
        self.assertEqual(captured_valid_lengths[1].tolist(), [0, 1, 0, 0])

    def test_prefill_tagged_capture_and_replay_updates(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            TaggedBlockTableModel(),
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
        )

        self._assert_replay_signature(
            runner,
            _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}),
            33,
        )
        self._assert_replay_signature(
            runner,
            _build_prefill_inputs(GROUP_TAGS, {"full": 4, "aux": 3}),
            52,
        )

    def test_duplicate_capture_tag_is_rejected(self) -> None:
        runner = CudaGraphRunner()
        with self.assertRaisesRegex(RuntimeError, "duplicate CUDA graph test tag=full"):
            runner.init_decode(
                TaggedBlockTableModel(),
                HIDDEN_SIZE,
                TOKENS_PER_BLOCK,
                TOKENS_PER_BLOCK,
                TOKENS_PER_BLOCK,
                [1],
                ["full", "full"],
            )

    def test_empty_capture_tag_is_rejected(self) -> None:
        runner = CudaGraphRunner()
        with self.assertRaisesRegex(RuntimeError, "must not be empty"):
            runner.init_decode(
                TaggedBlockTableModel(),
                HIDDEN_SIZE,
                TOKENS_PER_BLOCK,
                TOKENS_PER_BLOCK,
                TOKENS_PER_BLOCK,
                [1],
                ["full", ""],
            )

    def test_capture_tag_declaration_order_does_not_change_replay(self) -> None:
        # Capture buffers are addressed by an adapter-local group_ordinal taken
        # from the sorted tag order, so declaring the same tags in the reverse
        # order must replay to exactly the same per-tag values.
        for tags in (GROUP_TAGS, list(reversed(GROUP_TAGS))):
            with self.subTest(tags=tags):
                runner = CudaGraphRunner()
                runner.init_decode(
                    TaggedBlockTableModel(),
                    HIDDEN_SIZE,
                    TOKENS_PER_BLOCK,
                    TOKENS_PER_BLOCK,
                    TOKENS_PER_BLOCK,
                    [2],
                    tags,
                )
                self._assert_replay_signature(
                    runner,
                    _build_decode_inputs(GROUP_TAGS, {"full": 5, "aux": 3}),
                    53,
                )

    def test_target_verify_validates_exact_tag_set(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [2],
            GROUP_TAGS,
            True,
        )

        valid = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        self.assertTrue(runner.canRun(valid))

        missing = _build_target_verify_inputs(
            ["full"], {"full": 2}, batch_size=2, query_len=1, prefix_len=1
        )
        self.assertFalse(runner.canRun(missing))

        wrong = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        remapped = dict(wrong.cache_group_attn_inputs)
        remapped["unexpected"] = remapped.pop("full")
        wrong.cache_group_attn_inputs = remapped
        self.assertFalse(runner.canRun(wrong))

        non_prefill = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
            is_prefill=False,
        )
        self.assertFalse(runner.canRun(non_prefill))

    def test_target_verify_clears_rounded_batch_sequence_lengths(self) -> None:
        query_len = 5
        prefix_len = 11
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedSequenceLengthModel(),
            HIDDEN_SIZE,
            64,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            GROUP_TAGS,
            True,
            query_len,
        )

        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                inputs = _build_target_verify_inputs(
                    GROUP_TAGS,
                    {"full": 2, "aux": 1},
                    batch_size=batch_size,
                    query_len=query_len,
                    prefix_len=prefix_len,
                )
                self.assertTrue(runner.canRun(inputs))
                self.assertEqual(runner.getCurrentRealGraphSize(), 4)

                output = runner.forward(inputs)
                torch.cuda.synchronize()
                total_query_length = batch_size * query_len
                total_kv_length = batch_size * (query_len + prefix_len)
                expected_signature = torch.tensor(
                    [
                        total_query_length,
                        total_kv_length,
                        total_query_length,
                        batch_size * prefix_len,
                    ],
                    dtype=output.hidden_states.dtype,
                    device=output.hidden_states.device,
                )
                torch.testing.assert_close(
                    output.hidden_states,
                    expected_signature.unsqueeze(0).expand_as(output.hidden_states),
                )


if __name__ == "__main__":
    unittest.main()
