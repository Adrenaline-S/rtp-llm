from unittest import TestCase, main

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v4 import (
    _build_dsv4_kv_cache_specs,
    _refresh_dsv4_kv_cache_specs,
)
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3NextBase
from rtp_llm.ops import (
    DSV4KVSpec,
    DSV4StateSpec,
    DataType,
    HybridAttentionType,
    KVCacheSpecType,
    KvCacheDataType,
    LinearKVCacheSpec,
    MHAKVCacheSpec,
)


class KVCacheSpecsTest(TestCase):
    def _basic_model_config(self) -> ModelConfig:
        config = ModelConfig()
        config.num_layers = 3
        config.attn_config.kv_head_num = 2
        config.attn_config.size_per_head = 16
        config.attn_config.tokens_per_block = 8
        config.attn_config.use_mla = False
        return config

    def test_base_model_builds_default_mha_spec(self):
        config = self._basic_model_config()

        BaseModel._post_build_model_config(config)

        self.assertEqual(len(config.kv_cache_specs), 1)
        spec = config.kv_cache_specs[0]
        self.assertIsInstance(spec, MHAKVCacheSpec)
        self.assertEqual(spec.tag, "default")
        self.assertEqual(spec.type, KVCacheSpecType.MultiHeadAttention)
        self.assertEqual(spec.layers, [0, 1, 2])
        self.assertEqual(spec.layer_num, 3)
        self.assertEqual(spec.seq_size_per_block, 8)
        self.assertEqual(spec.size_per_head, 16)

    def test_qwen3_next_builds_full_and_linear_specs(self):
        config = self._basic_model_config()
        config.num_layers = 4
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ]
        config.linear_attention_config.linear_key_head_dim = 32
        config.linear_attention_config.linear_value_head_dim = 32
        config.linear_attention_config.linear_conv_kernel_dim = 4

        Qwen3NextBase._post_build_model_config(config)

        self.assertEqual(len(config.kv_cache_specs), 2)
        full_spec, linear_spec = config.kv_cache_specs
        self.assertIsInstance(full_spec, MHAKVCacheSpec)
        self.assertEqual(full_spec.tag, "full")
        self.assertEqual(full_spec.layers, [1, 3])
        self.assertEqual(full_spec.layer_num, 2)
        self.assertEqual(full_spec.seq_size_per_block, 8)
        self.assertEqual(full_spec.size_per_head, 16)

        self.assertIsInstance(linear_spec, LinearKVCacheSpec)
        self.assertEqual(linear_spec.tag, "linear")
        self.assertEqual(linear_spec.type, KVCacheSpecType.LinearAttention)
        self.assertEqual(linear_spec.layers, [0, 2])
        self.assertEqual(linear_spec.layer_num, 2)
        self.assertEqual(linear_spec.head_k_dim, 32)
        self.assertEqual(linear_spec.head_v_dim, 32)
        self.assertEqual(linear_spec.conv_kernel_dim, 4)

    def test_deepseek_v4_builds_expected_spec_declarations(self):
        config = self._basic_model_config()
        config.num_layers = 5
        config.attn_config.size_per_head = 512
        config.attn_config.indexer_head_dim = 128
        config.attn_config.layer_compress_ratios = [4, 128, 4, 128, 0]

        specs = _build_dsv4_kv_cache_specs(config)

        self.assertEqual(
            [spec.tag for spec in specs],
            [
                "csa_kv",
                "hca_kv",
                "indexer_kv",
                "indexer_state",
                "csa_state",
                "hca_state",
                "swa_kv",
            ],
        )
        self.assertTrue(all(spec.dtype == DataType.TYPE_UINT8 for spec in specs[:3]))
        self.assertTrue(all(spec.dtype == DataType.TYPE_FP32 for spec in specs[3:6]))
        self.assertEqual(specs[0].layers, [0, 2])
        self.assertEqual(specs[1].layers, [1, 3])
        self.assertEqual(specs[2].layers, [0, 2])
        self.assertEqual(specs[6].layers, [0, 1, 2, 3, 4])
        self.assertIsInstance(specs[0], DSV4KVSpec)
        self.assertIsInstance(specs[3], DSV4StateSpec)
        self.assertEqual(specs[0].entry_elems, 1024)
        self.assertEqual(specs[2].entry_elems, 256)
        self.assertEqual(specs[3].state_dim, 512)
        self.assertEqual(specs[4].state_dim, 2048)
        self.assertEqual(specs[5].state_dim, 1024)
        self.assertEqual(specs[6].state_dim, 1024)

    def test_deepseek_v4_refresh_updates_fp8_entry_sizes(self):
        config = self._basic_model_config()
        config.num_layers = 2
        config.attn_config.size_per_head = 512
        config.attn_config.indexer_head_dim = 128
        config.attn_config.layer_compress_ratios = [4, 128]
        config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
        config.kv_cache_specs = _build_dsv4_kv_cache_specs(config)

        _refresh_dsv4_kv_cache_specs(config)

        by_tag = {spec.tag: spec for spec in config.kv_cache_specs}
        self.assertEqual(by_tag["csa_kv"].entry_elems, 584)
        self.assertEqual(by_tag["hca_kv"].entry_elems, 584)
        self.assertEqual(by_tag["swa_kv"].state_dim, 584)
        self.assertEqual(by_tag["indexer_kv"].entry_elems, 132)
        self.assertEqual(by_tag["indexer_state"].state_dim, 512)
        self.assertEqual(by_tag["csa_state"].state_dim, 2048)
        self.assertEqual(by_tag["hca_state"].state_dim, 1024)


if __name__ == "__main__":
    main()
