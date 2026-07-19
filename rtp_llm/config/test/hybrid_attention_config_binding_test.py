import ast
import unittest
from pathlib import Path

from rtp_llm.ops import HybridAttentionConfig, HybridAttentionType


class HybridAttentionConfigBindingTest(unittest.TestCase):
    def test_default_constructor(self) -> None:
        config = HybridAttentionConfig()

        self.assertFalse(config.enable_hybrid_attention)
        self.assertFalse(config.enable_independent_kv_cache_pools)
        self.assertEqual(config.hybrid_attention_types, [])

    def test_three_argument_constructor(self) -> None:
        config = HybridAttentionConfig(
            True, True, [HybridAttentionType.LINEAR]
        )

        self.assertTrue(config.enable_hybrid_attention)
        self.assertTrue(config.enable_independent_kv_cache_pools)
        self.assertEqual(
            config.hybrid_attention_types, [HybridAttentionType.LINEAR]
        )

    def test_two_argument_constructor_is_not_exposed(self) -> None:
        with self.assertRaises(TypeError):
            HybridAttentionConfig(True, [HybridAttentionType.LINEAR])

    def test_stub_exposes_only_default_and_three_required_arguments(self) -> None:
        stub_path = Path(__file__).parents[2] / "ops/libth_transformer_config.pyi"
        lines = stub_path.read_text(encoding="utf-8").splitlines()
        class_start = lines.index("class HybridAttentionConfig:")
        class_end = next(
            i
            for i in range(class_start + 1, len(lines))
            if lines[i].startswith("class ")
        )
        module = ast.parse("\n".join(lines[class_start:class_end]))
        class_node = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "HybridAttentionConfig"
        )
        constructors = [
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ]

        self.assertEqual(
            [len(node.args.args) - 1 for node in constructors], [0, 3]
        )
        self.assertTrue(all(not node.args.defaults for node in constructors))


if __name__ == "__main__":
    unittest.main()
