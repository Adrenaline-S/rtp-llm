#include <gtest/gtest.h>

#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {
namespace {

CacheConfig makeKimiHybridConfig() {
    ModelConfig config;
    config.num_layers                                      = 4;
    config.attn_config.head_num                            = 4;
    config.attn_config.kv_head_num                         = 2;
    config.attn_config.size_per_head                       = 32;
    config.attn_config.tokens_per_block                    = 8;
    config.hybrid_attention_config.enable_hybrid_attention = true;
    config.hybrid_attention_config.hybrid_attention_types  = {
        HybridAttentionType::LINEAR, HybridAttentionType::NONE, HybridAttentionType::LINEAR, HybridAttentionType::NONE};
    config.linear_attention_config.linear_conv_kernel_dim = 4;
    config.linear_attention_config.linear_key_head_dim    = 16;
    config.linear_attention_config.linear_value_head_dim  = 16;
    config.linear_attention_config.linear_num_key_heads   = 2;
    config.linear_attention_config.linear_num_value_heads = 2;
    setHybridAttentionKvCacheSpecs(config);

    ParallelismConfig parallelism;
    return CacheConfigCreator::createBasicConfig(config, parallelism, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
}

CacheConfig makeDeepSeekV4HybridPoolConfig() {
    ModelConfig config;
    config.num_layers                                                = 2;
    config.attn_config.head_num                                      = 128;
    config.attn_config.kv_head_num                                   = 1;
    config.attn_config.size_per_head                                 = 512;
    config.attn_config.indexer_head_dim                              = 128;
    config.attn_config.tokens_per_block                              = 128;
    config.hybrid_attention_config.enable_hybrid_attention           = true;
    config.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    setDsv4KvCacheSpecs(config, {128, 4});

    ParallelismConfig parallelism;
    return CacheConfigCreator::createBasicConfig(config, parallelism, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
}

TEST(CacheSemanticSnapshotTest, SingleMhaMatchesGolden) {
    const auto config = makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                                 /*block_num=*/7,
                                                 /*tokens_per_block=*/8,
                                                 DataType::TYPE_FP16,
                                                 /*local_head_num_kv=*/2,
                                                 /*size_per_head=*/4);

    const CacheSemanticSnapshot expected = {{"default",
                                             KVCacheSpecType::MultiHeadAttention,
                                             CacheGroupType::FULL,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             0,
                                             true,
                                             CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                             CpBlockSliceMode::NONE,
                                             {0, 1},
                                             7,
                                             8,
                                             8,
                                             512,
                                             256,
                                             0,
                                             1,
                                             makeExpectedSpecSemanticSnapshot(KVCacheSpecType::MultiHeadAttention,
                                                            DataType::TYPE_FP16,
                                                            8,
                                                            128,
                                                            64,
                                                            64,
                                                            256,
                                                            128,
                                                            128,
                                                            256,
                                                            128,
                                                            128,
                                                            0,
                                                            0,
                                                            0)}};

    EXPECT_EQ(snapshotCacheConfig(config), expected);
}

TEST(CacheSemanticSnapshotTest, KimiHybridMatchesGolden) {
    const CacheSemanticSnapshot expected = {{"full",
                                             KVCacheSpecType::MultiHeadAttention,
                                             CacheGroupType::FULL,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             0,
                                             true,
                                             CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                             CpBlockSliceMode::NONE,
                                             {1, 3},
                                             0,
                                             8,
                                             8,
                                             4096,
                                             2048,
                                             0,
                                             2,
                                             makeExpectedSpecSemanticSnapshot(KVCacheSpecType::MultiHeadAttention,
                                                            DataType::TYPE_FP16,
                                                            8,
                                                            1024,
                                                            512,
                                                            512,
                                                            2048,
                                                            1024,
                                                            1024,
                                                            2048,
                                                            1024,
                                                            1024,
                                                            0,
                                                            0,
                                                            0)},
                                            {"linear",
                                             KVCacheSpecType::LinearAttention,
                                             CacheGroupType::LINEAR,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             1,
                                             true,
                                             CpBlockMappingMode::NONE,
                                             CpBlockSliceMode::NONE,
                                             {0, 2},
                                             0,
                                             8,
                                             8,
                                             3200,
                                             1600,
                                             0,
                                             2,
                                             makeExpectedSpecSemanticSnapshot(KVCacheSpecType::LinearAttention,
                                                            DataType::TYPE_FP16,
                                                            8,
                                                            800,
                                                            512,
                                                            288,
                                                            1600,
                                                            1024,
                                                            576,
                                                            1600,
                                                            1024,
                                                            576,
                                                            0,
                                                            0,
                                                            0)}};

    const auto config = makeKimiHybridConfig();
    EXPECT_TRUE(config.use_independent_block_pools);
    EXPECT_EQ(snapshotCacheConfig(config), expected);
}

TEST(CacheSemanticSnapshotTest, DeepSeekV4HybridPoolMatchesGolden) {
    const CacheGroupSemanticSnapshot full_csa      = {"csa_kv",
                                                 KVCacheSpecType::OpaqueKV,
                                                 CacheGroupType::FULL,
                                                 true,
                                                 CacheEvictPolicy::CHAIN,
                                                 true,
                                                 0,
                                                 0,
                                                 true,
                                                 CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                                 CpBlockSliceMode::NONE,
                                                 {1},
                                                 0,
                                                 128,
                                                 128,
                                                 32768,
                                                 32768,
                                                 0,
                                                 1,
                                                 makeExpectedSpecSemanticSnapshot(KVCacheSpecType::OpaqueKV,
                                                                DataType::TYPE_UINT8,
                                                                128,
                                                                32768,
                                                                32768,
                                                                0,
                                                                32768,
                                                                32768,
                                                                0,
                                                                32768,
                                                                32768,
                                                                0,
                                                                0,
                                                                0,
                                                                0)};
    const CacheGroupSemanticSnapshot state_csa     = {"csa_state",
                                                  KVCacheSpecType::OpaqueState,
                                                  CacheGroupType::SWA,
                                                  true,
                                                  CacheEvictPolicy::INDEPENDENT,
                                                  true,
                                                  0,
                                                  2,
                                                  true,
                                                  CpBlockMappingMode::COMPACT_LAST_RANK,
                                                  CpBlockSliceMode::PAYLOAD_BYTES,
                                                  {1},
                                                  0,
                                                  128,
                                                  128,
                                                  65536,
                                                  65536,
                                                  0,
                                                  1,
                                                  makeExpectedSpecSemanticSnapshot(KVCacheSpecType::OpaqueState,
                                                                 DataType::TYPE_FP32,
                                                                 128,
                                                                 16384,
                                                                 16384,
                                                                 0,
                                                                 65536,
                                                                 65536,
                                                                 0,
                                                                 65536,
                                                                 65536,
                                                                 0,
                                                                 0,
                                                                 0,
                                                                 0)};
    const CacheGroupSemanticSnapshot full_hca      = {"hca_kv",
                                                 KVCacheSpecType::OpaqueKV,
                                                 CacheGroupType::FULL,
                                                 true,
                                                 CacheEvictPolicy::CHAIN,
                                                 true,
                                                 0,
                                                 0,
                                                 true,
                                                 CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                                 CpBlockSliceMode::NONE,
                                                 {0},
                                                 0,
                                                 128,
                                                 128,
                                                 1024,
                                                 1024,
                                                 0,
                                                 1,
                                                 makeExpectedSpecSemanticSnapshot(KVCacheSpecType::OpaqueKV,
                                                                DataType::TYPE_UINT8,
                                                                128,
                                                                1024,
                                                                1024,
                                                                0,
                                                                1024,
                                                                1024,
                                                                0,
                                                                1024,
                                                                1024,
                                                                0,
                                                                0,
                                                                0,
                                                                0)};
    const CacheGroupSemanticSnapshot state_hca     = {"hca_state",
                                                  KVCacheSpecType::OpaqueState,
                                                  CacheGroupType::SWA,
                                                  false,
                                                  CacheEvictPolicy::INDEPENDENT,
                                                  true,
                                                  256,
                                                  1,
                                                  false,
                                                  CpBlockMappingMode::COMPACT_LAST_RANK,
                                                  CpBlockSliceMode::PAYLOAD_BYTES,
                                                  {0},
                                                  0,
                                                  128,
                                                  128,
                                                  524288,
                                                  524288,
                                                  0,
                                                  1,
                                                  makeExpectedSpecSemanticSnapshot(KVCacheSpecType::OpaqueState,
                                                                 DataType::TYPE_FP32,
                                                                 128,
                                                                 131072,
                                                                 131072,
                                                                 0,
                                                                 524288,
                                                                 524288,
                                                                 0,
                                                                 524288,
                                                                 524288,
                                                                 0,
                                                                 0,
                                                                 0,
                                                                 0)};
    const CacheGroupSemanticSnapshot full_indexer  = {"indexer_kv",
                                                     KVCacheSpecType::OpaqueKV,
                                                     CacheGroupType::FULL,
                                                     true,
                                                     CacheEvictPolicy::CHAIN,
                                                     true,
                                                     0,
                                                     0,
                                                     true,
                                                     CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                                     CpBlockSliceMode::NONE,
                                                     {1},
                                                     0,
                                                     128,
                                                     128,
                                                     8192,
                                                     8192,
                                                     0,
                                                     1,
                                                     makeExpectedSpecSemanticSnapshot(KVCacheSpecType::OpaqueKV,
                                                                    DataType::TYPE_UINT8,
                                                                    128,
                                                                    8192,
                                                                    8192,
                                                                    0,
                                                                    8192,
                                                                    8192,
                                                                    0,
                                                                    8192,
                                                                    8192,
                                                                    0,
                                                                    0,
                                                                    0,
                                                                    0)};
    const CacheGroupSemanticSnapshot state_indexer = {"indexer_state",
                                                      KVCacheSpecType::OpaqueState,
                                                      CacheGroupType::SWA,
                                                      true,
                                                      CacheEvictPolicy::INDEPENDENT,
                                                      true,
                                                      0,
                                                      2,
                                                      true,
                                                      CpBlockMappingMode::COMPACT_LAST_RANK,
                                                      CpBlockSliceMode::PAYLOAD_BYTES,
                                                      {1},
                                                      0,
                                                      128,
                                                      128,
                                                      16384,
                                                      16384,
                                                      0,
                                                      1,
                                                      makeExpectedSpecSemanticSnapshot(KVCacheSpecType::OpaqueState,
                                                                     DataType::TYPE_FP32,
                                                                     128,
                                                                     4096,
                                                                     4096,
                                                                     0,
                                                                     16384,
                                                                     16384,
                                                                     0,
                                                                     16384,
                                                                     16384,
                                                                     0,
                                                                     0,
                                                                     0,
                                                                     0)};
    const CacheGroupSemanticSnapshot state_swa     = {"swa_kv",
                                                  KVCacheSpecType::OpaqueState,
                                                  CacheGroupType::SWA,
                                                  true,
                                                  CacheEvictPolicy::INDEPENDENT,
                                                  true,
                                                  0,
                                                  2,
                                                  true,
                                                  CpBlockMappingMode::COMPACT_LAST_RANK,
                                                  CpBlockSliceMode::EQUAL_BYTES,
                                                  {0, 1},
                                                  0,
                                                  128,
                                                  128,
                                                  262144,
                                                  131072,
                                                  0,
                                                  1,
                                                  makeExpectedSpecSemanticSnapshot(KVCacheSpecType::OpaqueState,
                                                                 DataType::TYPE_UINT8,
                                                                 128,
                                                                 131072,
                                                                 131072,
                                                                 0,
                                                                 131072,
                                                                 131072,
                                                                 0,
                                                                 131072,
                                                                 131072,
                                                                 0,
                                                                 0,
                                                                 0,
                                                                 0)};
    const CacheSemanticSnapshot      expected      = {
        full_csa, state_csa, full_hca, state_hca, full_indexer, state_indexer, state_swa};

    EXPECT_EQ(snapshotCacheConfig(makeDeepSeekV4HybridPoolConfig()), expected);
}

TEST(CacheSemanticSnapshotTest, MlaFp8PreservesPhysicalStrideAcrossKernelSubdivision) {
    CacheConfig base;
    base.dtype                     = DataType::TYPE_FP8_E4M3;
    base.layer_num                 = 1;
    base.layer_all_num             = 1;
    base.block_num                 = 5;
    base.seq_size_per_block        = 8;
    base.kernel_seq_size_per_block = 4;
    auto spec = makeResolvedMlaSpec(DataType::TYPE_FP8_E4M3, /*kv_lora_rank=*/128, /*rope_head_dim=*/64, 8, "mla");
    const auto config = buildTestCacheConfigFromGroupedSpecs(
        std::move(base), {spec}, {{0}}, {CacheGroupType::FULL}, {"mla"});

    const CacheSemanticSnapshot expected = {{"mla",
                                             KVCacheSpecType::MultiHeadLatentAttention,
                                             CacheGroupType::FULL,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             0,
                                             true,
                                             CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                             CpBlockSliceMode::NONE,
                                             {0},
                                             5,
                                             8,
                                             4,
                                             2080,
                                             2080,
                                             0,
                                             1,
                                             makeExpectedSpecSemanticSnapshot(KVCacheSpecType::MultiHeadLatentAttention,
                                                            DataType::TYPE_FP8_E4M3,
                                                            8,
                                                            2080,
                                                            1024,
                                                            512,
                                                            2080,
                                                            1024,
                                                            512,
                                                            2080,
                                                            1024,
                                                            512,
                                                            0,
                                                            0,
                                                            0)}};
    EXPECT_EQ(snapshotCacheConfig(config), expected);
    EXPECT_EQ(config.group("mla").layout.kernelBlocksPerPoolBlock(), 2u);
}

TEST(CacheSemanticSnapshotTest, ExplicitBlocksAndReversedDeclarationsAreValueEquivalent) {
    auto make_config = [](bool reversed) {
        CacheConfig base;
        base.dtype                     = DataType::TYPE_FP16;
        base.layer_num                 = 2;
        base.layer_all_num             = 2;
        base.block_num                 = 11;
        base.seq_size_per_block        = 8;
        base.kernel_seq_size_per_block = 4;
        auto full = makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 2, 8, "full");
        auto swa  = makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 2, 8, "swa");
        auto full_policy               = defaultCacheGroupPolicy(CacheGroupType::FULL);
        full_policy.explicit_block_num = 3;
        auto swa_policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
        if (reversed) {
            return buildTestCacheConfigFromGroupedSpecs(std::move(base),
                                                        {swa, full},
                                                        {{1}, {0}},
                                                        {CacheGroupType::SWA, CacheGroupType::FULL},
                                                        {"swa", "full"},
                                                        {swa_policy, full_policy});
        }
        return buildTestCacheConfigFromGroupedSpecs(std::move(base),
                                                    {full, swa},
                                                    {{0}, {1}},
                                                    {CacheGroupType::FULL, CacheGroupType::SWA},
                                                    {"full", "swa"},
                                                    {full_policy, swa_policy});
    };

    const auto declared = make_config(false);
    const auto reversed = make_config(true);
    const CacheSemanticSnapshot expected = {{"full",
                                             KVCacheSpecType::MultiHeadAttention,
                                             CacheGroupType::FULL,
                                             true,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             3,
                                             0,
                                             true,
                                             CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                             CpBlockSliceMode::NONE,
                                             {0},
                                             11,
                                             8,
                                             4,
                                             64,
                                             64,
                                             0,
                                             1,
                                             makeExpectedSpecSemanticSnapshot(KVCacheSpecType::MultiHeadAttention,
                                                            DataType::TYPE_FP16,
                                                            8,
                                                            32,
                                                            16,
                                                            16,
                                                            64,
                                                            32,
                                                            32,
                                                            64,
                                                            32,
                                                            32,
                                                            0,
                                                            0,
                                                            0)},
                                            {"swa",
                                             KVCacheSpecType::MultiHeadAttention,
                                             CacheGroupType::SWA,
                                             false,
                                             CacheEvictPolicy::CHAIN,
                                             true,
                                             0,
                                             2,
                                             true,
                                             CpBlockMappingMode::COMPACT_LAST_RANK,
                                             CpBlockSliceMode::NONE,
                                             {1},
                                             11,
                                             8,
                                             8,
                                             64,
                                             64,
                                             0,
                                             1,
                                             makeExpectedSpecSemanticSnapshot(KVCacheSpecType::MultiHeadAttention,
                                                            DataType::TYPE_FP16,
                                                            8,
                                                            32,
                                                            16,
                                                            16,
                                                            64,
                                                            32,
                                                            32,
                                                            64,
                                                            32,
                                                            32,
                                                            0,
                                                            0,
                                                            0)}};
    EXPECT_EQ(snapshotCacheConfig(declared), expected);
    EXPECT_EQ(snapshotCacheConfig(reversed), expected);
    EXPECT_EQ(declared.explicitIndependentBlocks("full"), 3u);
    EXPECT_EQ(declared.groupTagsForLayer(0), std::vector<std::string>{"full"});
    EXPECT_EQ(declared.groupTagsForLayer(1), std::vector<std::string>{"swa"});
}

TEST(CacheSemanticSnapshotTest, MtpMergePreservesMainAndModuleRecords) {
    auto main = makeSimpleMhaCacheConfig(
        /*layer_num=*/2, /*block_num=*/7, /*tokens_per_block=*/8, DataType::TYPE_FP16, 1, 2);
    main.group_layer_num = 2;
    auto propose = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/7, /*tokens_per_block=*/8, DataType::TYPE_FP16, 1, 2);
    const auto module = TestCacheConfigBuilder::mergeMTPModule(main, propose, /*module_index=*/0, /*main_layer_num=*/2);

    ASSERT_NE(module, nullptr);
    const CacheSemanticSnapshot expected_main = {{"default",
                                                  KVCacheSpecType::MultiHeadAttention,
                                                  CacheGroupType::FULL,
                                                  true,
                                                  CacheEvictPolicy::CHAIN,
                                                  true,
                                                  0,
                                                  0,
                                                  true,
                                                  CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                                  CpBlockSliceMode::NONE,
                                                  {0, 1, 2},
                                                  7,
                                                  8,
                                                  8,
                                                  192,
                                                  64,
                                                  0,
                                                  1,
                                                  makeExpectedSpecSemanticSnapshot(KVCacheSpecType::MultiHeadAttention,
                                                                 DataType::TYPE_FP16,
                                                                 8,
                                                                 32,
                                                                 16,
                                                                 16,
                                                                 64,
                                                                 32,
                                                                 32,
                                                                 64,
                                                                 32,
                                                                 32,
                                                                 0,
                                                                 0,
                                                                 0)}};
    const CacheSemanticSnapshot expected_module = {{"default",
                                                    KVCacheSpecType::MultiHeadAttention,
                                                    CacheGroupType::FULL,
                                                    true,
                                                    CacheEvictPolicy::CHAIN,
                                                    true,
                                                    0,
                                                    0,
                                                    true,
                                                    CpBlockMappingMode::BLOCK_ROUND_ROBIN,
                                                    CpBlockSliceMode::NONE,
                                                    {0},
                                                    7,
                                                    8,
                                                    8,
                                                    64,
                                                    64,
                                                    0,
                                                    1,
                                                    makeExpectedSpecSemanticSnapshot(KVCacheSpecType::MultiHeadAttention,
                                                                   DataType::TYPE_FP16,
                                                                   8,
                                                                   32,
                                                                   16,
                                                                   16,
                                                                   64,
                                                                   32,
                                                                   32,
                                                                   64,
                                                                   32,
                                                                   32,
                                                                   0,
                                                                   0,
                                                                   0)}};
    EXPECT_EQ(snapshotCacheConfig(main), expected_main);
    EXPECT_EQ(snapshotCacheConfig(*module), expected_module);
    ASSERT_EQ(main.group("default").layer_ids, (std::vector<int>{0, 1, 2}));
    ASSERT_EQ(module->group("default").layer_ids, (std::vector<int>{0}));
    EXPECT_EQ(main.group("default").layout.kv_block_stride_bytes,
              module->group("default").layout.kv_block_stride_bytes);
    EXPECT_EQ(main.group("default").policy.group_type, module->group("default").policy.group_type);
    EXPECT_EQ(main.groupTagsForLayer(2), std::vector<std::string>{"default"});
}

TEST(CacheSemanticSnapshotTest, FinalizingCopyDoesNotMutatePublishedMtpDescendants) {
    auto grandchild = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/5, /*tokens_per_block=*/8, DataType::TYPE_FP16, 1, 2);
    auto child = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/5, /*tokens_per_block=*/8, DataType::TYPE_FP16, 1, 2);
    child = TestCacheConfigBuilder::withMTPModules(std::move(child), {std::move(grandchild)});
    auto source = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/5, /*tokens_per_block=*/8, DataType::TYPE_FP16, 1, 2);
    source = TestCacheConfigBuilder::withMTPModules(std::move(source), {std::move(child)});

    const auto original_child_address      = &source.mtpModule(0);
    const auto original_grandchild_address = &source.mtpModule(0).mtpModule(0);
    const auto finalized =
        CacheConfigCreator::finalizeBlockNums(source, /*global_block_num=*/19, RuntimeConfig{});

    ASSERT_EQ(source.mtpModuleCount(), 1u);
    ASSERT_EQ(source.mtpModule(0).mtpModuleCount(), 1u);
    EXPECT_EQ(source.block_num, 5u);
    EXPECT_EQ(source.mtpModule(0).block_num, 5u);
    EXPECT_EQ(source.mtpModule(0).mtpModule(0).block_num, 5u);
    EXPECT_EQ(source.mtpModule(0).group("default").layout.block_num, 5u);
    EXPECT_EQ(source.mtpModule(0).mtpModule(0).group("default").layout.block_num, 5u);

    ASSERT_EQ(finalized.mtpModuleCount(), 1u);
    ASSERT_EQ(finalized.mtpModule(0).mtpModuleCount(), 1u);
    EXPECT_EQ(finalized.block_num, 19u);
    EXPECT_EQ(finalized.mtpModule(0).block_num, 19u);
    EXPECT_EQ(finalized.mtpModule(0).mtpModule(0).block_num, 19u);
    EXPECT_EQ(finalized.mtpModule(0).group("default").layout.block_num, 19u);
    EXPECT_EQ(finalized.mtpModule(0).mtpModule(0).group("default").layout.block_num, 19u);
    EXPECT_NE(&finalized.mtpModule(0), original_child_address);
    EXPECT_NE(&finalized.mtpModule(0).mtpModule(0), original_grandchild_address);
}

}  // namespace
}  // namespace rtp_llm::test
