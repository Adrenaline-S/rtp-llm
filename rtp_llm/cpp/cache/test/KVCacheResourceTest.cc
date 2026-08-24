#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {
namespace test {

namespace {

CacheGroup makeResourceGroup(std::string tag, CacheGroupType type) {
    KVCacheSpecPtr spec;
    if (type == CacheGroupType::LINEAR) {
        spec = makeResolvedLinearSpec(
            DataType::TYPE_FP16, 1, 1, 1, 1, 2, 8, DataType::TYPE_FP16, DataType::TYPE_FP16, tag);
    } else {
        spec = makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, 8, tag);
    }

    CacheGroup group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(type);
    group.layer_ids                 = {0};
    group.block_num                 = 16;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = type == CacheGroupType::FULL ? 2 : 8;
    return group;
}

CacheConfig makeResourceConfig(std::vector<CacheGroup> groups, std::vector<CacheLayer> layers) {
    return TestCacheConfigBuilder().setTopology(std::move(groups), std::move(layers)).build();
}

}  // namespace

TEST(KVCacheResourceTest, InitGroups_RespectsGroupTypesAndBlocksPerKvBlock) {
    KVCacheResource resource;
    resource.initGroups(makeTestCacheConfigByTag(
        /*group_num=*/2,
        /*layer_num=*/3,
        /*layer_group_tags=*/{{"group0"}, {"group1"}, {"group0"}},
        /*kernel_blocks_per_kv_block=*/4,
        /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR}));

    ASSERT_EQ(resource.groupNums(), 2);
    ASSERT_EQ(resource.blocksByTag().size(), 2u);
    EXPECT_EQ(&resource.blockBindingForLayer(0, "group0"), &resource.blockBinding("group0"));
    EXPECT_EQ(&resource.blockBindingForLayer(1, "group1"), &resource.blockBinding("group1"));
    EXPECT_EQ(&resource.blockBindingForLayer(2, "group0"), &resource.blockBinding("group0"));

    auto& g0 = resource.mutableBlockBinding("group0");
    auto& g1 = resource.mutableBlockBinding("group1");

    g0.append(poolBlockSnapshotForTest(BlockIndicesType{1}));
    g1.append(poolBlockSnapshotForTest(BlockIndicesType{1}));

    ASSERT_EQ(encodedPoolBlocksForTest(resource.blockBinding("group0")), (BlockIndicesType{1}));
    ASSERT_EQ(encodedPoolBlocksForTest(resource.blockBinding("group1")), (BlockIndicesType{1}));
}

TEST(KVCacheResourceTest, LayerBlocksRejectsMultipleGroupsForOneLayer) {
    KVCacheResource resource;
    resource.initGroups(makeTestCacheConfigByTag(
        /*group_num=*/2,
        /*layer_num=*/1,
        /*layer_group_tags=*/{{"group0", "group1"}},
        /*kernel_blocks_per_kv_block=*/1,
        /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR}));

    EXPECT_THROW(resource.blockBindingForLayer(0, "unknown"), std::exception);
}

TEST(KVCacheResourceTest, TagAccessKeepsSameLayerGroupsIndependent) {
    KVCacheResource resource;
    auto            config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});
    resource.initGroups(config);

    resource.mutableBlockBindingForLayer(0, "full").append(poolBlockSnapshotForTest(BlockIndicesType{1, 2}));
    resource.mutableBlockBindingForLayer(0, "linear").append(poolBlockSnapshotForTest(BlockIndicesType{7}));

    EXPECT_EQ(encodedPoolBlocksForTest(resource.blockBindingForLayer(0, "full")), (BlockIndicesType{1, 2}));
    EXPECT_EQ(encodedPoolBlocksForTest(resource.blockBindingForLayer(0, "linear")), (BlockIndicesType{7}));
    EXPECT_NE(&resource.blockBinding("full"), &resource.blockBinding("linear"));
}

TEST(KVCacheResourceTest, BlocksByTagOwnsOneBlockTablePerTag) {
    auto linear = makeResourceGroup("linear", CacheGroupType::LINEAR);
    auto full   = makeResourceGroup("full", CacheGroupType::FULL);
    auto config = makeResourceConfig({std::move(linear), std::move(full)}, {{"full", "linear"}});

    KVCacheResource resource;
    resource.initGroups(config);
    resource.mutableBlockBinding("full").append(poolBlockSnapshotForTest({11}));
    resource.mutableBlockBinding("linear").append(poolBlockSnapshotForTest({22}));

    const auto& blocks_by_tag = resource.blocksByTag();
    ASSERT_EQ(blocks_by_tag.size(), 2u);
    EXPECT_EQ(blocks_by_tag.begin()->first, "full");
    EXPECT_EQ(encodedPoolBlocksForTest(blocks_by_tag.at("full")), (BlockIndicesType{11}));
    EXPECT_EQ(encodedPoolBlocksForTest(blocks_by_tag.at("linear")), (BlockIndicesType{22}));

    EXPECT_EQ(encodedPoolBlocksForTest(resource.blockBinding("full")), (BlockIndicesType{11}));
    EXPECT_EQ(encodedPoolBlocksForTest(resource.blockBinding("linear")), (BlockIndicesType{22}));
    EXPECT_EQ(encodedPoolBlocksForTest(resource.blockBindingForLayer(0, "full")), (BlockIndicesType{11}));
    EXPECT_EQ(encodedPoolBlocksForTest(resource.blockBindingForLayer(0, "linear")), (BlockIndicesType{22}));
}

TEST(BatchKVCacheResourceTest, CheckValidatesEveryTagAcrossBatches) {
    auto config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});

    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(config);
    batch.mutableBlockBinding(0, "full").assign(poolBlockSnapshotForTest({1, 2}));
    batch.mutableBlockBinding(1, "full").assign(poolBlockSnapshotForTest({3, 4}));
    batch.mutableBlockBinding(0, "linear").assign(poolBlockSnapshotForTest({5}));
    batch.mutableBlockBinding(1, "linear").assign(poolBlockSnapshotForTest({6, 7}));

    EXPECT_THROW(batch.check(), std::exception);
}

TEST(BatchKVCacheResourceTest, CheckAllowsDifferentBlockCountsBetweenTags) {
    auto config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});

    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(config);
    batch.mutableBlockBinding(0, "full").assign(poolBlockSnapshotForTest({1, 2}));
    batch.mutableBlockBinding(1, "full").assign(poolBlockSnapshotForTest({3, 4}));
    batch.mutableBlockBinding(0, "linear").assign(poolBlockSnapshotForTest({5}));
    batch.mutableBlockBinding(1, "linear").assign(poolBlockSnapshotForTest({6}));

    EXPECT_NO_THROW(batch.check());
}

TEST(BatchKVCacheResourceTest, CheckRejectsDifferentTagSetsWithTheSameSize) {
    auto expected_config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});
    auto different_config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("state", CacheGroupType::LINEAR)},
        {{"full", "state"}});

    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(expected_config);
    batch.mutableBlockBinding(0, "full").assign(poolBlockSnapshotForTest({1}));
    batch.mutableBlockBinding(0, "linear").assign(poolBlockSnapshotForTest({2}));

    KVCacheResource different_resource;
    different_resource.initGroups(different_config);
    different_resource.mutableBlockBinding("full").assign(poolBlockSnapshotForTest({3}));
    different_resource.mutableBlockBinding("state").assign(poolBlockSnapshotForTest({4}));
    batch.moveBatchResource(1, std::move(different_resource));

    EXPECT_THROW(batch.check(), std::exception);
}

TEST(KVCacheResourceTest, UnknownTagIsRejectedWithoutMutatingStorage) {
    auto config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});

    KVCacheResource missing;
    missing.initGroups(config);
    EXPECT_THROW(missing.blockBinding("other"), std::exception);
    EXPECT_EQ(missing.blocksByTag().size(), 2u);
    EXPECT_EQ(missing.blocksByTag().count("full"), 1u);
    EXPECT_EQ(missing.blocksByTag().count("linear"), 1u);
}

TEST(KVCacheResourceTest, OwnsOnlyPerTagGroupPositionToPoolBlockBindings) {
    auto config = makeResourceConfig(
        {makeResourceGroup("full", CacheGroupType::FULL), makeResourceGroup("linear", CacheGroupType::LINEAR)},
        {{"full", "linear"}});

    KVCacheResource resource;
    resource.initGroups(config);
    auto& full_binding = resource.mutableBlockBinding("full");
    full_binding.resize(2);
    full_binding.bind(GroupBlockPosition{1}, PoolBlockId{0});

    EXPECT_FALSE(resource.blockBinding("full").lookup(GroupBlockPosition{0}).has_value());
    EXPECT_EQ(resource.blockBindingForLayer(0, "full").lookup(GroupBlockPosition{1}), PoolBlockId{0});
}

TEST(KVCacheResourceTest, TaggedStorageHasOneRecordPerConfigGroupNotPerLayer) {
    auto full      = makeResourceGroup("full", CacheGroupType::FULL);
    full.layer_ids = {0, 1, 2};
    auto config    = makeResourceConfig({std::move(full)}, {{"full"}, {"full"}, {"full"}});

    KVCacheResource resource;
    resource.initGroups(config);

    ASSERT_EQ(resource.blocksByTag().size(), 1u);
    EXPECT_EQ(resource.blocksByTag().count("full"), 1u);
    EXPECT_EQ(&resource.blockBindingForLayer(0, "full"), &resource.blockBinding("full"));
    EXPECT_EQ(&resource.blockBindingForLayer(1, "full"), &resource.blockBinding("full"));
    EXPECT_EQ(&resource.blockBindingForLayer(2, "full"), &resource.blockBinding("full"));
}

TEST(KVCacheResourceTest, InitializationOwnsRoutingAfterConfigLifetime) {
    KVCacheResource resource;
    {
        auto config = makeTestCacheConfigByTag(/*group_num=*/1, /*layer_num=*/1, {{"group0"}});
        resource.initGroups(config);
    }

    EXPECT_EQ(resource.soleGroupTagForLayer(0), "group0");
    resource.mutableBlockBindingForLayer(0, "group0").append(poolBlockSnapshotForTest(BlockIndicesType{3}));
    EXPECT_EQ(encodedPoolBlocksForTest(resource.blockBindingForLayer(0, "group0")), (BlockIndicesType{3}));
}

TEST(PrefillCPConfigTest, ToStringIncludesShardingFields) {
    PrefillCPConfig config;
    config.kv_cache_sharded = true;
    config.prefill_cp_size  = 2;

    const auto text = config.to_string();
    EXPECT_NE(text.find("kv_cache_sharded: 1"), std::string::npos);
    EXPECT_NE(text.find("prefill_cp_size: 2"), std::string::npos);
}

TEST(KVCacheResourceTest, ExplicitTimelineRejectsMismatchedLengthsWithoutChangingState) {
    KVCacheResource resource;
    resource.setCacheKeys(CacheKeysType{10});

    const BlockDependenciesType dependencies = {
        BlockDependency{false, 0, 7},
    };
    EXPECT_THROW(resource.setCacheKeysAndBlockDependencies(CacheKeysType{20, 30}, dependencies), std::exception);

    EXPECT_EQ(resource.cacheKeys(), (CacheKeysType{10}));
    ASSERT_EQ(resource.blockDependencies().size(), 1u);
    EXPECT_FALSE(resource.blockDependencies()[0].has_parent);
    EXPECT_EQ(resource.blockDependencies()[0].ordinal, 0u);
}

TEST(KVCacheResourceTest, ExplicitTimelinePreservesDependencyOrdinalAndParent) {
    KVCacheResource             resource;
    const BlockDependenciesType dependencies = {
        BlockDependency{true, 17, 7},
        BlockDependency{true, 29, 11},
    };

    resource.setCacheKeysAndBlockDependencies(CacheKeysType{100, 200}, dependencies);

    EXPECT_EQ(resource.cacheKeys(), (CacheKeysType{100, 200}));
    ASSERT_EQ(resource.blockDependencies().size(), 2u);
    EXPECT_TRUE(resource.blockDependencies()[0].has_parent);
    EXPECT_EQ(resource.blockDependencies()[0].parent_key, 17);
    EXPECT_EQ(resource.blockDependencies()[0].ordinal, 7u);
    EXPECT_TRUE(resource.blockDependencies()[1].has_parent);
    EXPECT_EQ(resource.blockDependencies()[1].parent_key, 29);
    EXPECT_EQ(resource.blockDependencies()[1].ordinal, 11u);
}

TEST(KVCacheResourceTest, AppendPopAndClearKeepTimelineAligned) {
    KVCacheResource resource;
    resource.setCacheKeys(CacheKeysType{10, 20});
    resource.appendCacheKey(30);

    ASSERT_EQ(resource.blockDependencies().size(), 3u);
    EXPECT_FALSE(resource.blockDependencies()[0].has_parent);
    EXPECT_EQ(resource.blockDependencies()[0].ordinal, 0u);
    EXPECT_TRUE(resource.blockDependencies()[1].has_parent);
    EXPECT_EQ(resource.blockDependencies()[1].parent_key, 10);
    EXPECT_EQ(resource.blockDependencies()[1].ordinal, 1u);
    EXPECT_TRUE(resource.blockDependencies()[2].has_parent);
    EXPECT_EQ(resource.blockDependencies()[2].parent_key, 20);
    EXPECT_EQ(resource.blockDependencies()[2].ordinal, 2u);

    resource.popBackCacheKey();
    EXPECT_EQ(resource.cacheKeys(), (CacheKeysType{10, 20}));
    ASSERT_EQ(resource.blockDependencies().size(), 2u);

    resource.clearCacheKeys();
    EXPECT_TRUE(resource.cacheKeys().empty());
    EXPECT_TRUE(resource.blockDependencies().empty());
}

TEST(BatchKVCacheResourceTest, BasicBatchOperations_WorkAsExpected) {
    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(makeTestCacheConfigByTag(
        /*group_num=*/2,
        /*layer_num=*/3,
        /*layer_group_tags=*/{{"group0"}, {"group1"}, {"group0"}},
        /*kernel_blocks_per_kv_block=*/4,
        /*group_types=*/{CacheGroupType::FULL, CacheGroupType::LINEAR}));

    ASSERT_EQ(batch.batchSize(), 2);
    ASSERT_EQ(batch.groupNums(), 2);

    batch.mutableBlockBinding(/*batch_id=*/0, "group0").assign(poolBlockSnapshotForTest(BlockIndicesType{1, 2}));
    ASSERT_EQ(encodedPoolBlocksForTest(batch.blockBinding(0, "group0")), (BlockIndicesType{1, 2}));

    batch.mutableBlockBinding(/*batch_id=*/0, "group1").assign(poolBlockSnapshotForTest(BlockIndicesType{9, 10}));
    ASSERT_EQ(encodedPoolBlocksForTest(batch.blockBinding(0, "group1")), (BlockIndicesType{9, 10}));

    std::vector<BlockIndicesType> all_g0;
    for (size_t batch_id = 0; batch_id < batch.batchSize(); ++batch_id) {
        all_g0.push_back(encodedPoolBlocksForTest(batch.blockBinding(batch_id, "group0")));
    }
    ASSERT_EQ(all_g0.size(), 2u);
    ASSERT_EQ(all_g0[0], (BlockIndicesType{1, 2}));

    batch.pushBackCacheKey(0, 100);
    batch.pushBackCacheKey(1, 200);
    ASSERT_TRUE(batch.hasCacheKeys());

    batch.popBackAllBatchCacheKeys();
    ASSERT_EQ(batch.cacheKeys(0).size(), 0u);
    ASSERT_EQ(batch.cacheKeys(1).size(), 0u);
    ASSERT_FALSE(batch.hasCacheKeys());

    batch.setLastBlockAligned(true);
    ASSERT_TRUE(batch.lastBlockAligned());
    batch.cacheResource(1).setLastBlockAligned(false);
    ASSERT_FALSE(batch.lastBlockAligned());

    std::vector<KVCacheResource> old_resources;
    batch.resetAndReturnOldResources(/*new_batch_size=*/1, old_resources);
    ASSERT_EQ(old_resources.size(), 2u);
    ASSERT_EQ(batch.batchSize(), 1);

    KVCacheResource moved;
    moved.initGroups(makeTestCacheConfigByTag(/*group_num=*/1,
                                              /*layer_num=*/1,
                                              /*layer_group_tags=*/{{"group0"}},
                                              /*kernel_blocks_per_kv_block=*/2,
                                              /*group_types=*/{CacheGroupType::FULL}));
    moved.mutableBlockBinding("group0").append(poolBlockSnapshotForTest(BlockIndicesType{3}));
    batch.moveBatchResource(0, std::move(moved));
    ASSERT_EQ(batch.cacheResource(0).groupNums(), 1);
    ASSERT_EQ(encodedPoolBlocksForTest(batch.cacheResource(0).blockBinding("group0")), (BlockIndicesType{3}));
}

TEST(BatchKVCacheResourceTest, CopyOwnsTagMappedBlocksWhileMoveAndTimelineStateStayIntact) {
    auto config = makeResourceConfig({makeResourceGroup("full", CacheGroupType::FULL)}, {{"full"}});

    BatchKVCacheResource batch;
    batch.resetBatchSize(2);
    batch.initGroups(config);
    batch.mutableBlockBinding(0, "full").assign(poolBlockSnapshotForTest({3, 4}));
    batch.swapBlocks(0, "full", 0, 1);
    EXPECT_EQ(encodedPoolBlocksForTest(batch.blockBinding(0, "full")), (BlockIndicesType{4, 3}));
    batch.swapBlocks(0, "full", 0, 1);
    batch.cacheResource(0).setCacheKeysAndBlockDependencies({101, 202}, {{true, 7, 9}, {true, 101, 12}});
    batch.cacheResource(0).setCacheKeysAreCpCanonical(true);

    BatchKVCacheResource copied = batch;
    copied.mutableBlockBinding(0, "full").bind(GroupBlockPosition{1}, PoolBlockId{8});
    EXPECT_EQ(encodedPoolBlocksForTest(batch.blockBinding(0, "full")), (BlockIndicesType{3, 4}));
    EXPECT_EQ(copied.cacheKeys(0), (CacheKeysType{101, 202}));
    ASSERT_EQ(copied.cacheResource(0).blockDependencies().size(), 2u);
    EXPECT_EQ(copied.cacheResource(0).blockDependencies()[0].parent_key, 7);
    EXPECT_EQ(copied.cacheResource(0).blockDependencies()[0].ordinal, 9u);
    EXPECT_EQ(copied.cacheResource(0).blockDependencies()[1].parent_key, 101);
    EXPECT_EQ(copied.cacheResource(0).blockDependencies()[1].ordinal, 12u);
    EXPECT_TRUE(copied.cacheResource(0).cacheKeysAreCpCanonical());

    std::vector<KVCacheResource> old_resources;
    copied.resetAndReturnOldResources(/*new_batch_size=*/3, old_resources);
    copied.initGroups(config);
    copied.moveBatchResource(2, std::move(old_resources[0]));

    EXPECT_EQ(copied.batchSize(), 3);
    EXPECT_EQ(encodedPoolBlocksForTest(copied.blockBinding(2, "full")), (BlockIndicesType{3, 8}));
    EXPECT_EQ(copied.cacheKeys(2), (CacheKeysType{101, 202}));
    ASSERT_EQ(copied.cacheResource(2).blockDependencies().size(), 2u);
    EXPECT_EQ(copied.cacheResource(2).blockDependencies()[0].ordinal, 9u);
    EXPECT_EQ(copied.cacheResource(2).blockDependencies()[1].ordinal, 12u);
    EXPECT_TRUE(copied.cacheResource(2).cacheKeysAreCpCanonical());
    ASSERT_EQ(copied.cacheResource(0).blocksByTag().size(), 1u);
    ASSERT_EQ(copied.cacheResource(1).blocksByTag().size(), 1u);
    ASSERT_EQ(copied.cacheResource(2).blocksByTag().size(), 1u);
    EXPECT_EQ(copied.cacheResource(2).blocksByTag().count("full"), 1u);
}

}  // namespace test
}  // namespace rtp_llm
