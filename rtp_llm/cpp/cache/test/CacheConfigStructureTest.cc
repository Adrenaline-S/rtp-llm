#include <set>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm {
namespace {

template<typename T, typename = void>
struct HasPublicValueFinalizer: std::false_type {};

template<typename T>
struct HasPublicValueFinalizer<T,
                               std::void_t<decltype(std::declval<const T&>().withFinalizedBlockNums(
                                   std::declval<uint32_t>(), std::declval<const RuntimeConfig&>()))>>:
    std::true_type {};

static_assert(!HasPublicValueFinalizer<CacheConfig>::value,
              "CacheConfig resolution must remain private and creator-owned");
static_assert(!std::is_default_constructible_v<CacheConfig>,
              "published CacheConfig values must be created through a sealing authority");
static_assert(std::is_same_v<decltype(test::TestCacheConfigBuilder::makeBase(1, 1, 1, 1, DataType::TYPE_FP16)),
                             test::TestCacheConfigBuilder>,
              "test scalar factories must retain incomplete assembly inside the builder");

static_assert(std::is_same_v<CacheLayer, std::vector<std::string>>,
              "a cache layer is the ordered group identities at its vector-index layer id");

TEST(CacheConfigStructureTest, PublishesFlatGroupsIndexedLayersAndReadOnlyGlobals) {
    auto spec = test::makeResolvedMhaSpec(DataType::TYPE_FP16, 2, 1, 8, "full");

    CacheGroup group;
    group.tag                       = "full";
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = {0};
    group.block_num                 = 77;
    group.local_kv_head_num         = 2;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = 2;
    group.kv_block_stride_bytes     = group.spec->block_size_bytes();
    group.kv_scale_stride_bytes     = group.spec->scale_block_size_bytes();

    const auto config = test::TestCacheConfigBuilder()
                            .setMainLayerCount(1)
                            .setBlockCountBasis(77)
                            .setCacheKeyBlockTokens(8)
                            .setKernelBlockTokens(2)
                            .setDType(DataType::TYPE_FP16)
                            .addGroup(std::move(group))
                            .setLayerTags(0, {"full"})
                            .build();

    EXPECT_EQ(config.layers(), (std::vector<CacheLayer>{{"full"}}));
    EXPECT_EQ(config.group("full").local_kv_head_num, 2u);
    EXPECT_EQ(config.group("full").kernelBlocksPerPoolBlock(), 4u);
    EXPECT_EQ(config.mainLayerCount(), 1u);
    EXPECT_EQ(config.layerCount(), 1u);
    EXPECT_EQ(config.blockCountBasis(), 77u);
    EXPECT_EQ(config.cacheKeyBlockTokens(), 8u);
    EXPECT_EQ(config.kernelBlockTokens(), 2u);
    EXPECT_EQ(config.dtype(), DataType::TYPE_FP16);
}

CacheGroup makeGroup(std::string tag, std::vector<int> layer_ids, CacheGroupType type = CacheGroupType::FULL) {
    auto spec = test::makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, 8, tag);

    CacheGroup group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(type);
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = 16;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = type == CacheGroupType::FULL ? 2 : 8;
    return group;
}

CacheConfig makeConfig(std::vector<CacheGroup> groups, std::vector<CacheLayer> layers) {
    test::TestCacheConfigBuilder builder;
    builder.setMainLayerCount(static_cast<uint32_t>(layers.size()));
    for (auto& group : groups) {
        builder.addGroup(std::move(group));
    }
    for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
        builder.setLayerTags(static_cast<int>(layer_id), std::move(layers[layer_id]));
    }
    return builder.build();
}

// Build a group with a fully explicit physical layout so tag lookup can prove
// that each group retains its own complete record.
CacheGroup makeSigGroup(std::string      tag,
                        std::vector<int> layer_ids,
                        uint32_t         heads,
                        uint32_t         physical_b,
                        uint32_t         kernel_b,
                        CacheGroupType   type = CacheGroupType::FULL) {
    KVCacheSpecPtr spec;
    if (type == CacheGroupType::LINEAR) {
        spec = test::makeResolvedLinearSpec(DataType::TYPE_FP16,
                                            heads,
                                            heads,
                                            /*head_k_dim=*/1,
                                            /*head_v_dim=*/1,
                                            /*conv_kernel_dim=*/2,
                                            physical_b,
                                            DataType::TYPE_FP16,
                                            DataType::TYPE_FP16,
                                            tag);
    } else {
        spec = test::makeResolvedMhaSpec(DataType::TYPE_FP16, heads, /*size_per_head=*/1, physical_b, tag);
    }

    CacheGroup group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(type);
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = 16;
    group.local_kv_head_num         = heads;
    group.seq_size_per_block        = physical_b;
    group.kernel_seq_size_per_block = kernel_b;
    return group;
}

void expectCompleteGroupRecordEq(const CacheGroup& lhs, const CacheGroup& rhs) {
    ASSERT_NE(lhs.spec, nullptr);
    ASSERT_NE(rhs.spec, nullptr);
    EXPECT_EQ(lhs.tag, rhs.tag);
    EXPECT_EQ(lhs.layer_ids, rhs.layer_ids);
    EXPECT_TRUE(CacheConfig::samePolicy(lhs.policy, rhs.policy));
    EXPECT_EQ(lhs.block_num, rhs.block_num);
    EXPECT_EQ(lhs.local_kv_head_num, rhs.local_kv_head_num);
    EXPECT_EQ(lhs.seq_size_per_block, rhs.seq_size_per_block);
    EXPECT_EQ(lhs.kernel_seq_size_per_block, rhs.kernel_seq_size_per_block);
    EXPECT_EQ(lhs.kv_block_stride_bytes, rhs.kv_block_stride_bytes);
    EXPECT_EQ(lhs.kv_scale_stride_bytes, rhs.kv_scale_stride_bytes);
    EXPECT_EQ(lhs.spec->fingerprint(), rhs.spec->fingerprint());
}

TEST(CacheConfigStructureTest, TwoTagsOnOneLayerCarryDistinctPhysicalRecords) {
    // Two FULL groups share layer 0 but have different heads, B, K and strides.
    // No global scalar or group-parallel projection can describe layer 0; every
    // consumer must read the exact group record by tag.
    auto config = makeConfig({makeSigGroup("a", {0}, /*heads=*/2, /*B=*/8, /*kernel=*/2),
                              makeSigGroup("b", {0}, /*heads=*/5, /*B=*/4, /*kernel=*/4)},
                             {{"a", "b"}});

    const auto& a = config.groupForLayer(0, "a");
    const auto& b = config.groupForLayer(0, "b");
    EXPECT_NE(a.local_kv_head_num, b.local_kv_head_num);
    EXPECT_NE(a.spec->seq_size_per_block, b.spec->seq_size_per_block);
    EXPECT_NE(a.kernel_seq_size_per_block, b.kernel_seq_size_per_block);
    EXPECT_NE(a.kv_block_stride_bytes, b.kv_block_stride_bytes);

    EXPECT_EQ(a.local_kv_head_num, 2u);
    EXPECT_EQ(b.local_kv_head_num, 5u);
    EXPECT_EQ(a.kv_block_stride_bytes, 64u);
    EXPECT_EQ(b.kv_block_stride_bytes, 80u);
    EXPECT_EQ(a.kv_scale_stride_bytes, 0u);
    EXPECT_EQ(b.kv_scale_stride_bytes, 0u);
}

TEST(CacheConfigStructureTest, SupportsSingleGlobalGroupAsNEqualsOne) {
    auto config = makeConfig({makeGroup("full", {0, 1})}, {{"full"}, {"full"}});

    EXPECT_TRUE(config.hasSingleGlobalGroup());
    EXPECT_TRUE(config.hasOneGroupPerLayer());
    EXPECT_EQ(config.soleGroupForLayer(0).tag, "full");
    EXPECT_EQ(config.group(config.groupTagsForLayer(1).front()).tag, "full");
}

TEST(CacheConfigStructureTest, SupportsDistinctOneToOneGroupsAndOneToManyLayers) {
    auto config = makeConfig({makeGroup("full", {0, 2}), makeGroup("linear", {1, 2}, CacheGroupType::LINEAR)},
                             {{"full"}, {"linear"}, {"full", "linear"}});

    EXPECT_FALSE(config.hasSingleGlobalGroup());
    EXPECT_FALSE(config.hasOneGroupPerLayer());
    EXPECT_EQ(config.groupForLayer(2, "linear").policy.group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(config.groupTagsForLayer(2).size(), 2u);
    EXPECT_ANY_THROW(config.soleGroupForLayer(2));
}

TEST(CacheConfigStructureTest, CacheConfigPublishesTagLookupAndLayerMembership) {
    auto config = makeConfig({makeGroup("full", {0, 2}), makeGroup("linear", {1, 2}, CacheGroupType::LINEAR)},
                             {{"full"}, {"linear"}, {"full", "linear"}});

    const std::string_view full_tag   = "full";
    const std::string_view linear_tag = "linear";
    EXPECT_EQ(config.group(full_tag).tag, "full");
    EXPECT_EQ(config.groupTagsForLayer(2), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(config.groupForLayer(2, linear_tag).policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(config.soleGroupForLayer(0).tag, "full");
    EXPECT_ANY_THROW(config.groupForLayer(0, linear_tag));
}

TEST(CacheConfigStructureTest, TaggedGroupRecordsAreStableAndReadOnly) {
    auto config =
        makeConfig({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)}, {{"full", "linear"}});

    // The tagged group records are the only published view: repeated reads return
    // the same storage and every group stays reachable by tag.
    EXPECT_EQ(&config.group("full"), &config.group("full"));
    std::set<std::string> tags;
    for (const auto& group : config.groups()) {
        tags.insert(group.tag);
    }
    EXPECT_EQ(tags, (std::set<std::string>{"full", "linear"}));
    EXPECT_EQ(config.group("full").spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(config.group("linear").spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(config.groupTagsForLayer(0), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(config.groupForLayer(0, "linear").tag, "linear");
}

TEST(CacheConfigStructureTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first =
        makeConfig({makeSigGroup("full", {0}, 2, 8, 2), makeSigGroup("linear", {0}, 3, 4, 4, CacheGroupType::LINEAR)},
                   {{"full", "linear"}});
    auto reversed =
        makeConfig({makeSigGroup("linear", {0}, 3, 4, 4, CacheGroupType::LINEAR), makeSigGroup("full", {0}, 2, 8, 2)},
                   {{"full", "linear"}});

    // The two topologies declare the same tags in opposite storage order.
    EXPECT_NE(first.groups().front().tag, reversed.groups().front().tag);
    expectCompleteGroupRecordEq(first.group("full"), reversed.group("full"));
    expectCompleteGroupRecordEq(first.group("linear"), reversed.group("linear"));
    EXPECT_EQ(first.group("full").local_kv_head_num, 2u);
    EXPECT_EQ(first.group("linear").local_kv_head_num, 3u);
    EXPECT_NE(first.group("full").spec->fingerprint(), first.group("linear").spec->fingerprint());
    EXPECT_EQ(first.groupForLayer(0, "full").tag, reversed.groupForLayer(0, "full").tag);
    EXPECT_EQ(first.groupForLayer(0, "linear").tag, reversed.groupForLayer(0, "linear").tag);
}

TEST(CacheConfigStructureTest, RejectsInconsistentReverseMembership) {
    EXPECT_ANY_THROW(makeConfig({makeGroup("full", {0})}, {{"full"}, {"full"}}));
}

TEST(CacheConfigTest, OwnsGroupsAndLayerMembershipWithoutTopology) {
    CacheGroup linear = makeSigGroup("linear", {1}, /*heads=*/2, /*B=*/64, /*kernel=*/64, CacheGroupType::LINEAR);
    CacheGroup full   = makeSigGroup("full", {0}, /*heads=*/5, /*B=*/512, /*kernel=*/64);

    CacheConfig config = test::TestCacheConfigBuilder()
                             .addGroup(std::move(linear))
                             .addGroup(std::move(full))
                             .setLayerTags(0, {"full"})
                             .setLayerTags(1, {"linear"})
                             .build();

    EXPECT_EQ(config.group("full").seq_size_per_block, 512u);
    EXPECT_EQ(config.group("full").kernel_seq_size_per_block, 64u);
    EXPECT_EQ(config.group("linear").policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(config.groupTagsForLayer(0), std::vector<std::string>({"full"}));
    EXPECT_EQ(config.groupTagsForLayer(1), std::vector<std::string>({"linear"}));
}

TEST(CacheConfigTest, RejectsDuplicateTagAndInconsistentReverseMembership) {
    EXPECT_ANY_THROW(test::TestCacheConfigBuilder()
                         .addGroup(makeSigGroup("full", {0}, 1, 512, 64))
                         .addGroup(makeSigGroup("full", {0}, 1, 512, 64))
                         .setLayerTags(0, {"full"})
                         .build());
    EXPECT_ANY_THROW(test::TestCacheConfigBuilder()
                         .addGroup(makeSigGroup("full", {0}, 1, 512, 64))
                         .setLayerTags(0, {"missing"})
                         .build());
}

TEST(CacheConfigTest, RejectsResolvedStrideThatDiffersFromSpec) {
    auto group = makeSigGroup("full", {0}, 2, 8, 2);
    ASSERT_NE(group.spec, nullptr);
    group.kv_block_stride_bytes = group.spec->block_size_bytes() + 1;
    EXPECT_ANY_THROW(test::TestCacheConfigBuilder().addGroup(std::move(group)).setLayerTags(0, {"full"}).build());
}

TEST(CacheConfigTest, BuilderPreservesConfiguredGlobalsIndependentOfGroupOrder) {
    auto build = [](bool reversed) {
        auto builder =
            test::TestCacheConfigBuilder().setBlockCountBasis(77).setCacheKeyBlockTokens(99).setKernelBlockTokens(11);
        if (reversed) {
            builder.addGroup(makeSigGroup("linear", {1}, 3, 4, 4, CacheGroupType::LINEAR))
                .addGroup(makeSigGroup("full", {0}, 2, 8, 2));
        } else {
            builder.addGroup(makeSigGroup("full", {0}, 2, 8, 2))
                .addGroup(makeSigGroup("linear", {1}, 3, 4, 4, CacheGroupType::LINEAR));
        }
        return builder.setLayerTags(0, {"full"}).setLayerTags(1, {"linear"}).build();
    };

    const auto declared = build(false);
    const auto reversed = build(true);
    EXPECT_EQ(declared.blockCountBasis(), 77u);
    EXPECT_EQ(reversed.blockCountBasis(), 77u);
    EXPECT_EQ(declared.cacheKeyBlockTokens(), 99u);
    EXPECT_EQ(reversed.cacheKeyBlockTokens(), 99u);
    EXPECT_EQ(declared.kernelBlockTokens(), 11u);
    EXPECT_EQ(reversed.kernelBlockTokens(), 11u);
}

}  // namespace
}  // namespace rtp_llm
