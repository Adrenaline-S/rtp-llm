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
struct HasPublicValueFinalizer<
    T,
    std::void_t<decltype(std::declval<const T&>().withFinalizedBlockNums(
        std::declval<uint32_t>(), std::declval<const RuntimeConfig&>()))>>: std::true_type {};

static_assert(!HasPublicValueFinalizer<CacheConfig>::value,
              "CacheConfig resolution must remain private and creator-owned");

CacheGroup makeGroup(std::string tag, std::vector<int> layer_ids, CacheGroupType type = CacheGroupType::FULL) {
    auto spec = test::makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, 8, tag);

    CacheGroup group;
    group.tag                                = std::move(tag);
    group.layout.spec                        = std::move(spec);
    group.policy                             = defaultCacheGroupPolicy(type);
    group.layer_ids                          = std::move(layer_ids);
    group.layout.block_num                   = 16;
    group.layout.seq_size_per_block          = 8;
    group.layout.kernel_seq_size_per_block   = type == CacheGroupType::FULL ? 2 : 8;
    return group;
}

CacheConfig makeConfig(std::vector<CacheGroup> groups, std::vector<CacheLayerMembership> layers) {
    test::TestCacheConfigBuilder builder;
    for (auto& group : groups) {
        builder.addGroup(std::move(group));
    }
    for (auto& layer : layers) {
        builder.setLayerTags(layer.layer_id, std::move(layer.group_tags));
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
                       CacheGroupType   type      = CacheGroupType::FULL) {
    KVCacheSpecPtr spec;
    if (type == CacheGroupType::LINEAR) {
        spec = test::makeResolvedLinearSpec(
            DataType::TYPE_FP16, heads, heads, /*head_k_dim=*/1, /*head_v_dim=*/1, /*conv_kernel_dim=*/2, physical_b,
            DataType::TYPE_FP16, DataType::TYPE_FP16, tag);
    } else {
        spec = test::makeResolvedMhaSpec(DataType::TYPE_FP16, heads, /*size_per_head=*/1, physical_b, tag);
    }

    CacheGroup group;
    group.tag                              = std::move(tag);
    group.layout.spec                      = std::move(spec);
    group.policy                           = defaultCacheGroupPolicy(type);
    group.layer_ids                        = std::move(layer_ids);
    group.layout.block_num                 = 16;
    group.layout.local_kv_head_num         = heads;
    group.layout.seq_size_per_block        = physical_b;
    group.layout.kernel_seq_size_per_block = kernel_b;
    return group;
}

void expectCompleteGroupRecordEq(const CacheGroup& lhs, const CacheGroup& rhs) {
    ASSERT_NE(lhs.layout.spec, nullptr);
    ASSERT_NE(rhs.layout.spec, nullptr);
    EXPECT_EQ(lhs.tag, rhs.tag);
    EXPECT_EQ(lhs.layer_ids, rhs.layer_ids);
    EXPECT_TRUE(CacheConfig::samePolicy(lhs.policy, rhs.policy));
    EXPECT_EQ(lhs.layout.block_num, rhs.layout.block_num);
    EXPECT_EQ(lhs.layout.local_kv_head_num, rhs.layout.local_kv_head_num);
    EXPECT_EQ(lhs.layout.seq_size_per_block, rhs.layout.seq_size_per_block);
    EXPECT_EQ(lhs.layout.kernel_seq_size_per_block, rhs.layout.kernel_seq_size_per_block);
    EXPECT_EQ(lhs.layout.kv_block_stride_bytes, rhs.layout.kv_block_stride_bytes);
    EXPECT_EQ(lhs.layout.kv_scale_stride_bytes, rhs.layout.kv_scale_stride_bytes);
    EXPECT_EQ(lhs.layout.spec->fingerprint(), rhs.layout.spec->fingerprint());
}

TEST(CacheConfigStructureTest, TwoTagsOnOneLayerCarryDistinctPhysicalRecords) {
    // Two FULL groups share layer 0 but have different heads, B, K and strides.
    // No global scalar or group-parallel projection can describe layer 0; every
    // consumer must read the exact group record by tag.
    auto config = makeConfig({makeSigGroup("a", {0}, /*heads=*/2, /*B=*/8, /*kernel=*/2),
                              makeSigGroup("b", {0}, /*heads=*/5, /*B=*/4, /*kernel=*/4)},
                             {{0, {"a", "b"}}});

    const auto& a = config.groupForLayer(0, "a");
    const auto& b = config.groupForLayer(0, "b");
    EXPECT_NE(a.layout.local_kv_head_num, b.layout.local_kv_head_num);
    EXPECT_NE(a.layout.spec->seq_size_per_block, b.layout.spec->seq_size_per_block);
    EXPECT_NE(a.layout.kernel_seq_size_per_block, b.layout.kernel_seq_size_per_block);
    EXPECT_NE(a.layout.kv_block_stride_bytes, b.layout.kv_block_stride_bytes);

    EXPECT_EQ(a.layout.local_kv_head_num, 2u);
    EXPECT_EQ(b.layout.local_kv_head_num, 5u);
    EXPECT_EQ(a.layout.kv_block_stride_bytes, 64u);
    EXPECT_EQ(b.layout.kv_block_stride_bytes, 80u);
    EXPECT_EQ(a.layout.kv_scale_stride_bytes, 0u);
    EXPECT_EQ(b.layout.kv_scale_stride_bytes, 0u);
}


TEST(CacheConfigStructureTest, SupportsSingleGlobalGroupAsNEqualsOne) {
    auto config = makeConfig({makeGroup("full", {0, 1})}, {{0, {"full"}}, {1, {"full"}}});

    EXPECT_TRUE(config.hasSingleGlobalGroup());
    EXPECT_TRUE(config.hasOneGroupPerLayer());
    EXPECT_EQ(config.soleGroupForLayer(0).tag, "full");
    EXPECT_EQ(config.group(config.groupTagsForLayer(1).front()).tag, "full");
}

TEST(CacheConfigStructureTest, SupportsDistinctOneToOneGroupsAndOneToManyLayers) {
    auto config = makeConfig({makeGroup("full", {0, 2}), makeGroup("linear", {1, 2}, CacheGroupType::LINEAR)},
                             {{0, {"full"}}, {1, {"linear"}}, {2, {"full", "linear"}}});

    EXPECT_FALSE(config.hasSingleGlobalGroup());
    EXPECT_FALSE(config.hasOneGroupPerLayer());
    EXPECT_EQ(config.groupForLayer(2, "linear").policy.group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(config.groupTagsForLayer(2).size(), 2u);
    EXPECT_ANY_THROW(config.soleGroupForLayer(2));
}

TEST(CacheConfigStructureTest, CacheConfigPublishesTagLookupAndLayerMembership) {
    auto config = makeConfig({makeGroup("full", {0, 2}), makeGroup("linear", {1, 2}, CacheGroupType::LINEAR)},
                             {{0, {"full"}}, {1, {"linear"}}, {2, {"full", "linear"}}});

    const std::string_view full_tag   = "full";
    const std::string_view linear_tag = "linear";
    EXPECT_EQ(config.group(full_tag).tag, "full");
    EXPECT_EQ(config.groupTagsForLayer(2), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(config.groupForLayer(2, linear_tag).policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(config.soleGroupForLayer(0).tag, "full");
    EXPECT_ANY_THROW(config.groupForLayer(0, linear_tag));
}

TEST(CacheConfigStructureTest, TaggedGroupRecordsAreStableAndReadOnly) {
    auto config = makeConfig({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                             {{0, {"full", "linear"}}});

    // The tagged group records are the only published view: repeated reads return
    // the same storage and every group stays reachable by tag.
    EXPECT_EQ(&config.group("full"), &config.group("full"));
    std::set<std::string> tags;
    for (const auto& group : config.groups()) {
        tags.insert(group.tag);
    }
    EXPECT_EQ(tags, (std::set<std::string>{"full", "linear"}));
    EXPECT_EQ(config.group("full").layout.spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(config.group("linear").layout.spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(config.groupTagsForLayer(0), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(config.groupForLayer(0, "linear").tag, "linear");
}

TEST(CacheConfigStructureTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first = makeConfig({makeSigGroup("full", {0}, 2, 8, 2),
                             makeSigGroup("linear", {0}, 3, 4, 4, CacheGroupType::LINEAR)},
                            {{0, {"full", "linear"}}});
    auto reversed = makeConfig({makeSigGroup("linear", {0}, 3, 4, 4, CacheGroupType::LINEAR),
                                makeSigGroup("full", {0}, 2, 8, 2)},
                               {{0, {"full", "linear"}}});

    // The two topologies declare the same tags in opposite storage order.
    EXPECT_NE(first.groups().front().tag, reversed.groups().front().tag);
    expectCompleteGroupRecordEq(first.group("full"), reversed.group("full"));
    expectCompleteGroupRecordEq(first.group("linear"), reversed.group("linear"));
    EXPECT_EQ(first.group("full").layout.local_kv_head_num, 2u);
    EXPECT_EQ(first.group("linear").layout.local_kv_head_num, 3u);
    EXPECT_NE(first.group("full").layout.spec->fingerprint(), first.group("linear").layout.spec->fingerprint());
    EXPECT_EQ(first.groupForLayer(0, "full").tag, reversed.groupForLayer(0, "full").tag);
    EXPECT_EQ(first.groupForLayer(0, "linear").tag, reversed.groupForLayer(0, "linear").tag);
}

TEST(CacheConfigStructureTest, RejectsInconsistentReverseMembership) {
    EXPECT_ANY_THROW(makeConfig({makeGroup("full", {0})}, {{0, {"full"}}, {1, {"full"}}}));
}

TEST(CacheConfigTest, OwnsGroupsAndLayerMembershipWithoutTopology) {
    CacheGroup linear = makeSigGroup(
        "linear", {1}, /*heads=*/2, /*B=*/64, /*kernel=*/64, CacheGroupType::LINEAR);
    CacheGroup full = makeSigGroup(
        "full", {0}, /*heads=*/5, /*B=*/512, /*kernel=*/64);

    CacheConfig config = test::TestCacheConfigBuilder()
                             .addGroup(std::move(linear))
                             .addGroup(std::move(full))
                             .setLayerTags(0, {"full"})
                             .setLayerTags(1, {"linear"})
                             .build();

    EXPECT_EQ(config.group("full").layout.seq_size_per_block, 512u);
    EXPECT_EQ(config.group("full").layout.kernel_seq_size_per_block, 64u);
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
    ASSERT_NE(group.layout.spec, nullptr);
    group.layout.kv_block_stride_bytes = group.layout.spec->block_size_bytes() + 1;
    EXPECT_ANY_THROW(test::TestCacheConfigBuilder()
                         .addGroup(std::move(group))
                         .setLayerTags(0, {"full"})
                         .build());
}

TEST(CacheConfigTest, BuilderPreservesConfiguredGlobalsIndependentOfGroupOrder) {
    CacheConfig base;
    base.block_num                 = 77;
    base.seq_size_per_block        = 99;
    base.kernel_seq_size_per_block = 11;
    auto build = [&base](bool reversed) {
        auto builder = test::TestCacheConfigBuilder().configure(base);
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
    EXPECT_EQ(declared.block_num, 77u);
    EXPECT_EQ(reversed.block_num, 77u);
    EXPECT_EQ(declared.seq_size_per_block, 99u);
    EXPECT_EQ(reversed.seq_size_per_block, 99u);
    EXPECT_EQ(declared.kernel_seq_size_per_block, 11u);
    EXPECT_EQ(reversed.kernel_seq_size_per_block, 11u);
}

}  // namespace
}  // namespace rtp_llm
