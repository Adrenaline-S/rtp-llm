#include <set>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

GroupBase makeGroup(std::string tag, std::vector<int> layer_ids, CacheGroupType type = CacheGroupType::FULL) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 8;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(type);
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = 16;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = type == CacheGroupType::FULL ? 2 : 8;
    return group;
}

CacheConfig makeConfig(std::vector<GroupBase> groups, std::vector<LayerBase> layers) {
    CacheConfig config;
    config.layer_num     = static_cast<uint32_t>(layers.size());
    config.layer_all_num = config.layer_num;
    config.setTopology(std::move(groups), std::move(layers));
    return config;
}

// Build a group with a fully explicit physical layout so signature tests can
// vary each physical dimension independently. K = physical_b / kernel_b.
GroupBase makeSigGroup(std::string      tag,
                       std::vector<int> layer_ids,
                       uint32_t         heads,
                       uint32_t         physical_b,
                       uint32_t         kernel_b,
                       size_t           kv_stride,
                       size_t           scale_stride,
                       KVCacheSpecType  spec_type = KVCacheSpecType::MultiHeadAttention,
                       CacheGroupType   type      = CacheGroupType::FULL) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = physical_b;
    spec->type               = spec_type;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(type);
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = 16;
    group.local_kv_head_num         = heads;
    group.seq_size_per_block        = physical_b;
    group.kernel_seq_size_per_block = kernel_b;
    group.kv_block_stride_bytes     = kv_stride;
    group.kv_scale_stride_bytes     = scale_stride;
    return group;
}

TEST(CacheTopologySignatureTest, TwoTagsOnOneLayerCarryDistinctPhysicalRecords) {
    // Two FULL groups share layer 0 but have different heads, B, K and strides.
    // No global scalar or group-parallel projection can describe layer 0; every
    // consumer must read the exact group record by tag.
    auto config = makeConfig({makeSigGroup("a", {0}, /*heads=*/2, /*B=*/8, /*kernel=*/2, /*kv=*/64, /*scale=*/8),
                              makeSigGroup("b", {0}, /*heads=*/5, /*B=*/4, /*kernel=*/4, /*kv=*/48, /*scale=*/0)},
                             {{0, {"a", "b"}}});

    const auto& a = config.groupForLayer(0, "a");
    const auto& b = config.groupForLayer(0, "b");
    EXPECT_NE(a.local_kv_head_num, b.local_kv_head_num);
    EXPECT_NE(a.spec->seq_size_per_block, b.spec->seq_size_per_block);
    EXPECT_NE(a.kernel_seq_size_per_block, b.kernel_seq_size_per_block);
    EXPECT_NE(a.kv_block_stride_bytes, b.kv_block_stride_bytes);
    EXPECT_NE(a.kv_scale_stride_bytes, b.kv_scale_stride_bytes);

    EXPECT_EQ(a.local_kv_head_num, 2u);
    EXPECT_EQ(b.local_kv_head_num, 5u);
    EXPECT_EQ(a.kv_block_stride_bytes, 64u);
    EXPECT_EQ(b.kv_block_stride_bytes, 48u);

    // The signature reflects both distinct physical records, not a single value.
    const auto signature = config.physicalTopologySignature();
    EXPECT_FALSE(signature.empty());
}

TEST(CacheTopologySignatureTest, EncodesExactCanonicalBytes) {
    auto config = makeConfig({makeSigGroup("a", {0}, /*heads=*/3, /*B=*/4, /*kernel=*/2, /*kv=*/16, /*scale=*/0)},
                             {{0, {"a"}}});

    const auto put_u32 = [](std::string& out, uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            out.push_back(static_cast<char>((value >> (8 * i)) & 0xFFu));
        }
    };
    const auto put_u64 = [](std::string& out, uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            out.push_back(static_cast<char>((value >> (8 * i)) & 0xFFull));
        }
    };
    const auto put_str = [&](std::string& out, const std::string& value) {
        put_u32(out, static_cast<uint32_t>(value.size()));
        out.append(value);
    };

    std::string expected;
    put_u32(expected, 1);                                                 // group count
    put_str(expected, "a");                                               // tag
    put_u32(expected, static_cast<uint32_t>(KVCacheSpecType::MultiHeadAttention));  // spec layout
    put_u32(expected, 3);                                                 // local heads
    put_u64(expected, 4);                                                 // physical B
    put_u64(expected, 2);                                                 // K = B / kernel
    put_u64(expected, 16);                                                // KV stride
    put_u64(expected, 0);                                                 // scale stride
    put_u32(expected, 1);                                                 // layer count
    put_u32(expected, 1);                                                 // layer 0 tag count
    put_str(expected, "a");                                               // layer 0 tag

    EXPECT_EQ(config.physicalTopologySignature(), expected);
    EXPECT_EQ(config.physicalTopologySignature().size(), expected.size());
}

TEST(CacheTopologySignatureTest, IgnoresDeclarationOrderCapacityPolicyAndBudget) {
    auto baseline = makeConfig({makeSigGroup("a", {0}, 2, 8, 2, 64, 8),
                                makeSigGroup("b", {0}, 5, 4, 4, 48, 0)},
                               {{0, {"a", "b"}}});

    // Reversed declaration order.
    auto reversed = makeConfig({makeSigGroup("b", {0}, 5, 4, 4, 48, 0),
                                makeSigGroup("a", {0}, 2, 8, 2, 64, 8)},
                               {{0, {"a", "b"}}});
    EXPECT_EQ(baseline.physicalTopologySignature(), reversed.physicalTopologySignature());

    // Different capacity (block_num) and budget (explicit_block_num) and policy.
    auto variant_a          = makeSigGroup("a", {0}, 2, 8, 2, 64, 8);
    auto variant_b          = makeSigGroup("b", {0}, 5, 4, 4, 48, 0);
    variant_a.block_num     = 999;
    variant_b.block_num     = 7;
    variant_a.policy.explicit_block_num = 123;             // budget
    variant_a.policy.enable_prefix_reuse = !variant_a.policy.enable_prefix_reuse;  // behavior
    variant_b.policy.evict_policy        = CacheEvictPolicy::INDEPENDENT;
    auto capacity_policy_variant =
        makeConfig({std::move(variant_a), std::move(variant_b)}, {{0, {"a", "b"}}});
    EXPECT_EQ(baseline.physicalTopologySignature(), capacity_policy_variant.physicalTopologySignature());
}

TEST(CacheTopologySignatureTest, ChangesWithTagMembershipAndPhysicalLayout) {
    auto baseline = makeConfig({makeSigGroup("a", {0, 1}, 2, 8, 2, 64, 8),
                                makeSigGroup("b", {1}, 5, 4, 4, 48, 0)},
                               {{0, {"a"}}, {1, {"a", "b"}}});
    const auto base_sig = baseline.physicalTopologySignature();

    // Tag rename.
    auto tag_changed = makeConfig({makeSigGroup("a", {0, 1}, 2, 8, 2, 64, 8),
                                   makeSigGroup("c", {1}, 5, 4, 4, 48, 0)},
                                  {{0, {"a"}}, {1, {"a", "c"}}});
    EXPECT_NE(base_sig, tag_changed.physicalTopologySignature());

    // Layer membership change (move "b" to layer 0 as well).
    auto membership_changed = makeConfig({makeSigGroup("a", {0, 1}, 2, 8, 2, 64, 8),
                                          makeSigGroup("b", {0, 1}, 5, 4, 4, 48, 0)},
                                         {{0, {"a", "b"}}, {1, {"a", "b"}}});
    EXPECT_NE(base_sig, membership_changed.physicalTopologySignature());

    const auto sig_of = [](GroupBase a, GroupBase b) {
        return makeConfig({std::move(a), std::move(b)}, {{0, {"a"}}, {1, {"a", "b"}}}).physicalTopologySignature();
    };

    // spec layout.
    EXPECT_NE(base_sig,
              sig_of(makeSigGroup("a", {0, 1}, 2, 8, 2, 64, 8, KVCacheSpecType::MultiHeadLatentAttention),
                     makeSigGroup("b", {1}, 5, 4, 4, 48, 0)));
    // head count.
    EXPECT_NE(base_sig, sig_of(makeSigGroup("a", {0, 1}, 9, 8, 2, 64, 8), makeSigGroup("b", {1}, 5, 4, 4, 48, 0)));
    // physical B.
    EXPECT_NE(base_sig, sig_of(makeSigGroup("a", {0, 1}, 2, 16, 2, 64, 8), makeSigGroup("b", {1}, 5, 4, 4, 48, 0)));
    // K (same B, different kernel).
    EXPECT_NE(base_sig, sig_of(makeSigGroup("a", {0, 1}, 2, 8, 4, 64, 8), makeSigGroup("b", {1}, 5, 4, 4, 48, 0)));
    // KV stride.
    EXPECT_NE(base_sig, sig_of(makeSigGroup("a", {0, 1}, 2, 8, 2, 128, 8), makeSigGroup("b", {1}, 5, 4, 4, 48, 0)));
    // scale stride.
    EXPECT_NE(base_sig, sig_of(makeSigGroup("a", {0, 1}, 2, 8, 2, 64, 16), makeSigGroup("b", {1}, 5, 4, 4, 48, 0)));
}

TEST(CacheTopologySignatureTest, MatchGuardAcceptsEqualAndRejectsDifferentSignatures) {
    auto config = makeConfig({makeSigGroup("a", {0}, 2, 8, 2, 64, 8)}, {{0, {"a"}}});
    const auto sig = config.physicalTopologySignature();

    EXPECT_NO_THROW(checkPhysicalTopologyMatches(sig, sig, "unit"));
    EXPECT_ANY_THROW(checkPhysicalTopologyMatches(sig, sig + "x", "unit"));
}

TEST(CacheTopologySignatureTest, GroupPhysicalLayoutGuardValidatesSubsetTopologies) {
    auto config = makeConfig({makeSigGroup("a", {0, 1}, 2, 8, 2, 64, 8),
                              makeSigGroup("b", {1}, 5, 4, 4, 48, 0)},
                             {{0, {"a"}}, {1, {"a", "b"}}});

    // A layer-subset view carrying the same physical layout is compatible.
    auto subset =
        CacheTopology::create({makeSigGroup("a", {0}, 2, 8, 2, 64, 8)}, {{0, {"a"}}});
    EXPECT_NO_THROW(config.checkPhysicalGroupLayoutCompatible(*subset, "unit"));

    // A physically divergent group with the same tag fails fast.
    auto divergent =
        CacheTopology::create({makeSigGroup("a", {0}, 3, 8, 2, 64, 8)}, {{0, {"a"}}});
    EXPECT_ANY_THROW(config.checkPhysicalGroupLayoutCompatible(*divergent, "unit"));

    // An unknown tag fails fast.
    auto unknown =
        CacheTopology::create({makeSigGroup("z", {0}, 2, 8, 2, 64, 8)}, {{0, {"z"}}});
    EXPECT_ANY_THROW(config.checkPhysicalGroupLayoutCompatible(*unknown, "unit"));
}


TEST(CacheTopologyTest, SupportsSingleGlobalGroupAsNEqualsOne) {
    auto topology = CacheTopology::create({makeGroup("full", {0, 1})}, {{0, {"full"}}, {1, {"full"}}});

    EXPECT_TRUE(topology->hasSingleGlobalGroup());
    EXPECT_TRUE(topology->hasOneGroupPerLayer());
    EXPECT_EQ(topology->soleGroupForLayer(0).tag, "full");
    EXPECT_EQ(topology->groupsForLayer(1).front().get().tag, "full");
}

TEST(CacheTopologyTest, SupportsDistinctOneToOneGroupsAndOneToManyLayers) {
    auto topology =
        CacheTopology::create({makeGroup("full", {0, 2}), makeGroup("linear", {1, 2}, CacheGroupType::LINEAR)},
                              {{0, {"full"}}, {1, {"linear"}}, {2, {"full", "linear"}}});

    EXPECT_FALSE(topology->hasSingleGlobalGroup());
    EXPECT_FALSE(topology->hasOneGroupPerLayer());
    EXPECT_EQ(topology->groupForLayer(2, "linear").policy.group_type, CacheGroupType::LINEAR);
    ASSERT_EQ(topology->groupsForLayer(2).size(), 2u);
    EXPECT_ANY_THROW(topology->soleGroupForLayer(2));
}

TEST(CacheTopologyTest, CacheConfigPublishesTagLookupAndLayerMembership) {
    auto config = makeConfig({makeGroup("full", {0, 2}), makeGroup("linear", {1, 2}, CacheGroupType::LINEAR)},
                             {{0, {"full"}}, {1, {"linear"}}, {2, {"full", "linear"}}});

    const std::string_view full_tag   = "full";
    const std::string_view linear_tag = "linear";
    EXPECT_EQ(config.group(full_tag).tag, "full");
    EXPECT_EQ(config.groupsForLayer(2), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(config.groupForLayer(2, linear_tag).policy.group_type, CacheGroupType::LINEAR);
    EXPECT_EQ(config.soleGroupForLayer(0).tag, "full");
    EXPECT_ANY_THROW(config.groupForLayer(0, linear_tag));
}

TEST(CacheTopologyTest, TaggedGroupRecordsAreStableAndReadOnly) {
    auto topology = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});

    // The tagged group records are the only published view: repeated reads return
    // the same storage and every group stays reachable by tag.
    EXPECT_EQ(&topology->group("full"), &topology->group("full"));
    std::set<std::string> tags;
    for (const auto& group : topology->groups()) {
        tags.insert(group.tag);
    }
    EXPECT_EQ(tags, (std::set<std::string>{"full", "linear"}));
    EXPECT_EQ(topology->group("full").spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(topology->group("linear").spec->type, KVCacheSpecType::MultiHeadAttention);
    EXPECT_EQ(topology->layer(0).group_tags, (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(topology->groupForLayer(0, "linear").tag, "linear");
}

TEST(CacheTopologyTest, TagIdentityDoesNotDependOnNumericGroupOrder) {
    auto first    = CacheTopology::create({makeGroup("full", {0}), makeGroup("linear", {0}, CacheGroupType::LINEAR)},
                                          {{0, {"full", "linear"}}});
    auto reversed = CacheTopology::create({makeGroup("linear", {0}, CacheGroupType::LINEAR), makeGroup("full", {0})},
                                          {{0, {"full", "linear"}}});

    // The two topologies declare the same tags in opposite storage order.
    EXPECT_NE(first->groups().front().tag, reversed->groups().front().tag);
    EXPECT_EQ(first->group("full").policy.group_type, reversed->group("full").policy.group_type);
    EXPECT_EQ(first->group("linear").policy.group_type, reversed->group("linear").policy.group_type);
    EXPECT_EQ(first->groupForLayer(0, "full").tag, reversed->groupForLayer(0, "full").tag);
    EXPECT_EQ(first->groupForLayer(0, "linear").tag, reversed->groupForLayer(0, "linear").tag);
}

TEST(CacheTopologyTest, RejectsInconsistentReverseMembership) {
    EXPECT_ANY_THROW(CacheTopology::create({makeGroup("full", {0})}, {{0, {"full"}}, {1, {"full"}}}));
}

}  // namespace
}  // namespace rtp_llm
