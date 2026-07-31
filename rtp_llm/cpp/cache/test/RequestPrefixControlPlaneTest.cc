#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/RequestPrefixResource.h"
#include "rtp_llm/cpp/cache/connector/RequestPrefixManifestStore.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {
namespace {

std::shared_ptr<const CacheTopology>
makeTopology(size_t first_span, size_t second_span, std::string first_tag, std::string second_tag, bool reverse) {
    auto      first_spec  = makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, first_span, first_tag);
    auto      second_spec = makeResolvedMhaSpec(DataType::TYPE_FP16, 1, 1, second_span, second_tag);
    auto      first       = makeTestGroupBase(first_spec, defaultCacheGroupPolicy(CacheGroupType::FULL), {0});
    auto      second      = makeTestGroupBase(second_spec, defaultCacheGroupPolicy(CacheGroupType::FULL), {0});
    LayerBase layer{
        0, reverse ? std::vector<std::string>{second_tag, first_tag} : std::vector<std::string>{first_tag, second_tag}};
    return CacheTopology::create(
        reverse ? std::vector<GroupBase>{second, first} : std::vector<GroupBase>{first, second}, {layer});
}

TEST(RequestPrefixResourceTest, CanonicalHashesIgnoreTagsAndTopologyOrder) {
    std::vector<int32_t> tokens(37);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = static_cast<int32_t>(i * 7 + 3);
    }
    RequestPrefixResource lhs;
    lhs.configure(*makeTopology(4, 6, "full-a", "full-b", false));
    lhs.rebuild(tokens.data(), tokens.size());
    RequestPrefixResource rhs;
    rhs.configure(*makeTopology(4, 6, "renamed-x", "renamed-y", true));
    rhs.rebuild(tokens.data(), tokens.size());

    EXPECT_EQ(lhs.matchSpanTokens(), 12u);
    EXPECT_EQ(lhs.keys(), rhs.keys());
    EXPECT_EQ(lhs.tokenExtent(), 37u);
    EXPECT_EQ(lhs.matchLimitTokens(), 36u);
    EXPECT_EQ(lhs.writeLimitTokens(), 36u);
    ASSERT_EQ(lhs.keys().size(), 3u);

    tokens.push_back(999);
    lhs.rebuild(tokens.data(), tokens.size());
    EXPECT_EQ(lhs.keys().size(), 3u);
    EXPECT_EQ(lhs.tokenExtent(), 38u);
}

TEST(KVCacheTransferProjectorTest, HeterogeneousFullCountsAndPartialDirectTail) {
    GroupBase span4;
    span4.tag                = "four";
    span4.seq_size_per_block = 4;
    span4.policy             = defaultCacheGroupPolicy(CacheGroupType::FULL);
    GroupBase span6          = span4;
    span6.tag                = "six";
    span6.seq_size_per_block = 6;

    const auto four = projectTokenRangeForGroup(span4, 12, 36, CacheTransferRangeMode::PREFIX_ALIGNED);
    const auto six  = projectTokenRangeForGroup(span6, 12, 36, CacheTransferRangeMode::PREFIX_ALIGNED);
    EXPECT_EQ(four.global_positions.size(), 6u);
    EXPECT_EQ(six.global_positions.size(), 4u);
    EXPECT_EQ(four.global_positions.front(), 3u);
    EXPECT_EQ(six.global_positions.front(), 2u);

    const auto partial = projectTokenRangeForGroup(span6, 12, 37, CacheTransferRangeMode::DIRECT_TERMINAL);
    EXPECT_EQ(partial.global_positions.size(), 5u);
    EXPECT_EQ(partial.global_positions.back(), 6u);
}

TEST(KVCacheTransferProjectorTest, TailAndCpSelectionsRemainExplicit) {
    GroupBase tail;
    tail.tag                       = "state";
    tail.seq_size_per_block        = 4;
    tail.policy                    = defaultCacheGroupPolicy(CacheGroupType::SWA);
    tail.policy.active_tail_blocks = 2;
    const auto tail_selection      = projectTokenRangeForGroup(tail, 0, 36, CacheTransferRangeMode::PREFIX_ALIGNED);
    EXPECT_EQ(tail_selection.global_positions, (std::vector<size_t>{7, 8}));

    GroupBase cp;
    cp.tag                = "full";
    cp.seq_size_per_block = 4;
    cp.policy             = defaultCacheGroupPolicy(CacheGroupType::FULL);
    const auto rank1      = projectTokenRangeForGroup(cp, 0, 36, CacheTransferRangeMode::PREFIX_ALIGNED, 1, 2);
    EXPECT_EQ(rank1.local_positions, (std::vector<size_t>{0, 1, 2, 3}));
    EXPECT_EQ(rank1.global_positions.size(), 9u);
}

TEST(RequestPrefixManifestStoreTest, AtomicChainPinAndEviction) {
    auto                  store = std::make_shared<RequestPrefixManifestStore>();
    RequestPrefixManifest root{{11, 12}, std::nullopt, {{"four", 101, 0, NativeCacheItemKind::FULL_INTERVAL, 0}}};
    RequestPrefixManifest child{
        {22, 24}, RequestPrefixManifestKey{11, 12}, {{"six", 202, 3, NativeCacheItemKind::ACTIVE_TAIL, 0}}};
    EXPECT_TRUE(store->publish(root));
    EXPECT_TRUE(store->publish(child));
    EXPECT_FALSE(store->publish({{33, 36}, RequestPrefixManifestKey{999, 24}, {}}));

    const std::vector<RequestPrefixKey> keys{11, 22, 33};
    RequestPrefixMatchView              view(keys, 12, 37, 36, 36, 0);
    auto                                pinned = store->match(view, 0);
    ASSERT_NE(pinned, nullptr);
    EXPECT_EQ(pinned->matchedTokenCount(), 24u);
    EXPECT_EQ(store->pinCount({11, 12}), 1u);
    EXPECT_TRUE(store->evict({11, 12}));
    EXPECT_EQ(store->visibleSize(), 1u);
    pinned.reset();
    EXPECT_EQ(store->pinCount({11, 12}), 0u);
}

TEST(RequestPrefixManifestStoreTest, CancelReleasesPins) {
    auto store = std::make_shared<RequestPrefixManifestStore>();
    ASSERT_TRUE(store->publish({{7, 12}, std::nullopt, {{"full", 70, 0, NativeCacheItemKind::FULL_INTERVAL, 0}}}));
    const std::vector<RequestPrefixKey> keys{7};
    {
        RequestPrefixMatchView view(keys, 12, 13, 12, 12, 0);
        auto                   pinned = store->match(view, 0);
        ASSERT_NE(pinned, nullptr);
        EXPECT_EQ(store->pinCount({7, 12}), 1u);
    }
    EXPECT_EQ(store->pinCount({7, 12}), 0u);
}

}  // namespace
}  // namespace rtp_llm::test
