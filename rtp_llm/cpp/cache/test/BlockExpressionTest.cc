#include <gtest/gtest.h>

#include <limits>

#include "rtp_llm/cpp/cache/BlockExpression.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace test {

TEST(BlockExpressionTest, MapsKeyPositionsRangesAndAnchors) {
    CacheKeyToGroupBlockMapping mapping(/*cache_key_tokens=*/64, /*group_block_tokens=*/512);

    EXPECT_EQ(mapping.toGroupBlock(CacheKeyPosition{9}), GroupBlockPosition{1});
    EXPECT_EQ(mapping.toCacheKeyRange(GroupBlockPosition{1}, 20),
              (CacheKeyRange{CacheKeyPosition{8}, CacheKeyPosition{16}}));
    EXPECT_EQ(mapping.toCacheKeyAnchor(GroupBlockPosition{1}, 20), CacheKeyPosition{15});
    EXPECT_EQ(mapping.completeGroupBlockCount(/*key_count=*/20), 2u);
    EXPECT_EQ(mapping.toCacheKeyPrefixLength(/*group_block_count=*/2), 16u);
}

TEST(BlockExpressionTest, KeepsNonemptyPartialRangeAndUsesItsLastKeyAsAnchor) {
    CacheKeyToGroupBlockMapping mapping(/*cache_key_tokens=*/64, /*group_block_tokens=*/512);

    EXPECT_EQ(mapping.toCacheKeyRange(GroupBlockPosition{2}, 20),
              (CacheKeyRange{CacheKeyPosition{16}, CacheKeyPosition{20}}));
    EXPECT_EQ(mapping.toCacheKeyAnchor(GroupBlockPosition{2}, 20), CacheKeyPosition{19});
    EXPECT_THROW((void)mapping.toCacheKeyAnchor(GroupBlockPosition{3}, 20), RTPException);
}

TEST(BlockExpressionTest, RejectsInvalidCacheKeyToGroupBlockDivisibility) {
    EXPECT_THROW((void)CacheKeyToGroupBlockMapping(/*cache_key_tokens=*/96, /*group_block_tokens=*/512), RTPException);
    EXPECT_THROW((void)CacheKeyToGroupBlockMapping(/*cache_key_tokens=*/0, /*group_block_tokens=*/512), RTPException);
}

TEST(BlockExpressionTest, MissingBindingDoesNotConfusePoolBlockZero) {
    GroupBlockToPoolBlockBinding binding;
    binding.resize(2);
    binding.bind(GroupBlockPosition{1}, PoolBlockId{0});

    EXPECT_FALSE(binding.lookup(GroupBlockPosition{0}).has_value());
    EXPECT_EQ(binding.lookup(GroupBlockPosition{1}), PoolBlockId{0});
}

TEST(BlockExpressionTest, BindingBulkSurfaceUsesOptionalStrongPoolIdentities) {
    GroupBlockToPoolBlockBinding binding;
    binding.assign({PoolBlockId{7}, std::nullopt, PoolBlockId{0}});

    const std::vector<std::optional<PoolBlockId>> expected = {PoolBlockId{7}, std::nullopt, PoolBlockId{0}};
    EXPECT_EQ(binding.snapshot(), expected);

    binding.append(std::vector<PoolBlockId>{PoolBlockId{9}});
    EXPECT_EQ(binding.lookup(GroupBlockPosition{3}), PoolBlockId{9});
}

TEST(BlockExpressionTest, ExpandsPoolBlocksIntoKernelBlocks) {
    PoolBlockToKernelBlockProjection projection(/*kernel_blocks_per_pool_block=*/4);
    BlockIndicesType                 destination;

    EXPECT_EQ(projection.projectedSize(2), 8u);
    projection.project({PoolBlockId{2}, PoolBlockId{0}}, destination);
    EXPECT_EQ(destination, (BlockIndicesType{8, 9, 10, 11, 0, 1, 2, 3}));
}

TEST(BlockExpressionTest, AppendsCheckedKernelProjectionWithoutClearingExistingBlocks) {
    PoolBlockToKernelBlockProjection projection(/*kernel_blocks_per_pool_block=*/4);
    BlockIndicesType                 destination{99};

    projection.append(PoolBlockId{2}, destination);
    projection.append(PoolBlockId{0}, destination);
    EXPECT_EQ(destination, (BlockIndicesType{99, 8, 9, 10, 11, 0, 1, 2, 3}));

    const auto                       before_overflow = destination;
    PoolBlockToKernelBlockProjection overflowing(std::numeric_limits<uint64_t>::max() / 4 + 2);
    EXPECT_THROW(overflowing.append(PoolBlockId{3}, destination), RTPException);
    EXPECT_EQ(destination, before_overflow);
}

TEST(BlockExpressionTest, RejectsInvalidPoolToKernelDivisibility) {
    EXPECT_THROW((void)PoolBlockToKernelBlockProjection(/*pool_block_tokens=*/512, /*kernel_block_tokens=*/96),
                 RTPException);
    EXPECT_THROW((void)PoolBlockToKernelBlockProjection(/*pool_block_tokens=*/512, /*kernel_block_tokens=*/0),
                 RTPException);
}

TEST(BlockExpressionTest, RejectsProjectionSizeAndFinalKernelIdOverflowBeforeArithmeticWraps) {
    PoolBlockToKernelBlockProjection pair_projection(/*kernel_blocks_per_pool_block=*/2);
    EXPECT_THROW((void)pair_projection.projectedSize(std::numeric_limits<size_t>::max()), RTPException);

    const size_t                     wrapping_factor = std::numeric_limits<uint64_t>::max() / 4 + 2;
    PoolBlockToKernelBlockProjection wrapping_projection(wrapping_factor);
    BlockIndicesType                 destination;
    EXPECT_THROW(wrapping_projection.project({PoolBlockId{3}}, destination), RTPException);
    EXPECT_TRUE(destination.empty());
}

}  // namespace test
}  // namespace rtp_llm
