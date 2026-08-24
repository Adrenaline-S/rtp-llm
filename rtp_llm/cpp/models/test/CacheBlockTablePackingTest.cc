#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/BlockExpression.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/models/CacheBlockTablePacking.h"
#include "rtp_llm/cpp/models/ModelTypes.h"

namespace rtp_llm {
namespace {

CacheGroup
makeGroup(std::string tag, CacheGroupType type, uint32_t block_num, size_t physical_tokens, size_t kernel_tokens) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = physical_tokens;
    CacheGroup group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.block_num                 = block_num;
    group.seq_size_per_block        = physical_tokens;
    group.kernel_seq_size_per_block = kernel_tokens;
    group.policy.group_type         = type;
    group.layer_ids                 = {0};
    return group;
}

CacheConfig makeHeterogeneousConfig(bool reversed) {
    test::TestCacheConfigBuilder builder;
    auto                         full = makeGroup("full", CacheGroupType::FULL, 3, 512, 256);
    auto                         swa  = makeGroup("swa", CacheGroupType::LINEAR, 2, 64, 64);
    if (reversed) {
        builder.addGroup(std::move(swa)).addGroup(std::move(full));
    } else {
        builder.addGroup(std::move(full)).addGroup(std::move(swa));
    }
    builder.setLayerTags(0, {"full", "swa"});
    return builder.build();
}

TEST(CacheBlockTablePackingTest, PacksHeterogeneousGroupsWithoutGlobalWidthPadding) {
    const auto plan = CacheBlockTablePackingPlan::create(makeHeterogeneousConfig(false), 2, 3);

    ASSERT_EQ(plan.groupCount(), 2);
    EXPECT_EQ(plan.poolNumel(), 10);
    EXPECT_EQ(plan.kernelNumel(), 16);

    const auto& full = plan.group(0);
    EXPECT_EQ(full.tag, "full");
    EXPECT_EQ(full.execution_ordinal, 0);
    EXPECT_EQ(full.pool.offset, 0);
    EXPECT_EQ(full.pool.row_width, 3);
    EXPECT_EQ(full.pool.numel(), 6);
    EXPECT_EQ(full.kernel.offset, 0);
    EXPECT_EQ(full.kernel.row_width, 6);
    EXPECT_EQ(full.kernel.numel(), 12);

    const auto& swa = plan.group(1);
    EXPECT_EQ(swa.tag, "swa");
    EXPECT_EQ(swa.execution_ordinal, 1);
    EXPECT_EQ(swa.pool.offset, 6);
    EXPECT_EQ(swa.pool.row_width, 2);
    EXPECT_EQ(swa.kernel.offset, 12);
    EXPECT_EQ(swa.kernel.row_width, 2);
}

TEST(CacheBlockTablePackingTest, ExecutionOrdinalsAreDeterministicAndViewsAliasBacking) {
    const auto plan_a = CacheBlockTablePackingPlan::create(makeHeterogeneousConfig(false), 2, 3);
    const auto plan_b = CacheBlockTablePackingPlan::create(makeHeterogeneousConfig(true), 2, 3);
    EXPECT_EQ(plan_a.group(0).tag, plan_b.group(0).tag);
    EXPECT_EQ(plan_a.group(1).tag, plan_b.group(1).tag);

    auto backing = torch::full({static_cast<int64_t>(plan_a.poolNumel())}, NULL_BLOCK_IDX, torch::kInt32);
    auto full    = plan_a.poolView(backing, 0);
    auto swa     = plan_a.poolView(backing, 1);
    ASSERT_TRUE(full.is_contiguous());
    ASSERT_TRUE(swa.is_contiguous());
    EXPECT_EQ(full.sizes(), (c10::IntArrayRef{2, 3}));
    EXPECT_EQ(swa.sizes(), (c10::IntArrayRef{2, 2}));
    full[0][0] = 0;
    EXPECT_EQ(backing[0].item<int32_t>(), 0);
    EXPECT_EQ(static_cast<char*>(swa.data_ptr()) - static_cast<char*>(backing.data_ptr()),
              static_cast<ptrdiff_t>(plan_a.group(1).pool.offset * sizeof(int32_t)));
}

TEST(CacheBlockTablePackingTest, AcceptsExactRuntimeWidthsByExecutionOrdinal) {
    const auto plan = CacheBlockTablePackingPlan::create(makeHeterogeneousConfig(false), 2, std::vector<size_t>{2, 1});

    EXPECT_EQ(plan.group(0).tag, "full");
    EXPECT_EQ(plan.group(0).pool.row_width, 2);
    EXPECT_EQ(plan.group(0).kernel.row_width, 4);
    EXPECT_EQ(plan.group(1).tag, "swa");
    EXPECT_EQ(plan.group(1).pool.row_width, 1);
    EXPECT_EQ(plan.group(1).kernel.row_width, 1);
    EXPECT_EQ(plan.poolNumel(), 6);
    EXPECT_EQ(plan.kernelNumel(), 10);
}

TEST(CacheBlockTablePackingTest, RejectsOverflowAndOrdinalOutOfRange) {
    const auto config = makeHeterogeneousConfig(false);
    EXPECT_THROW(CacheBlockTablePackingPlan::create(config, std::numeric_limits<size_t>::max(), 3), RTPException);
    const auto plan = CacheBlockTablePackingPlan::create(config, 2, 3);
    EXPECT_THROW((void)plan.group(2), RTPException);
}

TEST(CacheBlockTablePackingTest, TpKernelValidLengthsPreserveHeterogeneousRootTailsOnNonRoot) {
    const auto                 plan    = CacheBlockTablePackingPlan::create(makeHeterogeneousConfig(false), 2, 3);
    const auto                 options = torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    std::vector<torch::Tensor> root_lengths{
        torch::tensor({6, 2}, options),
        torch::tensor({1, 0}, options),
    };
    std::vector<torch::Tensor> non_root_lengths{
        torch::full({2}, 6, options),
        torch::full({2}, 2, options),
    };

    auto wire = packKernelValidLengthsForTp(root_lengths, plan);
    unpackKernelValidLengthsFromTp(wire, non_root_lengths, plan);

    EXPECT_TRUE(torch::equal(non_root_lengths[0], root_lengths[0]));
    EXPECT_TRUE(torch::equal(non_root_lengths[1], root_lengths[1]));
}

TEST(CacheBlockTablePackingTest, TpWirePreservesCompleteSignatureAndMtpRanksChooseTheSameBranch) {
    const auto root_current     = CacheBlockTablePackingPlan::create(makeHeterogeneousConfig(false), 2, 3);
    const auto wire             = encodeCacheBlockTablePackingPlan(root_current);
    const auto non_root_current = decodeCacheBlockTablePackingPlan(wire);

    ASSERT_EQ(non_root_current.groupCount(), 2);
    EXPECT_EQ(non_root_current.group(0).tag, "full");
    EXPECT_EQ(non_root_current.group(1).tag, "swa");
    EXPECT_TRUE(CacheBlockTablePackingSignature::fromPlan(root_current).matches(non_root_current));

    const auto unchanged              = CacheBlockTablePackingPlan::create(makeHeterogeneousConfig(true), 2, 3);
    const bool root_requires_full     = !CacheBlockTablePackingSignature::fromPlan(root_current).matches(unchanged);
    const bool non_root_requires_full = !CacheBlockTablePackingSignature::fromPlan(non_root_current).matches(unchanged);
    EXPECT_FALSE(root_requires_full);
    EXPECT_EQ(non_root_requires_full, root_requires_full);

    const auto grown = CacheBlockTablePackingPlan::create(makeHeterogeneousConfig(true), 2, std::vector<size_t>{4, 2});
    const bool root_growth_requires_full = !CacheBlockTablePackingSignature::fromPlan(root_current).matches(grown);
    const bool non_root_growth_requires_full =
        !CacheBlockTablePackingSignature::fromPlan(non_root_current).matches(grown);
    EXPECT_TRUE(root_growth_requires_full);
    EXPECT_EQ(non_root_growth_requires_full, root_growth_requires_full);
}

}  // namespace
}  // namespace rtp_llm
