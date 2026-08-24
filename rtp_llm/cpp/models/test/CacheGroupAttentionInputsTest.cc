#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/BlockExpression.h"
#include "rtp_llm/cpp/models/CacheGroupAttentionInputs.h"

namespace rtp_llm {

namespace {

CacheBlockTablePackingPlan makeTwoGroupPlan() {
    CacheGroupBlockTableRegion full;
    full.tag                   = "default";
    full.execution_ordinal     = 0;
    full.pool.offset           = 0;
    full.pool.row_width        = 8;
    full.pool.batch_capacity   = 4;
    full.kernel.offset         = 0;
    full.kernel.row_width      = 16;
    full.kernel.batch_capacity = 4;

    CacheGroupBlockTableRegion linear;
    linear.tag                   = "linear";
    linear.execution_ordinal     = 1;
    linear.pool.offset           = 32;
    linear.pool.row_width        = 2;
    linear.pool.batch_capacity   = 4;
    linear.kernel.offset         = 64;
    linear.kernel.row_width      = 2;
    linear.kernel.batch_capacity = 4;

    return CacheBlockTablePackingPlan::fromRegions({full, linear});
}

PackedBlockTableStorage makeStorage() {
    const auto              opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    PackedBlockTableStorage storage;
    storage.pool_host     = torch::arange(40, opts);
    storage.pool_device   = torch::arange(40, opts);
    storage.kernel_host   = torch::arange(72, opts);
    storage.kernel_device = torch::arange(72, opts);
    return storage;
}

torch_ext::PyAttentionInputs makeBase() {
    torch_ext::PyAttentionInputs base;
    base.input_lengths    = torch::ones({4}, torch::TensorOptions().dtype(torch::kInt32));
    base.sequence_lengths = torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt32));
    return base;
}

}  // namespace

TEST(CacheGroupAttentionInputsTest, KeysAreTagsAndTablesUseGroupGeometry) {
    const auto plan    = makeTwoGroupPlan();
    const auto storage = makeStorage();
    const auto base    = makeBase();

    std::vector<CacheGroupBinding> bindings;
    const auto                     group_inputs = bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    ASSERT_EQ(group_inputs.size(), 2u);
    ASSERT_EQ(group_inputs.count("default"), 1u);
    ASSERT_EQ(group_inputs.count("linear"), 1u);

    const auto& full = group_inputs.at("default");
    EXPECT_EQ(full.kv_cache_block_id.size(0), 4);
    EXPECT_EQ(full.kv_cache_block_id.size(1), 8);
    EXPECT_EQ(full.kv_cache_kernel_block_id.size(0), 4);
    EXPECT_EQ(full.kv_cache_kernel_block_id.size(1), 16);

    const auto& linear = group_inputs.at("linear");
    EXPECT_EQ(linear.kv_cache_block_id.size(1), 2);
    EXPECT_EQ(linear.kv_cache_kernel_block_id.size(1), 2);
}

TEST(CacheGroupAttentionInputsTest, SharedFieldsAliasBaseStorage) {
    const auto plan    = makeTwoGroupPlan();
    const auto storage = makeStorage();
    const auto base    = makeBase();

    std::vector<CacheGroupBinding> bindings;
    const auto                     group_inputs = bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    for (const std::string tag : {"default", "linear"}) {
        const auto& group = group_inputs.at(tag);
        EXPECT_EQ(group.input_lengths.data_ptr(), base.input_lengths.data_ptr());
        EXPECT_EQ(group.sequence_lengths.data_ptr(), base.sequence_lengths.data_ptr());
    }
}

TEST(CacheGroupAttentionInputsTest, TablesAliasPackedBackingSoWritesAreVisible) {
    const auto plan    = makeTwoGroupPlan();
    auto       storage = makeStorage();
    const auto base    = makeBase();

    std::vector<CacheGroupBinding> bindings;
    const auto                     group_inputs = bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    // Writing the backing must be visible through every group view: that is what
    // lets value-only update copy once and stay correct for all groups.
    storage.kernel_host.fill_(7);
    EXPECT_EQ(group_inputs.at("linear").kv_cache_kernel_block_id.index({0, 0}).item<int32_t>(), 7);
    EXPECT_EQ(group_inputs.at("default").kv_cache_kernel_block_id.index({0, 0}).item<int32_t>(), 7);

    const auto* before = group_inputs.at("default").kv_cache_kernel_block_id.data_ptr<int32_t>();
    storage.kernel_host.fill_(9);
    const auto* after = group_inputs.at("default").kv_cache_kernel_block_id.data_ptr<int32_t>();
    EXPECT_EQ(before, after);
}

TEST(CacheGroupAttentionInputsTest, LinearSegmentStartsAtPlanOffset) {
    const auto plan    = makeTwoGroupPlan();
    const auto storage = makeStorage();
    const auto base    = makeBase();

    std::vector<CacheGroupBinding> bindings;
    const auto                     group_inputs = bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    // storage.kernel_host is arange(72); the linear segment starts at element 64.
    EXPECT_EQ(group_inputs.at("linear").kv_cache_kernel_block_id.index({0, 0}).item<int32_t>(), 64);
    EXPECT_EQ(group_inputs.at("default").kv_cache_kernel_block_id.index({0, 0}).item<int32_t>(), 0);
}

TEST(CacheGroupAttentionInputsTest, BindingsCarryOrdinalAndTag) {
    const auto plan    = makeTwoGroupPlan();
    const auto storage = makeStorage();
    const auto base    = makeBase();

    std::vector<CacheGroupBinding> bindings;
    bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    ASSERT_EQ(bindings.size(), 2u);
    EXPECT_EQ(bindings[0].execution_ordinal, 0u);
    EXPECT_EQ(bindings[0].tag, "default");
    EXPECT_EQ(bindings[1].execution_ordinal, 1u);
    EXPECT_EQ(bindings[1].tag, "linear");
}

TEST(CacheGroupAttentionInputsTest, AdoptRejectsNonFlatBacking) {
    const auto plan = makeTwoGroupPlan();
    const auto i32  = torch::TensorOptions().dtype(torch::kInt32);

    PackedBlockTableStorage storage;
    auto                    two_dim = torch::zeros({4, 10}, i32);
    auto                    flat    = torch::zeros({72}, i32);

    EXPECT_THROW(adoptPackedBlockTables(plan, two_dim, two_dim, flat, flat, storage), std::exception);
}

TEST(CacheGroupAttentionInputsTest, AdoptRejectsBackingSmallerThanPlan) {
    const auto plan = makeTwoGroupPlan();
    const auto i32  = torch::TensorOptions().dtype(torch::kInt32);

    PackedBlockTableStorage storage;
    auto                    pool  = torch::zeros({40}, i32);
    auto                    small = torch::zeros({8}, i32);  // plan needs 72

    EXPECT_THROW(adoptPackedBlockTables(plan, pool, pool, small, small, storage), std::exception);
}

TEST(CacheGroupAttentionInputsTest, AdoptSharesStorageWithoutCopying) {
    const auto plan = makeTwoGroupPlan();
    const auto i32  = torch::TensorOptions().dtype(torch::kInt32);

    PackedBlockTableStorage storage;
    auto                    pool   = torch::zeros({40}, i32);
    auto                    kernel = torch::zeros({72}, i32);
    adoptPackedBlockTables(plan, pool, pool, kernel, kernel, storage);

    // Adoption must not copy: the caller's storage is what we hold.
    EXPECT_EQ(storage.kernel_host.data_ptr(), kernel.data_ptr());
    EXPECT_EQ(storage.pool_host.data_ptr(), pool.data_ptr());
}

TEST(CacheGroupAttentionInputsTest, TailFillClearsBeyondValidLength) {
    const auto plan    = makeTwoGroupPlan();
    auto       storage = makeStorage();
    const auto base    = makeBase();

    std::vector<CacheGroupBinding> bindings;
    auto                           group_inputs = bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    const auto i32                                  = torch::TensorOptions().dtype(torch::kInt32);
    group_inputs.at("default").kernel_valid_lengths = torch::full({4}, 16, i32);
    group_inputs.at("linear").kernel_valid_lengths  = torch::ones({4}, i32);  // row_width is 2
    group_inputs.at("linear").kv_cache_kernel_block_id.fill_(5);

    normalizeKernelTailFill(group_inputs, bindings, torch::Tensor(), torch::Tensor(), 0);

    const auto linear_table = group_inputs.at("linear").kv_cache_kernel_block_id;
    EXPECT_EQ(linear_table.index({0, 0}).item<int32_t>(), 5);
    EXPECT_EQ(linear_table.index({0, 1}).item<int32_t>(), NULL_BLOCK_IDX);
    EXPECT_EQ(linear_table.index({3, 1}).item<int32_t>(), NULL_BLOCK_IDX);
}

TEST(CacheGroupAttentionInputsTest, TailFillRejectsValidLengthAboveRowWidth) {
    const auto plan    = makeTwoGroupPlan();
    auto       storage = makeStorage();
    const auto base    = makeBase();

    std::vector<CacheGroupBinding> bindings;
    auto                           group_inputs = bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    const auto i32                                  = torch::TensorOptions().dtype(torch::kInt32);
    group_inputs.at("default").kernel_valid_lengths = torch::full({4}, 16, i32);
    group_inputs.at("linear").kernel_valid_lengths  = torch::full({4}, 3, i32);  // row_width is 2

    EXPECT_THROW(normalizeKernelTailFill(group_inputs, bindings, torch::Tensor(), torch::Tensor(), 0), std::exception);
}

TEST(CacheGroupAttentionInputsTest, RefreshKeepsDataPtrAndUpdatesValues) {
    const auto plan    = makeTwoGroupPlan();
    auto       storage = makeStorage();
    const auto base    = makeBase();

    std::vector<CacheGroupBinding> bindings;
    auto                           group_inputs = bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    const auto i32                                  = torch::TensorOptions().dtype(torch::kInt32);
    group_inputs.at("default").kernel_valid_lengths = torch::full({4}, 16, i32);
    group_inputs.at("linear").kernel_valid_lengths  = torch::full({4}, 2, i32);

    const auto* before = group_inputs.at("linear").kv_cache_kernel_block_id.data_ptr<int32_t>();

    auto                       fresh_host   = torch::full({72}, 3, i32);
    auto                       fresh_device = torch::full({72}, 3, i32);
    std::vector<torch::Tensor> fresh_lengths{torch::full({4}, 16, i32), torch::full({4}, 2, i32)};
    refreshPackedBlockTableValues(fresh_host, fresh_device, fresh_lengths, storage, bindings, group_inputs);

    const auto* after = group_inputs.at("linear").kv_cache_kernel_block_id.data_ptr<int32_t>();
    EXPECT_EQ(before, after);
    EXPECT_EQ(group_inputs.at("linear").kv_cache_kernel_block_id.index({0, 0}).item<int32_t>(), 3);
    EXPECT_EQ(group_inputs.at("default").kv_cache_kernel_block_id.index({0, 0}).item<int32_t>(), 3);
}

TEST(CacheGroupAttentionInputsTest, RefreshRejectsShapeChange) {
    const auto plan    = makeTwoGroupPlan();
    auto       storage = makeStorage();
    const auto base    = makeBase();

    std::vector<CacheGroupBinding> bindings;
    auto                           group_inputs = bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    const auto                 i32          = torch::TensorOptions().dtype(torch::kInt32);
    auto                       wrong_host   = torch::full({8}, 3, i32);
    auto                       wrong_device = torch::full({8}, 3, i32);
    std::vector<torch::Tensor> lengths{torch::full({4}, 16, i32), torch::full({4}, 2, i32)};

    EXPECT_THROW(refreshPackedBlockTableValues(wrong_host, wrong_device, lengths, storage, bindings, group_inputs),
                 std::exception);
}

TEST(CacheGroupAttentionInputsTest, EmptyPlanProducesNoGroups) {
    // No KV cache (embedding models, warmup): the pure function yields nothing and
    // PyWrappedModel::bindCacheGroups short-circuits before reaching here.
    const auto base = makeBase();
    const auto plan = CacheBlockTablePackingPlan::fromRegions({});
    ASSERT_EQ(plan.groupCount(), 0u);

    const auto                     storage = makeStorage();
    std::vector<CacheGroupBinding> bindings;
    const auto                     group_inputs = bindCacheGroupAttentionInputs(base, plan, storage, bindings);

    EXPECT_TRUE(group_inputs.empty());
    EXPECT_TRUE(bindings.empty());
}

}  // namespace rtp_llm
