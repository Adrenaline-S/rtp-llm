#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

static CacheConfig makeTestConfig(int block_num = 20, int seq_size_per_block = 4) {
    return makeSimpleMhaCacheConfig(
        /*layer_num=*/2,
        block_num,
        /*tokens_per_block=*/static_cast<size_t>(seq_size_per_block),
        rtp_llm::DataType::TYPE_FP16,
        /*local_head_num_kv=*/1,
        /*size_per_head=*/16);
}

static CompleteTokenIdsPtr makeTokenIds(int batch_size, int seq_len, int block_size) {
    auto  ids       = std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_len + 100, block_size);
    auto  input_ids = torch::empty({(int64_t)seq_len}, torch::kInt32);
    auto* ptr       = input_ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_len; ++i)
        ptr[i] = i + 1;
    auto gi             = std::make_shared<GenerateInput>();
    gi->input_ids       = input_ids;
    gi->generate_config = std::make_shared<GenerateConfig>();
    ids->init(gi);
    return ids;
}

static BatchKVCacheResourcePtr makeResource(int batch_size, int layer_num) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    std::vector<std::vector<int>> layer_group_ids(static_cast<size_t>(layer_num), std::vector<int>{0});
    res->initGroups(makeTestCacheTopology(/*group_num=*/1, layer_num, layer_group_ids));
    return res;
}

static ParallelismConfig makePrefillConfig(int tp_size, int tp_rank = 0, bool sharded = true) {
    ParallelismConfig par;
    par.role_type                           = RoleType::PREFILL;
    par.tp_size                             = tp_size;
    par.tp_rank                             = tp_rank;
    par.prefill_cp_config.method            = CPRotateMethod::ALL_GATHER;
    par.prefill_cp_config.kv_cache_sharded = sharded;
    return par;
}

static ParallelismConfig makeDecodeConfig(int decode_tp_size, int prefill_cp_size) {
    ParallelismConfig par;
    par.role_type                            = RoleType::DECODE;
    par.tp_size                              = decode_tp_size;
    par.prefill_cp_config.method             = CPRotateMethod::PREFILL_CP;
    par.prefill_cp_config.kv_cache_sharded = true;
    par.prefill_cp_config.prefill_cp_size  = prefill_cp_size;
    return par;
}

static std::shared_ptr<KVCacheManager> makeInitializedManager(const CacheConfig&       config,
                                                              const ParallelismConfig& par,
                                                              const KVCacheConfig&     kv_config = {}) {
    auto manager = std::make_shared<KVCacheManager>(config, /*warmup=*/true, nullptr, kv_config, par);
    // warmup mode avoids collective block-count synchronization. Restore the
    // requested test capacity before creating the rank-local allocator.
    manager->config_ = config;
    RTP_LLM_CHECK(manager->init());
    return manager;
}

class KVCacheManagerCPSlotMapperTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// When kv_cache_sharded is false (default), cpSlotMapper() should return nullptr.
TEST_F(KVCacheManagerCPSlotMapperTest, NoCPSharding_ReturnsNullMapper) {
    auto              config = makeTestConfig();
    ParallelismConfig par    = makePrefillConfig(/*tp_size=*/2, /*tp_rank=*/0, /*sharded=*/false);

    // warmup=true skips allocateAndSync (which would NCCL all-gather across the
    // tp_size process group; in single-process UT there are no peers).  cp_slot_mapper_
    // is constructed regardless of warmup, so cpSlotMapper() check is unaffected.
    auto mgr = makeInitializedManager(config, par);

    EXPECT_EQ(mgr->cpSlotMapper(), nullptr);
    EXPECT_EQ(mgr->cacheKeyCpSize(), 1);
}

TEST_F(KVCacheManagerCPSlotMapperTest, InvalidShardingConfigurationsAreRejected) {
    auto pdfusion = makePrefillConfig(/*tp_size=*/4);
    pdfusion.role_type = RoleType::PDFUSION;
    EXPECT_THROW(pdfusion.validateKvCacheSharding(), std::invalid_argument);

    auto wrong_prefill_method = makePrefillConfig(/*tp_size=*/4);
    wrong_prefill_method.prefill_cp_config.method = CPRotateMethod::PREFILL_CP;
    EXPECT_THROW(wrong_prefill_method.validateKvCacheSharding(), std::invalid_argument);

    auto invalid_prefill_size = makePrefillConfig(/*tp_size=*/1);
    EXPECT_THROW(invalid_prefill_size.validateKvCacheSharding(), std::invalid_argument);

    auto wrong_decode_method = makeDecodeConfig(/*decode_tp_size=*/2, /*prefill_cp_size=*/4);
    wrong_decode_method.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_THROW(wrong_decode_method.validateKvCacheSharding(), std::invalid_argument);

    auto invalid_decode_size = makeDecodeConfig(/*decode_tp_size=*/2, /*prefill_cp_size=*/1);
    EXPECT_THROW(invalid_decode_size.validateKvCacheSharding(), std::invalid_argument);
}

// When kv_cache_sharded is true and tp_size > 1, cpSlotMapper() should return a valid mapper.
TEST_F(KVCacheManagerCPSlotMapperTest, CPShardingEnabled_ReturnsValidMapper) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;
    par = makePrefillConfig(/*tp_size=*/4, /*tp_rank=*/3);
    ASSERT_NO_THROW(par.validateKvCacheSharding());

    // warmup=true skips allocateAndSync (which would NCCL all-gather across the
    // tp_size process group; in single-process UT there are no peers).  cp_slot_mapper_
    // is constructed regardless of warmup, so cpSlotMapper() check is unaffected.
    auto mgr = makeInitializedManager(config, par);

    auto mapper = mgr->cpSlotMapper();
    ASSERT_NE(mapper, nullptr);
    EXPECT_TRUE(mapper->isSharded());
    EXPECT_EQ(mapper->cpRank(), 3);
    EXPECT_EQ(mapper->cpSize(), 4);
    EXPECT_EQ(mapper->blockSize(), seq_size_per_block);
    EXPECT_EQ(mapper->virtualBlockSize(), seq_size_per_block * 4);
    EXPECT_EQ(mgr->cacheKeyCpSize(), 4);
}

TEST_F(KVCacheManagerCPSlotMapperTest, DecodeUsesPrefillCp4KeysWithoutLocalSharding) {
    auto config = makeTestConfig();
    for (int decode_tp_size : {1, 2}) {
        auto par = makeDecodeConfig(decode_tp_size, /*prefill_cp_size=*/4);
        ASSERT_NO_THROW(par.validateKvCacheSharding());
        auto mgr = makeInitializedManager(config, par);
        EXPECT_EQ(mgr->cpSlotMapper(), nullptr) << "decode_tp_size=" << decode_tp_size;
        EXPECT_EQ(mgr->cacheKeyCpSize(), 4) << "decode_tp_size=" << decode_tp_size;
        EXPECT_EQ(mgr->allocator_->cpSlotMapper(), nullptr) << "decode_tp_size=" << decode_tp_size;
    }
}

TEST_F(KVCacheManagerCPSlotMapperTest, CPShardingEnabled_CacheInfoReportsVirtualBlockSize) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par = makePrefillConfig(/*tp_size=*/4);

    auto mgr = makeInitializedManager(config, par);

    auto info = mgr->getKVCacheInfo(/*latest_version=*/-1, /*need_cache_keys=*/false);
    EXPECT_EQ(info.block_size, static_cast<size_t>(seq_size_per_block * par.tp_size));
}

// Partial tails may be allocated as live KV blocks before they become cacheable
// full blocks. CP invariants must therefore be based on logical sequence length,
// not cacheKeys().size().
TEST_F(KVCacheManagerCPSlotMapperTest, CPShardedMallocAllowsPartialTailWithoutCacheKey) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par;

    auto mgr = std::make_shared<KVCacheManager>(config, /*warmup=*/false, nullptr, KVCacheConfig{}, par);
    ASSERT_TRUE(mgr->init());

    auto resource  = makeResource(1, config.layer_num);
    auto token_ids = makeTokenIds(1, /*seq_len=*/1, seq_size_per_block);

    MallocInfo info{resource, token_ids};
    auto       cp_mapper = std::make_shared<CPSlotMapper>(0, 2, seq_size_per_block);
    mgr->cp_slot_mapper_ = cp_mapper;
    mgr->allocator_->setCPSlotMapper(cp_mapper);

    auto result = mgr->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(resource->blocksNum(0, 0), 1);

    token_ids->setSeqLength(2);
    result = mgr->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(resource->blocksNum(0, 0), 1);
    EXPECT_EQ(resource->cacheKeys(0).size(), 0);
}

// malloc() should use the manager-level cpSlotMapper.
// With CP sharding (cp_size=2, block_size=4), virtual_block_size=8.
// A sequence of 16 tokens needs ceil(16/8)=2 physical blocks per batch (not 4).
TEST_F(KVCacheManagerCPSlotMapperTest, MallocAutoInjectReducesBlockCount) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par = makePrefillConfig(/*tp_size=*/2);

    // warmup=true skips allocateAndSync (which would NCCL all-gather across the
    // tp_size process group; in single-process UT there are no peers).  cp_slot_mapper_
    // is constructed regardless of warmup, so cpSlotMapper() check is unaffected.
    auto mgr = makeInitializedManager(config, par);

    const int seq_len   = 16;
    auto      resource  = makeResource(1, config.layer_num);
    auto      token_ids = makeTokenIds(1, seq_len, seq_size_per_block);

    MallocInfo info{resource, token_ids};
    auto       result = mgr->malloc(info);
    ASSERT_TRUE(result.success);

    // virtual_block_size = 4 * 2 = 8
    // effectiveSeqLenForAlloc(16) = ceil(16/8) * 4 = 8 tokens worth => ceil(8/4) = 2 blocks
    EXPECT_EQ(resource->blocksNum(0, 0), 2);
}

// Without CP sharding, the same seq_len should allocate more blocks.
TEST_F(KVCacheManagerCPSlotMapperTest, MallocWithoutCPAllocatesFullBlocks) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par = makePrefillConfig(/*tp_size=*/2, /*tp_rank=*/0, /*sharded=*/false);

    // warmup=true skips allocateAndSync (which would NCCL all-gather across the
    // tp_size process group; in single-process UT there are no peers).  cp_slot_mapper_
    // is constructed regardless of warmup, so cpSlotMapper() check is unaffected.
    auto mgr = makeInitializedManager(config, par);

    const int seq_len   = 16;
    auto      resource  = makeResource(1, config.layer_num);
    auto      token_ids = makeTokenIds(1, seq_len, seq_size_per_block);

    MallocInfo info{resource, token_ids};
    auto       result = mgr->malloc(info);
    ASSERT_TRUE(result.success);

    // Without CP: ceil(16/4) = 4 blocks
    EXPECT_EQ(resource->blocksNum(0, 0), 4);
}

// Allocator-level cp_slot_mapper should drive malloc sharding.
TEST_F(KVCacheManagerCPSlotMapperTest, AllocatorMapperControlsMalloc) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/30, seq_size_per_block);

    ParallelismConfig par = makePrefillConfig(/*tp_size=*/2);

    // warmup=true skips allocateAndSync (which would NCCL all-gather across the
    // tp_size process group; in single-process UT there are no peers).  cp_slot_mapper_
    // is constructed regardless of warmup, so cpSlotMapper() check is unaffected.
    auto mgr = makeInitializedManager(config, par);

    const int seq_len   = 64;
    auto      resource  = makeResource(1, config.layer_num);
    auto      token_ids = makeTokenIds(1, seq_len, seq_size_per_block);

    auto explicit_mapper = std::make_shared<CPSlotMapper>(0, 4, seq_size_per_block);
    // virtual_block_size = 4 * 4 = 16
    // effectiveSeqLenForAlloc(64) = ceil(64/16)*4 = 16 tokens => ceil(16/4) = 4 blocks

    MallocInfo info{resource, token_ids};
    mgr->cp_slot_mapper_ = explicit_mapper;
    mgr->allocator_->setCPSlotMapper(explicit_mapper);
    auto result = mgr->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(resource->blocksNum(0, 0), 4);
}

// insertIntoCache() should also use the manager-level mapper.
TEST_F(KVCacheManagerCPSlotMapperTest, InsertAutoInjectsMapper) {
    const int seq_size_per_block = 4;
    auto      config             = makeTestConfig(/*block_num=*/20, seq_size_per_block);

    ParallelismConfig par = makePrefillConfig(/*tp_size=*/2);

    KVCacheConfig kv_cfg;
    kv_cfg.reuse_cache         = true;
    kv_cfg.enable_device_cache = true;

    auto mgr = makeInitializedManager(config, par, kv_cfg);
    // virtual_block_size = 4 * 2 = 8
    // effectiveSeqLenForAlloc(16) = ceil(16/8) * 4 = 8 tokens worth => ceil(8/4) = 2 blocks

    const int seq_len   = 16;
    auto      resource  = makeResource(1, config.layer_num);
    auto      token_ids = makeTokenIds(1, seq_len, seq_size_per_block);

    MallocInfo malloc_info{resource, token_ids};
    malloc_info.reuse_cache         = true;
    malloc_info.enable_device_cache = true;
    auto result                     = mgr->malloc(malloc_info);
    ASSERT_TRUE(result.success);

    // Insert into cache using the allocator-level cp_slot_mapper.
    // This should not crash and should use sharded insert logic.
    InsertInfo insert_info{resource, token_ids, /*is_resident=*/false};
    EXPECT_NO_THROW(mgr->insertIntoCache(insert_info));

    // Now try to malloc again with the same token_ids -- should get reuse hit.
    auto       resource2 = makeResource(1, config.layer_num);
    MallocInfo malloc_info2{resource2, token_ids};
    malloc_info2.reuse_cache         = true;
    malloc_info2.enable_device_cache = true;
    auto result2                     = mgr->malloc(malloc_info2);
    ASSERT_TRUE(result2.success);
    // With CP sharding (cp_size=2, block_size=4), virtual_block_size=8.
    // seq_len=16 produces 2 cache keys (each covering 8 tokens).
    // match drops the last key → 1 matched key → reuse_len = 1 * virtual_block_size = 8.
    // The sharded reuse_length adjustment ensures this is 1 * virtual_block_size = 8, not 1 * seq_size_per_block = 4.
    EXPECT_EQ(result2.reuse_len, seq_size_per_block * par.tp_size);  // = 4 * 2 = 8
}

}  // namespace test
}  // namespace rtp_llm
