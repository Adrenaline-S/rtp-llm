#include <algorithm>
#include <functional>
#include <map>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {

namespace {

DecodeRpcServer::LoadKVCacheContext makeLoadContext(const std::string&               request_key,
                                                    const std::vector<std::string>&  peer_addrs,
                                                    const std::vector<CacheKeyType>& cache_keys,
                                                    const CacheGroupBlockRecords&    group_blocks,
                                                    int32_t                          prefill_cp_size,
                                                    int64_t                          reuse_block_size = 0) {
    return {/*request_id=*/42,
            request_key,
            peer_addrs,
            cache_keys,
            group_blocks,
            reuse_block_size,
            /*timeout_ms=*/1000,
            /*partition_count=*/1,
            /*partition_id=*/0,
            /*server_context=*/nullptr,
            prefill_cp_size};
}

std::shared_ptr<CacheGroupBlocks> makeGroupRecord(std::string tag, BlockIndicesType blocks) {
    auto record = std::make_shared<CacheGroupBlocks>();
    record->tag = std::move(tag);
    record->blocks.assign(std::move(blocks));
    return record;
}

std::map<std::string, BlockIndicesType> taggedRowsOf(const BroadcastLoadRequestPB& request) {
    std::map<std::string, BlockIndicesType> rows;
    for (const auto& row : request.tagged_group_block_ids()) {
        auto [it, inserted] = rows.emplace(row.tag(), BlockIndicesType(row.block_ids().begin(), row.block_ids().end()));
        EXPECT_TRUE(inserted) << "duplicate wire tag=" << row.tag();
    }
    return rows;
}

std::map<std::string, BlockIndicesType> taggedRecordsOf(const CacheGroupBlockRecords& records) {
    std::map<std::string, BlockIndicesType> rows;
    for (const auto& record : records) {
        EXPECT_NE(record, nullptr);
        auto [it, inserted] = rows.emplace(record->tag, record->blocks.blocks());
        EXPECT_TRUE(inserted) << "duplicate record tag=" << record->tag;
    }
    return rows;
}

GroupBase makeRpcGroup(std::string tag, std::vector<int> layer_ids) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 8;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = 8;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = 8;
    return group;
}

CacheConfig makeRpcCacheConfig() {
    CacheConfig config;
    config.layer_num = 2;
    config.setTopology({makeRpcGroup("linear", {0}), makeRpcGroup("full", {1})}, {{0, {"linear"}}, {1, {"full"}}});
    return config;
}

// DeepSeek-V4 shaped identity fixture: one layer owning every semantic group.
// Only tag identity matters at the RPC boundary, so the physical layout is the
// simple shared MHA layout used by the other RPC fixtures.
const std::vector<std::string>& dsv4RpcTags() {
    static const std::vector<std::string> tags = {
        "swa_kv", "csa_kv", "indexer_kv", "indexer_state", "csa_state", "hca_kv", "hca_state"};
    return tags;
}

std::shared_ptr<const CacheTopology> makeDsv4RpcTopology(bool reversed) {
    auto tags = dsv4RpcTags();
    if (reversed) {
        std::reverse(tags.begin(), tags.end());
    }
    std::vector<GroupBase> groups;
    groups.reserve(tags.size());
    for (const auto& tag : tags) {
        groups.push_back(makeRpcGroup(tag, {0}));
    }
    LayerBase layer;
    layer.layer_id   = 0;
    layer.group_tags = tags;
    return CacheTopology::create(std::move(groups), {std::move(layer)});
}

// One cache group's physical geometry, so a test can vary exactly one of
// heads / physical B / K / stride at a time.
struct RpcGroupGeometry {
    uint32_t local_kv_head_num         = 2;
    size_t   seq_size_per_block        = 8;
    size_t   kernel_seq_size_per_block = 8;
    size_t   kv_block_stride_bytes     = 256;
};

CacheConfig makeRpcGeometryConfig(const RpcGroupGeometry& geometry) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = geometry.seq_size_per_block;

    GroupBase group;
    group.tag                       = "full";
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = {0};
    group.block_num                 = 8;
    group.local_kv_head_num         = geometry.local_kv_head_num;
    group.seq_size_per_block        = geometry.seq_size_per_block;
    group.kernel_seq_size_per_block = geometry.kernel_seq_size_per_block;
    group.kv_block_stride_bytes     = geometry.kv_block_stride_bytes;

    CacheConfig config;
    config.layer_num = 1;
    config.setTopology({std::move(group)}, {{0, {"full"}}});
    return config;
}

// The cache layout an allocator materializes from `allocator_plan`: one dense
// all-layer entry per group tag. Only the topology matters to the physical
// layout guard, so the per-layer buffers stay undefined.
GroupedCacheLayerLayout makeMaterializedLayout(const CacheConfig& allocator_plan) {
    auto                                  topology = allocator_plan.topologyPtr();
    GroupedCacheLayerLayout::GroupLayouts groups;
    for (const auto& group : topology->groups()) {
        groups.emplace(group.tag, CacheLayerLayout(std::vector<BlockBufferPtrInfo>(topology->layers().size())));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
}

class DecodeBoundaryTestEngine final: public EngineBase {
public:
    explicit DecodeBoundaryTestEngine(const CacheConfig& config): EngineBase(EngineInitParams()) {
        resource_context_.cache_manager = std::make_shared<KVCacheManager>(config, /*warmup=*/true);
    }

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void         enqueue(std::shared_ptr<GenerateStream>&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("not used by DecodeRpcServerTest");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
};

class DecodeBoundaryTestStream final: public GenerateStream {
public:
    DecodeBoundaryTestStream():
        GenerateStream(makeInput(), makeModelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}

    ErrorResult<GenerateOutputs> nextOutput(int64_t = 0) override {
        return ErrorResult<GenerateOutputs>(GenerateOutputs{});
    }
    void updateOutput(const StreamUpdateInfo&) override {}

private:
    static std::shared_ptr<GenerateInput> makeInput() {
        auto input             = std::make_shared<GenerateInput>();
        input->generate_config = std::make_shared<GenerateConfig>();
        input->input_ids       = torch::zeros({1}, torch::kInt32);
        return input;
    }

    static ModelConfig makeModelConfig() {
        ModelConfig config;
        config.max_seq_len = 128;
        return config;
    }
};

}  // namespace

class DecodeRpcResourceBoundaryTest: public ::testing::Test {
protected:
    void SetUp() override {
        config_                   = makeRpcCacheConfig();
        server_.engine_           = std::make_shared<DecodeBoundaryTestEngine>(config_);
        server_.resource_.workers = {"decode-0", "decode-1"};

        stream_  = std::make_shared<DecodeBoundaryTestStream>();
        context_ = std::make_unique<DecodeGenerateContext>(
            rpc_context_, /*timeout_ms=*/0, /*server_context=*/nullptr, metrics_reporter_, /*meta=*/nullptr);
        context_->peer_addrs = {"prefill-0", "prefill-1", "prefill-2"};
        context_->stream_    = stream_;
    }

    void TearDown() override {
        context_->stream_.reset();
        context_.reset();
        stream_.reset();
    }

    BatchKVCacheResource makeBatchResource() const {
        BatchKVCacheResource batch;
        batch.resetBatchSize(1);
        batch.initGroups(config_);
        batch.setBatchCacheKeys(0, {101, 102});
        batch.setBatchBlocks(0, "linear", {20, 21});
        batch.setBatchBlocks(0, "full", {10, 11, 12});
        return batch;
    }

    ErrorInfo load(BatchKVCacheResource batch) {
        stream_->setKVCache(batch);
        return server_.loadCacheForAllRank(*context_);
    }

protected:
    CacheConfig                            config_;
    DecodeRpcServer                        server_;
    std::shared_ptr<GenerateStream>        stream_;
    DecodeRpcContext                       rpc_context_{nullptr};
    kmonitor::MetricsReporterPtr           metrics_reporter_;
    std::unique_ptr<DecodeGenerateContext> context_;
};

TEST(ModelRpcProtoTest, GroupedCacheFieldsPreserveLegacyNumbers) {
    const auto* broadcast = BroadcastLoadRequestPB::descriptor();
    ASSERT_NE(broadcast, nullptr);
    EXPECT_TRUE(broadcast->IsReservedNumber(5));
    EXPECT_TRUE(broadcast->IsReservedNumber(12));
    EXPECT_EQ(broadcast->FindFieldByName("block_num")->number(), 6);
    EXPECT_EQ(broadcast->FindFieldByName("reuse_block_size")->number(), 7);
    EXPECT_EQ(broadcast->FindFieldByName("timeout_ms")->number(), 8);
    EXPECT_EQ(broadcast->FindFieldByName("dp_rank")->number(), 9);
    EXPECT_EQ(broadcast->FindFieldByName("partition_count")->number(), 10);
    EXPECT_EQ(broadcast->FindFieldByName("partition_id")->number(), 11);
    EXPECT_EQ(broadcast->FindFieldByName("prefill_cp_size")->number(), 13);
    EXPECT_EQ(broadcast->FindFieldByName("tagged_group_block_ids")->number(), 14);

    const auto* remote = RemoteOperationRequestPB::descriptor();
    ASSERT_NE(remote, nullptr);
    EXPECT_TRUE(remote->IsReservedNumber(3));
    EXPECT_EQ(remote->FindFieldByName("group_ids"), nullptr);
    EXPECT_EQ(remote->FindFieldByName("block_ids")->number(), 4);
    EXPECT_EQ(remote->FindFieldByName("uris")->number(), 5);
    EXPECT_EQ(remote->FindFieldByName("group_tags")->number(), 6);
}

TEST(DecodeRpcServerTest, CPShardedLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101, 102};
    const CacheGroupBlockRecords    group_blocks{makeGroupRecord("full", {10, 11}),
                                              makeGroupRecord("linear", {20, 21})};
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, group_blocks, /*cp_size=*/2, /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/0, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
    ASSERT_EQ(request.cache_keys_size(), 2);
    EXPECT_EQ(request.cache_keys(0), 101);
    EXPECT_EQ(request.cache_keys(1), 102);
    EXPECT_EQ(taggedRowsOf(request), taggedRecordsOf(group_blocks));
}

TEST(DecodeRpcServerTest, CPShardedMlaLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101};
    const CacheGroupBlockRecords    group_blocks{makeGroupRecord("full", {10}),
                                              makeGroupRecord("indexer_kv", {30})};
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, group_blocks, /*cp_size=*/2, /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequestForMla(load_context, /*index=*/1, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
    EXPECT_EQ(taggedRowsOf(request), taggedRecordsOf(group_blocks));
}

TEST(DecodeRpcServerTest, LoadRequestRowsCarryRecordTagsIndependentOfRecordOrder) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0"};

    const std::vector<std::string>  peer_addrs = {"prefill-0"};
    const std::vector<CacheKeyType> cache_keys = {101, 102};
    const CacheGroupBlockRecords    ordered{makeGroupRecord("linear", {20, 21}), makeGroupRecord("full", {10, 11})};
    CacheGroupBlockRecords          reversed(ordered.rbegin(), ordered.rend());

    const auto ordered_context  = makeLoadContext("request", peer_addrs, cache_keys, ordered, /*cp_size=*/1);
    const auto reversed_context = makeLoadContext("request", peer_addrs, cache_keys, reversed, /*cp_size=*/1);

    const auto expected = std::map<std::string, BlockIndicesType>{{"full", {10, 11}}, {"linear", {20, 21}}};
    EXPECT_EQ(taggedRowsOf(server.constructRemoteLoadRequest(ordered_context, /*index=*/0, peer_addrs)), expected);
    EXPECT_EQ(taggedRowsOf(server.constructRemoteLoadRequest(reversed_context, /*index=*/0, peer_addrs)), expected);
    EXPECT_EQ(taggedRowsOf(server.constructRemoteLoadRequestForMla(ordered_context, /*index=*/0, peer_addrs)),
              expected);
    EXPECT_EQ(taggedRowsOf(server.constructRemoteLoadRequestForMla(reversed_context, /*index=*/0, peer_addrs)),
              expected);
}

TEST(DecodeRpcServerTest, Dsv4MultiTagRowsRoundTripThroughReversedLocalTopology) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0"};

    const std::vector<std::string>  peer_addrs = {"prefill-0"};
    const std::vector<CacheKeyType> cache_keys = {101};
    CacheGroupBlockRecords          records;
    std::map<std::string, BlockIndicesType> expected;
    for (size_t i = 0; i < dsv4RpcTags().size(); ++i) {
        const auto&            tag    = dsv4RpcTags()[i];
        const BlockIndicesType blocks = {static_cast<BlockIdxType>(100 + i)};
        records.push_back(makeGroupRecord(tag, blocks));
        expected.emplace(tag, blocks);
    }
    std::reverse(records.begin(), records.end());

    const auto load_context = makeLoadContext("dsv4", peer_addrs, cache_keys, records, /*cp_size=*/1);
    const auto request      = server.constructRemoteLoadRequestForMla(load_context, /*index=*/0, peer_addrs);
    ASSERT_EQ(request.tagged_group_block_ids_size(), static_cast<int>(dsv4RpcTags().size()));
    EXPECT_EQ(taggedRowsOf(request), expected);

    for (const bool reversed_topology : {false, true}) {
        const auto topology = makeDsv4RpcTopology(reversed_topology);
        const auto decoded  = DecodeRpcServer::decodeGroupBlockRecords(request, *topology);
        EXPECT_EQ(taggedRecordsOf(decoded), expected) << "reversed_topology=" << reversed_topology;
    }
}

TEST(DecodeRpcServerTest, CacheTransferGuardRejectsDeclaredVsMaterializedPhysicalDrift) {
    const auto declared = makeRpcGeometryConfig(RpcGroupGeometry{});

    // The replaced guard passed the declared config's own topology into a check
    // that resolves every boundary group by tag from that same config, so it
    // could only fail on a null spec: it accepted any physical drift.
    EXPECT_NO_THROW(declared.checkPhysicalGroupLayoutCompatible(declared.topology(), "self-comparison"));

    // A materialized layout that physically agrees is still accepted.
    EXPECT_NO_THROW(DecodeRpcServer::checkCacheTransferLayout(
        declared, makeMaterializedLayout(makeRpcGeometryConfig(RpcGroupGeometry{}))));

    // Each physical dimension of the group record must be able to fail the guard.
    RpcGroupGeometry drifted_heads;
    drifted_heads.local_kv_head_num = 4;
    RpcGroupGeometry drifted_block_size;
    drifted_block_size.seq_size_per_block        = 16;
    drifted_block_size.kernel_seq_size_per_block = 16;
    RpcGroupGeometry drifted_kernel_blocks;
    drifted_kernel_blocks.kernel_seq_size_per_block = 4;
    RpcGroupGeometry drifted_stride;
    drifted_stride.kv_block_stride_bytes = 512;

    for (const auto& drift : {drifted_heads, drifted_block_size, drifted_kernel_blocks, drifted_stride}) {
        const auto materialized = makeRpcGeometryConfig(drift);
        EXPECT_ANY_THROW(DecodeRpcServer::checkCacheTransferLayout(declared, makeMaterializedLayout(materialized)));
    }
}

TEST(DecodeRpcServerTest, TaggedBlockRowsResolveByTagNotByLocalGroupOrder) {
    auto                   topology = CacheTopology::create({makeRpcGroup("linear", {0}), makeRpcGroup("full", {1})},
                                                            {{0, {"linear"}}, {1, {"full"}}});
    BroadcastLoadRequestPB request;
    auto*                  full = request.add_tagged_group_block_ids();
    full->set_tag("full");
    full->add_block_ids(10);
    auto* linear = request.add_tagged_group_block_ids();
    linear->set_tag("linear");
    linear->add_block_ids(20);

    const auto expected = std::map<std::string, BlockIndicesType>{{"full", {10}}, {"linear", {20}}};
    EXPECT_EQ(taggedRecordsOf(DecodeRpcServer::decodeGroupBlockRecords(request, *topology)), expected);

    auto reordered = CacheTopology::create({makeRpcGroup("full", {1}), makeRpcGroup("linear", {0})},
                                           {{0, {"linear"}}, {1, {"full"}}});
    EXPECT_EQ(taggedRecordsOf(DecodeRpcServer::decodeGroupBlockRecords(request, *reordered)), expected);
    EXPECT_EQ(DecodeRpcServer::makeTaggedRequestKey(42, 1, topology->group("full").tag),
              DecodeRpcServer::makeTaggedRequestKey(42, 1, reordered->group("full").tag));
}

TEST(DecodeRpcServerTest, EmptyTaggedBlockRowsAreRejected) {
    auto                   topology = CacheTopology::create({makeRpcGroup("full", {0})}, {{0, {"full"}}});
    BroadcastLoadRequestPB request;
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockRecords(request, *topology));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsRejectTopologyMismatch) {
    auto topology =
        CacheTopology::create({makeRpcGroup("full", {0}), makeRpcGroup("linear", {0})}, {{0, {"full", "linear"}}});
    BroadcastLoadRequestPB missing_tag;
    auto*                  row = missing_tag.add_tagged_group_block_ids();
    row->set_tag("full");
    row->add_block_ids(1);

    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockRecords(missing_tag, *topology));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsRejectInvalidTagIdentity) {
    auto topology =
        CacheTopology::create({makeRpcGroup("full", {0}), makeRpcGroup("linear", {0})}, {{0, {"full", "linear"}}});

    BroadcastLoadRequestPB duplicate_tag;
    for (int i = 0; i < 2; ++i) {
        auto* row = duplicate_tag.add_tagged_group_block_ids();
        row->set_tag("full");
        row->add_block_ids(i);
    }
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockRecords(duplicate_tag, *topology));

    BroadcastLoadRequestPB empty_tag;
    empty_tag.add_tagged_group_block_ids()->add_block_ids(1);
    auto* known = empty_tag.add_tagged_group_block_ids();
    known->set_tag("full");
    known->add_block_ids(2);
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockRecords(empty_tag, *topology));

    BroadcastLoadRequestPB unknown_tag;
    for (const auto* tag : {"full", "unknown"}) {
        auto* row = unknown_tag.add_tagged_group_block_ids();
        row->set_tag(tag);
        row->add_block_ids(3);
    }
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockRecords(unknown_tag, *topology));
}

TEST_F(DecodeRpcResourceBoundaryTest, LoadAdapterPreservesConfigOrderAfterLocalRecordShuffle) {
    auto       batch    = makeBatchResource();
    const auto expected = taggedRecordsOf(batch.groupBlocks(0));
    ASSERT_EQ(expected, (std::map<std::string, BlockIndicesType>{{"full", {10, 11, 12}}, {"linear", {20, 21}}}));

    auto& records = batch.groupBlocks(0);
    std::reverse(records.begin(), records.end());

    const std::vector<std::string> peer_addrs = {"prefill-0"};
    {
        const auto shuffled_context =
            makeLoadContext("shuffled", peer_addrs, batch.cacheKeys(0), batch.groupBlocks(0), /*cp_size=*/1);
        EXPECT_EQ(taggedRowsOf(server_.constructRemoteLoadRequest(shuffled_context, /*index=*/0, peer_addrs)),
                  expected);
        EXPECT_EQ(taggedRowsOf(server_.constructRemoteLoadRequestForMla(shuffled_context, /*index=*/0, peer_addrs)),
                  expected);
    }

    const auto error = load(std::move(batch));
    EXPECT_EQ(error.code(), ErrorCode::LOAD_KV_CACHE_FAILED);
}

TEST_F(DecodeRpcResourceBoundaryTest, LoadAdapterRejectsCorruptTaggedStorageAtEntry) {
    const auto expect_rejected = [this](const std::function<void(CacheGroupBlockRecords&)>& corrupt,
                                        const std::string&                                  expected_error) {
        auto batch = makeBatchResource();
        corrupt(batch.groupBlocks(0));
        try {
            const auto error = load(std::move(batch));
            FAIL() << "corrupt tagged storage reached the later load path, error=" << error.ToString();
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find(expected_error), std::string::npos) << e.what();
        }
    };

    expect_rejected([](auto& records) { records[0] = nullptr; }, "null group record");
    expect_rejected([](auto& records) { records[0]->tag.clear(); }, "empty group tag");
    expect_rejected([](auto& records) { records[0]->tag = "unknown"; }, "unknown tag=unknown");
    expect_rejected([](auto& records) { records[1]->tag = records[0]->tag; }, "duplicate tag=linear");
    expect_rejected([](auto& records) { records.erase(records.begin() + 1); }, "group storage size=1 expected=2");
}

TEST(DecodeRpcServerTest, MtpCacheKeyUsesSharedBaseModelIdForEverySlot) {
    constexpr size_t mtp_base_model_id = 17;

    for (size_t mtp_model_id = 0; mtp_model_id < 2; ++mtp_model_id) {
        EXPECT_EQ(DecodeRpcServer::makeMTPModuleCacheKey(mtp_base_model_id, "101", /*layer_id=*/0),
                  "model_id_17_token_id_str_101_layer_id_0")
            << "mtp_model_id=" << mtp_model_id;
    }
}

TEST(DecodeRpcServerTest, MtpLoadPlanContainsOnlyModule0) {
    auto module0          = std::make_unique<EngineInitParams>();
    module0->model_id     = 17;
    auto module1          = std::make_unique<EngineInitParams>();
    module1->model_id     = 23;
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::move(module0));
    mtp_model_params->push_back(std::move(module1));
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].module_index, 0);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
    EXPECT_EQ(plan[0].cache_model_id, 17);
}

TEST(DecodeRpcServerTest, MtpLoadPlanRejectsMissingModule0) {
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(nullptr).empty());

    ProposeModelEngineInitParams missing_params;
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&missing_params).empty());

    auto                         empty_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    ProposeModelEngineInitParams no_modules(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(empty_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&no_modules).empty());

    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(nullptr);
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    ProposeModelEngineInitParams null_module0(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&null_module0).empty());
}

TEST(DecodeRpcServerTest, MtpLoadPlanIgnoresInactiveModules) {
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    mtp_model_params->push_back(nullptr);
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
}

TEST(DecodeRpcServerTest, ReadFailureLogContainsPeerErrorAndEveryBlockKey) {
    test::TestLogCapture log_capture("read_cache_failure");
    DecodeRpcServer::logReadFailures(/*request_id=*/42,
                                     "127.0.0.1:1:2",
                                     ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED,
                                     "connect failed",
                                     {"blocks={kv_key_0,kv_key_1}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("PD_CACHE_KEY_READ_FAILED"), std::string::npos);
    EXPECT_NE(log_content.find("127.0.0.1:1:2"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_0"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_1"), std::string::npos);
}

TEST(DecodeRpcServerTest, ReadTimeoutLogsKeysAndCancellationIsSilent) {
    test::TestLogCapture log_capture("read_cache_timeout_cancel");
    DecodeRpcServer::logReadFailures(
        /*request_id=*/43, "peer", ErrorCode::LOAD_CACHE_TIMEOUT, "timeout", {"blocks={timeout_key}"});
    DecodeRpcServer::logReadFailures(
        /*request_id=*/44, "peer", ErrorCode::CANCELLED, "cancelled", {"blocks={cancelled_key}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("timeout_key"), std::string::npos);
    EXPECT_EQ(log_content.find("cancelled_key"), std::string::npos);
}

}  // namespace rtp_llm
