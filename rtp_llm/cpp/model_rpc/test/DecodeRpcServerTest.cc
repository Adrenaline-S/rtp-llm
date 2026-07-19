#include <algorithm>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {

namespace {

DecodeRpcServer::LoadKVCacheContext makeLoadContext(const std::string&               request_key,
                                                    const std::vector<std::string>&  peer_addrs,
                                                    const std::vector<CacheKeyType>& cache_keys,
                                                    const GroupBlockIds&             block_ids_by_group,
                                                    int32_t                          prefill_cp_size) {
    return {/*request_id=*/42,
            request_key,
            peer_addrs,
            cache_keys,
            block_ids_by_group,
            /*reuse_block_size=*/0,
            /*timeout_ms=*/1000,
            /*partition_count=*/1,
            /*partition_id=*/0,
            /*server_context=*/nullptr,
            prefill_cp_size};
}

GroupBase makeRpcGroup(std::string tag, std::vector<int> layer_ids) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = tag;
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

CacheConfig makeCpSlicedRpcConfig(const std::string& tag, int cp_size, CpBlockSliceMode slice_mode) {
    CacheConfig config;
    config.seq_size_per_block = 16 / static_cast<size_t>(cp_size);
    config.layer_num          = 1;
    config.layer_all_num      = 1;

    GroupBase group;
    group.tag                        = tag;
    group.spec                       = test::makeResolvedOpaqueSpec(true, tag, DataType::TYPE_UINT8, 64, 16);
    group.policy                     = defaultCacheGroupPolicy(CacheGroupType::SWA);
    group.policy.active_tail_blocks  = 3;
    group.policy.cp_mapping          = CpBlockMappingMode::COMPACT_LAST_RANK;
    group.policy.cp_slice            = slice_mode;
    group.layer_ids                  = {0};
    group.block_num                  = 8;
    group.seq_size_per_block         = 16;
    group.kernel_seq_size_per_block  = 16;

    LayerBase layer;
    layer.layer_id   = 0;
    layer.group_tags = {tag};
    config.setTopology({std::move(group)}, {std::move(layer)});
    return config;
}

}  // namespace

TEST(ModelRpcProtoTest, CacheRpcFieldNumbersRemainStable) {
    const auto* broadcast = BroadcastLoadRequestPB::descriptor();
    EXPECT_EQ(broadcast->FindFieldByName("block_ids")->number(), 5);
    EXPECT_EQ(broadcast->FindFieldByName("block_num")->number(), 6);
    EXPECT_EQ(broadcast->FindFieldByName("reuse_block_size")->number(), 7);
    EXPECT_EQ(broadcast->FindFieldByName("timeout_ms")->number(), 8);
    EXPECT_EQ(broadcast->FindFieldByName("dp_rank")->number(), 9);
    EXPECT_EQ(broadcast->FindFieldByName("partition_count")->number(), 10);
    EXPECT_EQ(broadcast->FindFieldByName("partition_id")->number(), 11);
    EXPECT_EQ(broadcast->FindFieldByName("group_block_ids")->number(), 12);
    EXPECT_EQ(broadcast->FindFieldByName("prefill_cp_size")->number(), 13);
    EXPECT_EQ(broadcast->FindFieldByName("tagged_group_block_ids")->number(), 14);
    EXPECT_TRUE(broadcast->FindFieldByName("block_ids")->options().deprecated());
    EXPECT_TRUE(broadcast->FindFieldByName("group_block_ids")->options().deprecated());

    const auto* remote = RemoteOperationRequestPB::descriptor();
    EXPECT_EQ(remote->FindFieldByName("group_ids")->number(), 3);
    EXPECT_EQ(remote->FindFieldByName("block_ids")->number(), 4);
    EXPECT_EQ(remote->FindFieldByName("uris")->number(), 5);
    EXPECT_EQ(remote->FindFieldByName("group_tags")->number(), 6);
    EXPECT_TRUE(remote->FindFieldByName("group_ids")->options().deprecated());
}

TEST(DecodeRpcServerTest, CPShardedLoadRequestReadsFromEveryPrefillPeer) {
    for (const int cp_size : {2, 4}) {
        DecodeRpcServer server;
        server.resource_.workers = {"decode-0", "decode-1"};

        const std::string               request_key = "request";
        std::vector<std::string>        peer_addrs;
        const std::vector<CacheKeyType> cache_keys = {101, 102};
        const GroupBlockIds             block_ids_by_group;
        for (int rank = 0; rank < cp_size; ++rank) {
            peer_addrs.push_back("prefill-" + std::to_string(rank));
        }
        const auto load_context = makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, cp_size);

        const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/0, peer_addrs);

        EXPECT_EQ(request.prefill_cp_size(), cp_size);
        EXPECT_EQ(request.partition_count(), 1);
        EXPECT_EQ(request.partition_id(), 0);
        ASSERT_EQ(request.peer_addrs_size(), cp_size);
        for (int rank = 0; rank < cp_size; ++rank) {
            EXPECT_EQ(request.peer_addrs(rank), "prefill-" + std::to_string(rank));
        }
        ASSERT_EQ(request.cache_keys_size(), 2);
        EXPECT_EQ(request.cache_keys(0), 101);
        EXPECT_EQ(request.cache_keys(1), 102);
    }
}

TEST(DecodeRpcServerTest, CPShardedMlaLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101};
    const GroupBlockIds             block_ids_by_group;
    const auto load_context = makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/2);

    const auto request = server.constructRemoteLoadRequestForMla(load_context, /*index=*/1, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
}

TEST(DecodeRpcServerTest, Cp2AndCp4PeerPayloadsReconstructDecodeBlocks) {
    constexpr size_t block_bytes = 64;
    constexpr size_t block_count = 3;

    for (const int cp_size : {2, 4}) {
        for (const auto slice_mode : {CpBlockSliceMode::PAYLOAD_BYTES, CpBlockSliceMode::EQUAL_BYTES}) {
            auto config = makeCpSlicedRpcConfig(
                slice_mode == CpBlockSliceMode::PAYLOAD_BYTES ? "state" : "swa", cp_size, slice_mode);
            std::vector<uint8_t> source(block_count * block_bytes);
            for (size_t i = 0; i < source.size(); ++i) {
                source[i] = static_cast<uint8_t>((i * 31 + cp_size + static_cast<int>(slice_mode)) % 251);
            }
            std::vector<uint8_t> reconstructed(source.size(), 0);

            const size_t logical_key_count = static_cast<size_t>(cp_size * 2 + 1);
            for (int peer_idx = 0; peer_idx < cp_size; ++peer_idx) {
                const auto load_plan = DecodeRpcServer::makeCpPeerLoadPlan(config,
                                                                           /*gid=*/0,
                                                                           block_count,
                                                                           logical_key_count,
                                                                           /*reuse_block_size=*/0,
                                                                           /*use_hybrid=*/true,
                                                                           cp_size,
                                                                           cp_size,
                                                                           peer_idx);
                ASSERT_EQ(load_plan.size(), block_count);
                for (size_t compact_idx = 0; compact_idx < load_plan.size(); ++compact_idx) {
                    const auto& plan_entry = load_plan[compact_idx];
                    EXPECT_EQ(plan_entry.block_pos, compact_idx);
                    EXPECT_EQ(plan_entry.cache_key_index,
                              std::min((compact_idx + 1) * static_cast<size_t>(cp_size) - 1,
                                       logical_key_count - 1));

                    BlockInfo destination;
                    destination.addr       = reconstructed.data() + plan_entry.block_pos * block_bytes;
                    destination.size_bytes = block_bytes;
                    auto peer_destination  = DecodeRpcServer::sliceCpDestinationForPeer(
                        config, /*gid=*/0, {destination}, cp_size, peer_idx);
                    ASSERT_EQ(peer_destination.size(), 1u);

                    const size_t offset = static_cast<uint8_t*>(peer_destination[0].addr)
                                          - static_cast<uint8_t*>(destination.addr);
                    ASSERT_LE(offset + peer_destination[0].size_bytes, block_bytes);
                    std::copy_n(source.data() + plan_entry.block_pos * block_bytes + offset,
                                peer_destination[0].size_bytes,
                                static_cast<uint8_t*>(peer_destination[0].addr));
                }
            }
            EXPECT_EQ(reconstructed, source) << "cp_size=" << cp_size << " slice=" << static_cast<int>(slice_mode);
        }
    }
}

TEST(DecodeRpcServerTest, Cp2AndCp4FullGroupLoadPlanRoutesPartialTailToOwningPeer) {
    for (const int cp_size : {2, 4}) {
        const size_t logical_blocks = static_cast<size_t>(cp_size * 2 + 1);
        CacheConfig  config;
        config.seq_size_per_block = 8;
        config.layer_num          = 1;
        config.layer_all_num      = 1;
        LayerBase layer;
        layer.layer_id   = 0;
        layer.group_tags = {"full"};
        config.setTopology({makeRpcGroup("full", {0})}, {std::move(layer)});

        std::vector<int> owners(logical_blocks, -1);
        for (int peer_idx = 0; peer_idx < cp_size; ++peer_idx) {
            const auto plan = DecodeRpcServer::makeCpPeerLoadPlan(config,
                                                                 /*gid=*/0,
                                                                 logical_blocks,
                                                                 logical_blocks,
                                                                 /*reuse_block_size=*/0,
                                                                 /*use_hybrid=*/false,
                                                                 cp_size,
                                                                 cp_size,
                                                                 peer_idx);
            for (const auto& entry : plan) {
                EXPECT_EQ(entry.block_pos, entry.cache_key_index);
                ASSERT_EQ(owners[entry.block_pos], -1);
                owners[entry.block_pos] = peer_idx;
            }
        }
        for (size_t block_pos = 0; block_pos < logical_blocks; ++block_pos) {
            EXPECT_EQ(owners[block_pos], static_cast<int>(block_pos % static_cast<size_t>(cp_size)));
        }
    }
}

TEST(DecodeRpcServerTest, TaggedBlockRowsResolveByLocalTagOrder) {
    auto                   topology = CacheTopology::create({makeRpcGroup("linear", {0}), makeRpcGroup("full", {1})},
                                                            {{0, {"linear"}}, {1, {"full"}}});
    BroadcastLoadRequestPB request;
    auto*                  full = request.add_tagged_group_block_ids();
    full->set_tag("full");
    full->add_block_ids(10);
    auto* linear = request.add_tagged_group_block_ids();
    linear->set_tag("linear");
    linear->add_block_ids(20);

    const auto blocks = DecodeRpcServer::decodeGroupBlockIds(request, *topology);
    EXPECT_EQ(blocks[topology->groupIdForTag("full")]->blocks(), (BlockIndicesType{10}));
    EXPECT_EQ(blocks[topology->groupIdForTag("linear")]->blocks(), (BlockIndicesType{20}));

    auto reordered = CacheTopology::create({makeRpcGroup("full", {1}), makeRpcGroup("linear", {0})},
                                           {{0, {"linear"}}, {1, {"full"}}});
    EXPECT_NE(topology->groupIdForTag("full"), reordered->groupIdForTag("full"));
    const auto reordered_blocks = DecodeRpcServer::decodeGroupBlockIds(request, *reordered);
    EXPECT_EQ(reordered_blocks[reordered->groupIdForTag("full")]->blocks(), (BlockIndicesType{10}));
    EXPECT_EQ(reordered_blocks[reordered->groupIdForTag("linear")]->blocks(), (BlockIndicesType{20}));
    EXPECT_EQ(DecodeRpcServer::makeTaggedRequestKey(42, 1, topology->group("full").tag),
              DecodeRpcServer::makeTaggedRequestKey(42, 1, reordered->group("full").tag));
}

TEST(DecodeRpcServerTest, EmptyTaggedBlockRowsAreRejected) {
    auto                   topology = CacheTopology::create({makeRpcGroup("full", {0})}, {{0, {"full"}}});
    BroadcastLoadRequestPB request;
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(request, *topology));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsRejectTopologyMismatch) {
    auto topology =
        CacheTopology::create({makeRpcGroup("full", {0}), makeRpcGroup("linear", {0})}, {{0, {"full", "linear"}}});
    BroadcastLoadRequestPB missing_tag;
    auto*                  row = missing_tag.add_tagged_group_block_ids();
    row->set_tag("full");
    row->add_block_ids(1);

    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(missing_tag, *topology));
}

TEST(PrefillRpcServerTest, PDSepEligibilityRejectsUnsupportedGenerationModes) {
    PrefillRpcServer server;
    GenerateInputPB  input;
    auto*            config = input.mutable_generate_config();
    config->set_max_new_tokens(2);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    EXPECT_TRUE(server.canUsePDSep(input));

    auto single_token = input;
    single_token.mutable_generate_config()->set_max_new_tokens(1);
    EXPECT_FALSE(server.canUsePDSep(single_token));

    auto beam_search = input;
    beam_search.mutable_generate_config()->set_num_beams(2);
    EXPECT_FALSE(server.canUsePDSep(beam_search));

    auto variable_beam = input;
    variable_beam.mutable_generate_config()->add_variable_num_beams(2);
    EXPECT_FALSE(server.canUsePDSep(variable_beam));

    auto multi_return = input;
    multi_return.mutable_generate_config()->set_num_return_sequences(2);
    EXPECT_FALSE(server.canUsePDSep(multi_return));

    auto explicitly_disabled = input;
    explicitly_disabled.mutable_generate_config()->set_can_use_pd_separation(false);
    EXPECT_FALSE(server.canUsePDSep(explicitly_disabled));

    auto expect_aux_output_rejected = [&](auto setter) {
        auto with_aux_output = input;
        setter(*with_aux_output.mutable_generate_config());
        EXPECT_FALSE(server.canUsePDSep(with_aux_output));
    };
    expect_aux_output_rejected([](GenerateConfigPB& config) { config.set_calculate_loss(1); });
    expect_aux_output_rejected([](GenerateConfigPB& config) { config.set_return_hidden_states(true); });
    expect_aux_output_rejected([](GenerateConfigPB& config) { config.set_return_all_hidden_states(true); });
    expect_aux_output_rejected([](GenerateConfigPB& config) { config.set_return_logits(true); });
    expect_aux_output_rejected([](GenerateConfigPB& config) { config.set_return_all_probs(true); });
    expect_aux_output_rejected([](GenerateConfigPB& config) { config.set_return_all_probs_mode(2); });
    expect_aux_output_rejected([](GenerateConfigPB& config) { config.set_return_softmax_probs(true); });
    expect_aux_output_rejected([](GenerateConfigPB& config) { config.set_return_cum_log_probs(true); });
    expect_aux_output_rejected([](GenerateConfigPB& config) { config.set_return_prompt_logits(true); });

    auto target_logprob_without_prompt_logits = input;
    target_logprob_without_prompt_logits.mutable_generate_config()->set_return_target_logprob(true);
    EXPECT_TRUE(server.canUsePDSep(target_logprob_without_prompt_logits));
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
