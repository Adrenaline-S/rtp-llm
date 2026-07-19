#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"

namespace rtp_llm {
namespace {

BatchGenerateInputPB makeStreamingBatch(int streaming_index) {
    BatchGenerateInputPB request;
    for (int i = 0; i < 3; ++i) {
        auto* input = request.add_inputs();
        input->mutable_generate_config()->set_is_streaming(i == streaming_index);
    }
    return request;
}

TEST(BatchGenerateValidationTest, LocalEntryRejectsStreamingInputBeforeEngineUse) {
    LocalRpcServer         server;
    grpc::ServerContext    context;
    auto                   request = makeStreamingBatch(/*streaming_index=*/1);
    BatchGenerateOutputsPB response;

    const auto status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.error_message().find("input index: 1"), std::string::npos);
    EXPECT_EQ(response.results_size(), 0);
}

TEST(BatchGenerateValidationTest, PrefillEntryRejectsStreamingInputBeforePdSetup) {
    PrefillRpcServer       server;
    grpc::ServerContext    context;
    auto                   request = makeStreamingBatch(/*streaming_index=*/2);
    BatchGenerateOutputsPB response;

    const auto status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.error_message().find("input index: 2"), std::string::npos);
    EXPECT_EQ(response.results_size(), 0);
}

TEST(BatchGenerateValidationTest, ReturnIncrementalAloneRemainsValid) {
    BatchGenerateInputPB request;
    request.add_inputs()->mutable_generate_config()->set_return_incremental(true);

    EXPECT_TRUE(LocalRpcServer::validateBatchGenerateRequest(request).ok());
}

}  // namespace
}  // namespace rtp_llm
