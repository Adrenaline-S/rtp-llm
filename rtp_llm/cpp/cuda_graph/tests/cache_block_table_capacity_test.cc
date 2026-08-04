#include <limits>
#include <string>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"

namespace rtp_llm {
namespace {

template<typename Fn>
void expectInvalidCapacity(Fn&& fn, std::string_view expected_message) {
    try {
        fn();
        FAIL() << "expected invalid cache block capacity";
    } catch (const RTPException& e) {
        EXPECT_NE(std::string(e.what()).find(expected_message), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("context=invalid-test"), std::string::npos) << e.what();
    }
}

TEST(CacheBlockTableCapacityTest, DefaultsKernelBlockSizeToPhysicalBlockSize) {
    EXPECT_EQ(calculateKernelBlockTableCapacity(65, 16, 0, 0, "default"), 5);
}

TEST(CacheBlockTableCapacityTest, DerivesHeterogeneousKernelCapacities) {
    EXPECT_EQ(calculateKernelBlockTableCapacity(65, 16, 4, 2, "full"), 28);
    EXPECT_EQ(calculateKernelBlockTableCapacity(65, 8, 8, 2, "linear"), 11);
}

TEST(CacheBlockTableCapacityTest, RejectsInvalidInputs) {
    expectInvalidCapacity(
        []() { (void)calculateKernelBlockTableCapacity(0, 16, 4, 0, "invalid-test"); }, "max_seq_len");
    expectInvalidCapacity(
        []() { (void)calculateKernelBlockTableCapacity(64, 0, 4, 0, "invalid-test"); }, "physical tokens");
    expectInvalidCapacity(
        []() { (void)calculateKernelBlockTableCapacity(64, 16, -1, 0, "invalid-test"); }, "kernel tokens");
    expectInvalidCapacity(
        []() { (void)calculateKernelBlockTableCapacity(64, 16, 6, 0, "invalid-test"); }, "must be divisible");
    expectInvalidCapacity(
        []() { (void)calculateKernelBlockTableCapacity(64, 16, 4, -1, "invalid-test"); }, "sp_steps");
}

TEST(CacheBlockTableCapacityTest, RejectsCapacityOverflow) {
    expectInvalidCapacity(
        []() {
            (void)calculateKernelBlockTableCapacity(
                1, 1, 1, std::numeric_limits<int64_t>::max(), "invalid-test");
        },
        "physical capacity overflow");
    expectInvalidCapacity(
        []() {
            (void)calculateKernelBlockTableCapacity(
                std::numeric_limits<int64_t>::max(), 2, 1, 0, "invalid-test");
        },
        "kernel capacity overflow");
}

}  // namespace
}  // namespace rtp_llm
