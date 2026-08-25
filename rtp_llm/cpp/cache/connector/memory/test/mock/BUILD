load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "mock_memory_connector_lib",
    hdrs = [
        "MockKVCacheMemoryConnector.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache/connector/memory:memory_connector",
    ],
    copts = test_copts,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "test_rpc_service_lib",
    hdrs = [
        "TestRpcService.h",
    ],
    deps = [
        "//rtp_llm/cpp/model_rpc:model_rpc_pool",
    ],
    copts = test_copts,
    visibility = ["//visibility:public"],
)
# Staged for //rtp_llm/cpp/cache/test:cache_tag_routing_source_test, the terminal
# source gate that rejects positional cache group routing in this package.
filegroup(
    name = "cache_tag_routing_scan_srcs",
    srcs = glob([
        "*.h",
        "*.cc",
    ]),
    visibility = ["//rtp_llm/cpp/cache/test:__pkg__"],
)
