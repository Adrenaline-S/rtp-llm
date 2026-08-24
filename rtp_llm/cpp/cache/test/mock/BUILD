load("//:def.bzl", "copts")

test_copts = [
    "-fno-access-control",
] + copts()

cc_library(
    name = "mock_coordinator_cache_manager_lib",
    hdrs = [
        "MockCoordinatorCacheManager.h",
    ],
    deps = [
        "//rtp_llm/cpp/cache:cache_core",
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
