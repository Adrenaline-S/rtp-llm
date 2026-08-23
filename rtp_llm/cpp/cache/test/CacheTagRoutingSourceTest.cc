// Terminal source gate for tag-only cache-group routing.
//
// A cache group's business identity is its semantic tag ("full", "swa",
// "linear", "indexer_kv", ...). A positional group index is an implementation
// detail of one container and must never be an interface value, a config
// accessor key, a resource key, an RPC/Python DTO field, or persistent state.
// This test scans the cache subsystem's own sources and fails when positional
// group routing reappears.
//
// The gate is deliberately precise in two directions:
//
//  1. It rejects the positional *API family* -- the CacheConfig/CacheTopology
//     accessors that take a slot index, the per-layer positional matrices, and
//     the parallel-vector snapshot builders -- and it also rejects the
//     conventional positional identifier spellings. Renaming a local variable
//     therefore cannot satisfy it: the API call itself is what is rejected.
//
//  2. It scans cache-routing sources only. "Group id" concepts that have
//     nothing to do with cache groups stay allowed because they live outside the
//     scanned roots: request/force-batch grouping (the GenerateInput request
//     group field and its RPC/Python plumbing), and expert/MoE plus
//     quantization group indices in kernels. Those are a different domain and
//     must not be disturbed.
//
// Explicitly allowed inside the scanned roots, by construction rather than by
// exception list:
//
//  * `group_ordinal` -- the adapter-local positional column produced at CUDA
//    tensor and wire-bitmask boundaries from sorted unique tags. It is never
//    persisted, returned to a config/resource API, or placed in a DTO, and it
//    contains none of the rejected tokens.
//  * the private `tag_to_slot` index inside a single container.
//
// This file is itself inside the scanned tree, so every rejected token is
// assembled from fragments at runtime and never appears literally here. The
// gate consequently polices itself and needs no self-exemption.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

namespace fs = std::filesystem;

// Workspace-relative roots that are scanned. A directory root is scanned
// recursively, so a newly added cache source is covered automatically.
const std::vector<std::string>& scannedRoots() {
    static const std::vector<std::string> roots = {
        // The whole cache subsystem: config/topology, resources, allocator,
        // pools, groups, shared cache, every connector, and all of their tests.
        "rtp_llm/cpp/cache",
        // The decode-side cache transfer boundary. Only the DecodeRpcServer
        // cache sources are staged here; the rest of model_rpc carries
        // request-batch group ids and is intentionally not scanned.
        "rtp_llm/cpp/model_rpc",
    };
    return roots;
}

// Files that must be present in the scan, so that a broken data dependency
// silently narrowing the scan cannot make this gate pass vacuously.
const std::vector<std::string>& requiredScannedFiles() {
    static const std::vector<std::string> required = {
        "rtp_llm/cpp/cache/CacheConfig.h",
        "rtp_llm/cpp/cache/CacheConfig.cc",
        "rtp_llm/cpp/cache/CacheTopology.h",
        "rtp_llm/cpp/cache/CacheTopology.cc",
        "rtp_llm/cpp/cache/CacheConfigCreator.cc",
        "rtp_llm/cpp/cache/KVCacheResource.h",
        "rtp_llm/cpp/cache/KVCacheResource.cc",
        "rtp_llm/cpp/cache/BatchKVCacheResource.h",
        "rtp_llm/cpp/cache/BufferTypes.h",
        "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.cc",
        "rtp_llm/cpp/cache/KVCacheManager.cc",
        "rtp_llm/cpp/cache/SharedBlockCache.h",
        "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc",
        "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.cc",
        "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.cc",
        "rtp_llm/cpp/cache/test/CacheTagRoutingSourceTest.cc",
        "rtp_llm/cpp/model_rpc/DecodeRpcServer.cc",
    };
    return required;
}

struct Rule {
    std::string why;
    std::regex  pattern;
};

// Rejected tokens are assembled from fragments at runtime so this source
// contains no literal rejected token and is therefore scannable by itself.
std::string lit(const std::string& fragment) {
    return fragment;
}

const std::vector<Rule>& rules() {
    static const std::vector<Rule> all = [] {
        const std::string kGroup    = lit("group");
        const std::string kGrp      = lit("Group");
        const std::string kForGroup = lit("For") + kGrp;

        // Slot-indexed CacheConfig/CacheTopology field projections. Every one of
        // these is replaced by reading the complete tagged group record via
        // CacheConfig::group(tag).
        const std::vector<std::string> projections = {"spec",
                                                     "type",
                                                     "tag",
                                                     "layerIds",
                                                     "blockNum",
                                                     "policy",
                                                     "localKvHeadNum",
                                                     "seqSizePerBlock",
                                                     "kernelSeqSizePerBlock",
                                                     "kernelBlocksPerKvBlock",
                                                     "kvBlockStrideBytes",
                                                     "kvScaleStrideBytes",
                                                     "blockSizeBytes"};
        std::string                    alternation;
        for (const auto& projection : projections) {
            if (!alternation.empty()) {
                alternation += "|";
            }
            alternation += projection;
        }

        std::vector<Rule> built;
        built.push_back(
            {"positional cache group index as a variable, field, parameter or map key (" + kGroup + "_id)",
             std::regex(kGroup + "_id")});
        built.push_back({"positional cache group accessor or helper (" + kGroup + "Id...)",
                         std::regex(kGroup + "Id")});
        built.push_back({"positional cache group type or helper (" + kGrp + "Id...)", std::regex(kGrp + "Id")});
        built.push_back({"positional cache group lookup (" + kGroup + "ById)", std::regex(kGroup + "By" + "Id")});
        built.push_back({"positional cache group index abbreviation", std::regex(R"(\bgids?\b)")});
        built.push_back(
            {"slot-indexed CacheConfig/CacheTopology group projection; read CacheConfig::group(tag) instead",
             std::regex("(" + alternation + ")" + kForGroup)});
        built.push_back({"parallel-vector group snapshot builder indexed by slot; iterate tagged group records instead",
                         std::regex(kGroup + R"([A-Za-z]*Snapshot)")});
        return built;
    }();
    return all;
}

std::string workspaceRoot() {
    const char* srcdir = std::getenv("TEST_SRCDIR");
    if (srcdir == nullptr) {
        return {};
    }
    const char* workspace = std::getenv("TEST_WORKSPACE");
    const fs::path base(srcdir);
    if (workspace != nullptr && !std::string(workspace).empty()) {
        const fs::path candidate = base / workspace;
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
    const fs::path fallback = base / "rtp_llm";
    if (fs::exists(fallback)) {
        return fallback.string();
    }
    return base.string();
}

bool isScannedExtension(const fs::path& path) {
    const std::string ext = path.extension().string();
    return ext == ".h" || ext == ".hpp" || ext == ".cc" || ext == ".cpp" || ext == ".cu" || ext == ".cuh";
}

struct Finding {
    std::string relative_path;
    int         line_no = 0;
    std::string why;
    std::string line;
};

void scanFile(const fs::path& absolute, const std::string& relative, std::vector<Finding>& findings) {
    std::ifstream input(absolute);
    if (!input.is_open()) {
        return;
    }
    std::string line;
    int         line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        for (const auto& rule : rules()) {
            if (std::regex_search(line, rule.pattern)) {
                std::string trimmed = line;
                const auto  first   = trimmed.find_first_not_of(" \t");
                if (first != std::string::npos) {
                    trimmed = trimmed.substr(first);
                }
                if (trimmed.size() > 160) {
                    trimmed = trimmed.substr(0, 160) + " ...";
                }
                findings.push_back({relative, line_no, rule.why, trimmed});
                break;
            }
        }
    }
}

}  // namespace

TEST(CacheTagRoutingSourceTest, CacheSourcesCarryNoPositionalGroupRouting) {
    const std::string root = workspaceRoot();
    ASSERT_FALSE(root.empty()) << "TEST_SRCDIR is unset; the source gate cannot locate the staged cache sources";

    std::set<std::string>  scanned;
    std::vector<Finding>   findings;

    for (const auto& relative_root : scannedRoots()) {
        const fs::path absolute_root = fs::path(root) / relative_root;
        if (!fs::exists(absolute_root)) {
            continue;
        }
        std::error_code ec;
        for (fs::recursive_directory_iterator it(absolute_root,
                                                 fs::directory_options::follow_directory_symlink
                                                     | fs::directory_options::skip_permission_denied,
                                                 ec);
             it != fs::recursive_directory_iterator();
             it.increment(ec)) {
            if (ec) {
                break;
            }
            if (!it->is_regular_file(ec) || ec) {
                continue;
            }
            if (!isScannedExtension(it->path())) {
                continue;
            }
            // Runfiles entries are symlinks into the execroot. fs::relative would
            // canonicalize them and escape the scan root, so the workspace-relative
            // path is derived lexically.
            const std::string relative = it->path().lexically_relative(fs::path(root)).generic_string();
            if (relative.empty() || relative.compare(0, 2, "..") == 0) {
                continue;
            }
            if (!scanned.insert(relative).second) {
                continue;
            }
            scanFile(it->path(), relative, findings);
        }
    }

    // Guard the scan itself: an empty or narrowed scan must fail loudly rather
    // than report a vacuous pass.
    ASSERT_GE(scanned.size(), 100u) << "source gate scanned only " << scanned.size()
                                    << " files; the staged cache sources are incomplete";
    for (const auto& required : requiredScannedFiles()) {
        EXPECT_EQ(scanned.count(required), 1u) << "source gate did not scan required file " << required;
    }

    if (!findings.empty()) {
        std::ostringstream report;
        report << "positional cache group routing survives in " << findings.size() << " place(s):\n";
        for (const auto& finding : findings) {
            report << "  " << finding.relative_path << ":" << finding.line_no << ": " << finding.why << "\n"
                   << "      " << finding.line << "\n";
        }
        FAIL() << report.str();
    }
}

}  // namespace rtp_llm
