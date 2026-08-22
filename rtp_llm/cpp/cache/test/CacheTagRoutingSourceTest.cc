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
// The gate scans production sources only. Characterization fixtures may name
// the legacy types while proving the current behavior; those names must not
// make a production architecture gate pass or fail vacuously.
//
// Explicitly allowed inside the scanned roots:
//
//  * Exactly one `group_ordinal` member: the first column of
//    `GroupOrdinalBlockIdPair`, the final [copies,3] CUDA tensor boundary
//    record. No file-wide or general adapter-local ordinal exception exists.
//  * CP-domain terminology in CPSlotMapper. CP's logical/physical slot is
//    distinct from cache-group identity; the gate rejects only the five
//    cache-group identity spellings below, never generic CP `slot` terms.
//
// This test source is staged with the cache tree, but the production-only
// filter below excludes it. The forbidden-name list can therefore stay
// literal and auditable without self-matching.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
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
        "rtp_llm/cpp/cache/CacheConfigCreator.cc",
        "rtp_llm/cpp/cache/KVCacheResource.h",
        "rtp_llm/cpp/cache/KVCacheResource.cc",
        "rtp_llm/cpp/cache/BatchKVCacheResource.h",
        "rtp_llm/cpp/cache/BufferTypes.h",
        "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.cc",
        "rtp_llm/cpp/cache/HybridPoolKVCacheAllocatorCoordinator.cc",
        "rtp_llm/cpp/cache/KVCacheManager.cc",
        "rtp_llm/cpp/cache/SharedBlockCache.h",
        "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.cc",
        "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.cc",
        "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheSender.cc",
        "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.cc",
        "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.cc",
        "rtp_llm/cpp/model_rpc/DecodeRpcServer.cc",
    };
    return required;
}

struct Rule {
    std::string why;
    std::regex  pattern;
    std::string identifier_fragment;
};

// Legacy positional patterns are assembled from fragments because they are
// still checked over a broad source scope.
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

        // These types were introduced during the Phase 1 attempt but are not
        // needed for the approved flat tag-keyed representation. Their
        // absence is an intentional architecture acceptance criterion, not a
        // proxy for behavior: the semantic fixtures exercise the behavior.
        const std::vector<std::string> forbidden_phase1_types = {
            "GroupAllocationCheckpoint", "BatchAllocationCheckpoint", "AllocationRollbackJournal",
            "NativeTransferSelection", "NativeTransferSelections", "PhysicalBlockTransferPlan",
            "TaggedBlockPool", "TaggedSharedGroupEntry", "TaggedCacheItem",
        };
        for (const auto& type : forbidden_phase1_types) {
            built.push_back({"premature Phase 1 abstraction " + type,
                             std::regex("$^"),
                             type});
        }

        // A tag is the cache-group business identity. A vector index may be
        // named idx only where local storage needs one; none of these names
        // may reintroduce a cache-group slot/id abstraction.
        const std::vector<std::string> forbidden_identity_names = {
            "tag_to_slot", "group_slot", "groupIdForTag", "groupById", "layer_group_ids",
            "CacheTopology", "topologyPtr", "groupTagsInConfigOrder",
        };
        for (const auto& name : forbidden_identity_names) {
            built.push_back({"cache-group identity must remain tag-keyed (" + name + ")",
                             std::regex("$^"),
                             name});
        }
        built.push_back({"cache metadata must be accessed directly through CacheConfig",
                         std::regex(R"(\.topology\(\))")});
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

bool isProductionSource(const fs::path& relative) {
    for (const auto& component : relative) {
        if (component == "test") {
            return false;
        }
    }
    return true;
}

bool isIdentifierStart(char c) {
    return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool isIdentifierContinue(char c) {
    return isIdentifierStart(c) || (c >= '0' && c <= '9');
}

// This intentionally tracks only block-comment continuation between lines. It
// is not a complete C++ lexer: string and character literals are consumed on
// their own line, which is sufficient for identifier-rule source gating.
struct IdentifierLexerState {
    bool in_block_comment = false;
};

std::vector<std::string_view> identifierTokens(const std::string& line, IdentifierLexerState& state) {
    std::vector<std::string_view> tokens;
    for (size_t cursor = 0; cursor < line.size();) {
        if (state.in_block_comment) {
            const auto comment_end = line.find("*/", cursor);
            if (comment_end == std::string::npos) {
                break;
            }
            state.in_block_comment = false;
            cursor                 = comment_end + 2;
            continue;
        }
        if (line.compare(cursor, 2, "//") == 0) {
            break;
        }
        if (line.compare(cursor, 2, "/*") == 0) {
            const auto comment_end = line.find("*/", cursor + 2);
            if (comment_end == std::string::npos) {
                state.in_block_comment = true;
                break;
            }
            cursor = comment_end + 2;
            continue;
        }
        if (line[cursor] == '"' || line[cursor] == '\'') {
            const char quote = line[cursor++];
            while (cursor < line.size()) {
                if (line[cursor++] == '\\' && cursor < line.size()) {
                    ++cursor;
                } else if (line[cursor - 1] == quote) {
                    break;
                }
            }
            continue;
        }
        if (!isIdentifierStart(line[cursor])) {
            ++cursor;
            continue;
        }
        const size_t begin = cursor++;
        while (cursor < line.size() && isIdentifierContinue(line[cursor])) {
            ++cursor;
        }
        tokens.emplace_back(line.data() + begin, cursor - begin);
    }
    return tokens;
}

bool containsIdentifierFragment(const std::vector<std::string_view>& tokens, std::string_view fragment) {
    for (const auto token : tokens) {
        if (token.find(fragment) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

bool matchesRule(const Rule& rule, const std::string& line, const std::vector<std::string_view>& tokens) {
    return rule.identifier_fragment.empty() ? std::regex_search(line, rule.pattern)
                                            : containsIdentifierFragment(tokens, rule.identifier_fragment);
}

bool matchesRule(const Rule& rule, const std::string& line) {
    IdentifierLexerState state;
    return matchesRule(rule, line, identifierTokens(line, state));
}

bool isAllowedFinalBoundaryOrdinal(const std::string&                   relative,
                                   std::string_view                     enclosing_record,
                                   const std::vector<std::string_view>& tokens) {
    return relative == "rtp_llm/cpp/cache/Types.h" && enclosing_record == "GroupOrdinalBlockIdPair"
           && tokens.size() == 2 && tokens[0] == "int32_t" && tokens[1] == "group_ordinal";
}

bool hasProjectedCacheGroupSlotIdentity(const std::vector<std::string_view>& tokens) {
    for (const auto token : tokens) {
        const auto tag_to = token.find("tag_to_");
        if (tag_to == std::string_view::npos || token.size() < 5) {
            continue;
        }
        if (token.substr(token.size() - 5) == "_slot") {
            return true;
        }
    }
    return false;
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
    std::string active_record;
    size_t      allowed_boundary_ordinals = 0;
    IdentifierLexerState lexer_state;
    while (std::getline(input, line)) {
        ++line_no;
        const auto tokens = identifierTokens(line, lexer_state);
        if (tokens.size() >= 2 && tokens[0] == "struct") {
            active_record = std::string(tokens[1]);
        }
        if (std::find(tokens.begin(), tokens.end(), "group_ordinal") != tokens.end()) {
            if (isAllowedFinalBoundaryOrdinal(relative, active_record, tokens)) {
                ++allowed_boundary_ordinals;
            } else {
                findings.push_back({relative,
                                    line_no,
                                    "group ordinal is allowed only as GroupOrdinalBlockIdPair::group_ordinal",
                                    line});
                continue;
            }
        }
        if (hasProjectedCacheGroupSlotIdentity(tokens)) {
            findings.push_back({relative,
                                line_no,
                                "cache-group vector positions must use idx rather than a projected slot identity",
                                line});
            continue;
        }
        for (const auto& rule : rules()) {
            if (matchesRule(rule, line, tokens)) {
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
        if (!active_record.empty() && line.find("};") != std::string::npos) {
            active_record.clear();
        }
    }
    if (relative == "rtp_llm/cpp/cache/Types.h" && allowed_boundary_ordinals != 1) {
        findings.push_back({relative,
                            0,
                            "Types.h must contain exactly one GroupOrdinalBlockIdPair::group_ordinal member",
                            "observed count=" + std::to_string(allowed_boundary_ordinals)});
    }
}

}  // namespace

TEST(CacheTagRoutingSourceTest, ProductionSourcesCarryNoForbiddenPhase1AbstractionsOrPositionalGroupRouting) {
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
            if (!isProductionSource(fs::path(relative))) {
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
        report << "forbidden Phase 1 abstractions or positional cache-group routing survive in " << findings.size()
               << " place(s):\n";
        for (const auto& finding : findings) {
            report << "  " << finding.relative_path << ":" << finding.line_no << ": " << finding.why << "\n"
                   << "      " << finding.line << "\n";
        }
        FAIL() << report.str();
    }
}

TEST(CacheTagRoutingSourceTest, ForbiddenIdentifierRulesCatchPrivateAndPrefixedSpellings) {
    const auto& all = rules();
    const auto find_rule = [&all](const std::string& fragment) -> const Rule& {
        const auto it = std::find_if(all.begin(), all.end(), [&fragment](const Rule& rule) {
            return rule.identifier_fragment == fragment;
        });
        EXPECT_NE(it, all.end()) << fragment;
        return *it;
    };

    EXPECT_TRUE(matchesRule(find_rule("tag_to_slot"), "size_t tag_to_slot_ = 0;"));
    EXPECT_TRUE(matchesRule(find_rule("tag_to_slot"), "size_t staged_tag_to_slot = 0;"));
    EXPECT_TRUE(matchesRule(find_rule("group_slot"), "size_t group_slot = 0;"));
    EXPECT_TRUE(matchesRule(find_rule("GroupAllocationCheckpoint"), "PrivateGroupAllocationCheckpoint value;"));
    EXPECT_TRUE(matchesRule(find_rule("NativeTransferSelection"), "NativeTransferSelections selections;"));
    EXPECT_FALSE(matchesRule(find_rule("tag_to_slot"), "const char* label = \"tag_to_slot\";"));
    EXPECT_FALSE(matchesRule(find_rule("tag_to_slot"), "// staged_tag_to_slot is forbidden only as code"));
    EXPECT_FALSE(matchesRule(find_rule("group_slot"), "CPSlotMapper cp_slot_mapper;"));
}

TEST(CacheTagRoutingSourceTest, IdentifierLexerCarriesBlockCommentStateAcrossLines) {
    const auto& all = rules();
    const auto it = std::find_if(all.begin(), all.end(), [](const Rule& rule) {
        return rule.identifier_fragment == "GroupAllocationCheckpoint";
    });
    ASSERT_NE(it, all.end());

    IdentifierLexerState state;
    EXPECT_FALSE(matchesRule(*it, "/* comment begins", identifierTokens("/* comment begins", state)));
    EXPECT_FALSE(matchesRule(*it,
                             "PrivateGroupAllocationCheckpoint remains comment text",
                             identifierTokens("PrivateGroupAllocationCheckpoint remains comment text", state)));
    EXPECT_TRUE(matchesRule(*it,
                            "*/ PrivateGroupAllocationCheckpoint real_identifier;",
                            identifierTokens("*/ PrivateGroupAllocationCheckpoint real_identifier;", state)));
}

TEST(CacheTagRoutingSourceTest, GroupOrdinalIsAllowedOnlyInFinalCudaBoundaryRecord) {
    IdentifierLexerState state;
    EXPECT_FALSE(isAllowedFinalBoundaryOrdinal(
        "rtp_llm/cpp/cache/HybridPoolKVCacheAllocatorCoordinator.cc",
        "",
        identifierTokens("size_t group_ordinal = 0;", state)));
    state = {};
    EXPECT_TRUE(isAllowedFinalBoundaryOrdinal(
        "rtp_llm/cpp/cache/Types.h",
        "GroupOrdinalBlockIdPair",
        identifierTokens("int32_t group_ordinal;", state)));
    state = {};
    EXPECT_FALSE(isAllowedFinalBoundaryOrdinal(
        "rtp_llm/cpp/cache/Types.h", "OtherRecord", identifierTokens("int32_t group_ordinal;", state)));
    state = {};
    EXPECT_FALSE(isAllowedFinalBoundaryOrdinal(
        "rtp_llm/cpp/cache/Types.h",
        "GroupOrdinalBlockIdPair",
        identifierTokens("int32_t extra_group_ordinal;", state)));
}

TEST(CacheTagRoutingSourceTest, ProjectedAndPrefixedCacheGroupSlotsAreForbidden) {
    IdentifierLexerState state;
    EXPECT_TRUE(hasProjectedCacheGroupSlotIdentity(identifierTokens(
        "std::unordered_map<std::string, size_t> tag_to_projected_slot;", state)));
    state = {};
    EXPECT_TRUE(hasProjectedCacheGroupSlotIdentity(identifierTokens("size_t private_tag_to_reordered_slot = 0;", state)));
    state = {};
    EXPECT_FALSE(hasProjectedCacheGroupSlotIdentity(identifierTokens("CPSlotMapper cp_slot_mapper;", state)));
}

TEST(CacheTagRoutingSourceTest, ScannerRejectsSameFileOrdinalAndPrefixedSlotMutations) {
    const char* tmpdir = std::getenv("TEST_TMPDIR");
    ASSERT_NE(tmpdir, nullptr);

    const fs::path ordinal_mutation = fs::path(tmpdir) / "cache_tag_routing_types_mutation.h";
    {
        std::ofstream output(ordinal_mutation);
        ASSERT_TRUE(output.is_open());
        output << "struct GroupOrdinalBlockIdPair {\n"
                  "    int32_t group_ordinal;\n"
                  "};\n"
                  "struct OtherRecord {\n"
                  "    int32_t group_ordinal;\n"
                  "};\n";
    }
    std::vector<Finding> ordinal_findings;
    scanFile(ordinal_mutation, "rtp_llm/cpp/cache/Types.h", ordinal_findings);
    ASSERT_EQ(ordinal_findings.size(), 1u);
    EXPECT_EQ(ordinal_findings.front().line_no, 5);
    EXPECT_NE(ordinal_findings.front().why.find("GroupOrdinalBlockIdPair::group_ordinal"), std::string::npos);

    const fs::path slot_mutation = fs::path(tmpdir) / "cache_tag_routing_slot_mutation.cc";
    {
        std::ofstream output(slot_mutation);
        ASSERT_TRUE(output.is_open());
        output << "size_t private_tag_to_projected_slot = 0;\n";
    }
    std::vector<Finding> slot_findings;
    scanFile(slot_mutation, "rtp_llm/cpp/cache/KVCacheManager.cc", slot_findings);
    ASSERT_EQ(slot_findings.size(), 1u);
    EXPECT_EQ(slot_findings.front().line_no, 1);
    EXPECT_NE(slot_findings.front().why.find("idx"), std::string::npos);

    std::error_code ec;
    fs::remove(ordinal_mutation, ec);
    ec.clear();
    fs::remove(slot_mutation, ec);
}

}  // namespace rtp_llm
