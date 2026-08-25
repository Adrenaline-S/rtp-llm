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

// Complete Phase 2 production surface. The test is local/no-sandbox so the
// canonical path of a staged source anchor leads back to the real checkout;
// this avoids a fragile, manually maintained filegroup per Bazel subpackage.
struct ProductionRoot {
    std::string relative;
    bool        internal = false;
};

std::vector<ProductionRoot> blockExpressionRoots(const fs::path& root) {
    std::vector<ProductionRoot> roots = {{"rtp_llm", false}};
    // Internal production is mandatory when this is an internal checkout, but
    // absent (or a broken public-layout symlink) in a standalone OSS checkout.
    if (fs::is_directory(root / "internal_source/rtp_llm")) {
        roots.push_back({"internal_source/rtp_llm", true});
    }
    // Optional downstream overlay. Canonical-file deduplication prevents the
    // common stub_source -> internal_source layout from scanning it twice.
    if (fs::is_directory(root / "stub_source/rtp_llm")) {
        roots.push_back({"stub_source/rtp_llm", true});
    }
    return roots;
}

const std::vector<std::string>& requiredPublicBlockExpressionFiles() {
    static const std::vector<std::string> required = {
        "rtp_llm/cpp/cache/CacheConfig.h",
        "rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc",
        "rtp_llm/cpp/engine_base/stream/StreamCacheResource.cc",
        "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h",
        "rtp_llm/cpp/model_rpc/DecodeRpcServer.cc",
        "rtp_llm/cpp/models/CacheBlockTablePacking.cc",
        "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.cc",
        "rtp_llm/models_py/bindings/OpDefs.h",
        "rtp_llm/models_py/modules/factory/attention/cuda_impl/py_flashinfer_mha.py",
        "rtp_llm/ops/librtp_compute_ops/__init__.pyi",
    };
    return required;
}

const std::vector<std::string>& requiredInternalBlockExpressionFiles() {
    static const std::vector<std::string> required = {
        "internal_source/rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaConnection.cc",
        "internal_source/rtp_llm/models_py/model_desc/__init__.py",
    };
    return required;
}

std::vector<std::string> blockExpressionCoverageFailures(const std::set<std::string>& scanned,
                                                         size_t                       public_scanned,
                                                         size_t                       internal_scanned,
                                                         bool                         internal_present) {
    std::vector<std::string> failures;
    if (public_scanned < 900) {
        failures.push_back("terminal public production-only block-expression source scan is unexpectedly narrow");
    }
    for (const auto& required : requiredPublicBlockExpressionFiles()) {
        if (scanned.count(required) != 1) {
            failures.push_back("terminal source gate did not scan required public file " + required);
        }
    }
    if (!internal_present) {
        return failures;
    }
    if (internal_scanned < 50) {
        failures.push_back("terminal internal production-only block-expression source scan is unexpectedly narrow");
    }
    for (const auto& required : requiredInternalBlockExpressionFiles()) {
        if (scanned.count(required) != 1) {
            failures.push_back("terminal source gate did not scan required internal file " + required);
        }
    }
    return failures;
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
        "rtp_llm/cpp/cache/CoordinatorCacheManager.cc",
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
        built.push_back({"positional cache group index as a variable, field, parameter or map key (" + kGroup + "_id)",
                         std::regex(kGroup + "_id")});
        built.push_back({"positional cache group accessor or helper (" + kGroup + "Id...)", std::regex(kGroup + "Id")});
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
            "GroupAllocationCheckpoint",
            "BatchAllocationCheckpoint",
            "AllocationRollbackJournal",
            "NativeTransferSelection",
            "NativeTransferSelections",
            "PhysicalBlockTransferPlan",
            "TaggedBlockPool",
            "TaggedSharedGroupEntry",
            "TaggedCacheItem",
        };
        for (const auto& type : forbidden_phase1_types) {
            built.push_back({"premature Phase 1 abstraction " + type, std::regex("$^"), type});
        }

        // A tag is the cache-group business identity. A vector index may be
        // named idx only where local storage needs one; none of these names
        // may reintroduce a cache-group slot/id abstraction.
        const std::vector<std::string> forbidden_identity_names = {
            "tag_to_slot",
            "group_slot",
            "groupIdForTag",
            "groupById",
            "layer_group_ids",
            "CacheTopology",
            "topologyPtr",
            "groupTagsInConfigOrder",
            std::string("KVCache") + "Allocator",
            std::string("HybridPoolKVCache") + "Allocator",
            std::string("KVCache") + "Group",
        };
        for (const auto& name : forbidden_identity_names) {
            built.push_back({"cache-group identity must remain tag-keyed (" + name + ")", std::regex("$^"), name});
        }
        built.push_back(
            {"cache metadata must be accessed directly through CacheConfig", std::regex(R"(\.topology\(\))")});
        return built;
    }();
    return all;
}

std::string workspaceRoot() {
    const char* srcdir = std::getenv("TEST_SRCDIR");
    if (srcdir == nullptr) {
        return {};
    }
    const char*    workspace = std::getenv("TEST_WORKSPACE");
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

std::string sourceWorkspaceRoot() {
    const fs::path runfiles_root(workspaceRoot());
    if (runfiles_root.empty()) {
        return {};
    }
    std::error_code ec;
    fs::path        anchor = fs::canonical(runfiles_root / "rtp_llm/cpp/cache/CacheConfig.h", ec);
    if (ec) {
        return {};
    }
    // CacheConfig.h -> cache -> cpp -> rtp_llm -> public workspace.
    fs::path root = anchor.parent_path().parent_path().parent_path().parent_path();
    if (!fs::exists(root / "WORKSPACE") || !fs::exists(root / "rtp_llm")) {
        return {};
    }
    return root.string();
}

bool isScannedExtension(const fs::path& path) {
    const std::string ext = path.extension().string();
    return ext == ".h" || ext == ".hpp" || ext == ".cc" || ext == ".cpp" || ext == ".cu" || ext == ".cuh"
           || ext == ".py" || ext == ".pyi";
}

const std::vector<std::string>& forbiddenBlockExpressionIdentifiers() {
    static const std::vector<std::string> identifiers = {
        "CacheTopology",
        "GroupBase",
        "BlockIds",
        "AttentionInputsByTag",
        "attention_inputs_by_tag",
        "kernel_block_indices_",
        "groupTagsInConfigOrder",
    };
    return identifiers;
}

bool isProductionSource(const fs::path& relative) {
    for (const auto& component : relative) {
        const auto name = component.string();
        if (name == "test" || name == "tests" || name == "testdata" || name == "golden" || name == "3rdparty"
            || name == "third_party" || name == "generated" || name == "build" || name == "__pycache__"
            || name.rfind("bazel-", 0) == 0) {
            return false;
        }
    }
    const auto filename = relative.filename().string();
    if (filename.find("Test.") != std::string::npos || filename.find("_test.") != std::string::npos
        || filename.rfind("test_", 0) == 0) {
        return false;
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
    bool in_block_comment    = false;
    char python_triple_quote = '\0';
};

std::vector<std::string_view>
identifierTokens(const std::string& line, IdentifierLexerState& state, bool python_source = false) {
    std::vector<std::string_view> tokens;
    for (size_t cursor = 0; cursor < line.size();) {
        if (state.python_triple_quote != '\0') {
            const std::string delimiter(3, state.python_triple_quote);
            const auto        triple_end = line.find(delimiter, cursor);
            if (triple_end == std::string::npos) {
                break;
            }
            state.python_triple_quote = '\0';
            cursor                    = triple_end + delimiter.size();
            continue;
        }
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
        if (python_source && line[cursor] == '#') {
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
        if (python_source && (line.compare(cursor, 3, "\"\"\"") == 0 || line.compare(cursor, 3, "'''") == 0)) {
            state.python_triple_quote = line[cursor];
            cursor += 3;
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

std::string sanitizedSourceLine(const std::string& line, IdentifierLexerState& state, bool python_source) {
    std::string sanitized;
    sanitized.reserve(line.size());
    for (size_t cursor = 0; cursor < line.size();) {
        if (state.python_triple_quote != '\0') {
            const std::string delimiter(3, state.python_triple_quote);
            const auto        triple_end = line.find(delimiter, cursor);
            if (triple_end == std::string::npos) {
                return sanitized;
            }
            state.python_triple_quote = '\0';
            cursor                    = triple_end + delimiter.size();
            sanitized.append(delimiter.size(), ' ');
            continue;
        }
        if (state.in_block_comment) {
            const auto comment_end = line.find("*/", cursor);
            if (comment_end == std::string::npos) {
                return sanitized;
            }
            state.in_block_comment = false;
            cursor                 = comment_end + 2;
            sanitized += "  ";
            continue;
        }
        if (line.compare(cursor, 2, "//") == 0 || (python_source && line[cursor] == '#')) {
            break;
        }
        if (line.compare(cursor, 2, "/*") == 0) {
            const auto comment_end = line.find("*/", cursor + 2);
            if (comment_end == std::string::npos) {
                state.in_block_comment = true;
                return sanitized;
            }
            cursor = comment_end + 2;
            sanitized += "  ";
            continue;
        }
        if (python_source && (line.compare(cursor, 3, "\"\"\"") == 0 || line.compare(cursor, 3, "'''") == 0)) {
            state.python_triple_quote = line[cursor];
            cursor += 3;
            sanitized += "   ";
            continue;
        }
        if (line[cursor] == '"' || line[cursor] == '\'') {
            const char quote = line[cursor++];
            sanitized += ' ';
            while (cursor < line.size()) {
                const char current = line[cursor++];
                sanitized += ' ';
                if (current == '\\' && cursor < line.size()) {
                    ++cursor;
                    sanitized += ' ';
                } else if (current == quote) {
                    break;
                }
            }
            continue;
        }
        sanitized += line[cursor++];
    }
    return sanitized;
}

std::string collapseWhitespace(std::string_view text) {
    std::string result;
    bool        pending_space = false;
    for (const char c : text) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            pending_space = !result.empty();
            continue;
        }
        if (pending_space) {
            result += ' ';
        }
        result += c;
        pending_space = false;
    }
    return result;
}

bool isPoolTokenSpan(std::string_view token) {
    if (token.find("kernel") != std::string_view::npos) {
        return false;
    }
    return token == "seq_size_per_block" || token == "tokens_per_block"
           || ((token.find("physical") != std::string_view::npos || token.find("pool") != std::string_view::npos)
               && (token.find("block") != std::string_view::npos || token.find("token") != std::string_view::npos
                   || token.find("size") != std::string_view::npos));
}

bool isKernelTokenSpan(std::string_view token) {
    return token.find("kernel") != std::string_view::npos
           && (token.find("block") != std::string_view::npos || token.find("size") != std::string_view::npos);
}

bool hasExplicitPoolToKernelArithmetic(const std::string& statement, const std::vector<std::string_view>& tokens) {
    for (size_t operator_pos = 0; operator_pos < statement.size(); ++operator_pos) {
        if (statement[operator_pos] != '/' && statement[operator_pos] != '%') {
            continue;
        }
        size_t left = tokens.size();
        for (size_t token_idx = 0; token_idx < tokens.size(); ++token_idx) {
            const size_t token_end =
                static_cast<size_t>(tokens[token_idx].data() - statement.data()) + tokens[token_idx].size();
            if (token_end <= operator_pos) {
                left = token_idx;
            } else {
                break;
            }
        }
        // The pool-width identifier must be the operand immediately to the
        // left. This prevents a preceding physical_block_num condition from
        // being paired with an unrelated `numel() % kernel_block_num`.
        if (left == tokens.size() || !isPoolTokenSpan(tokens[left])) {
            continue;
        }
        for (size_t right = left + 1; right < tokens.size(); ++right) {
            const size_t right_begin = static_cast<size_t>(tokens[right].data() - statement.data());
            if (right_begin <= operator_pos || !isKernelTokenSpan(tokens[right])) {
                continue;
            }
            const auto between = statement.substr(operator_pos + 1, right_begin - operator_pos - 1);
            if (between.find_first_of("+-*/%,;=&|?") == std::string::npos) {
                return true;
            }
            break;
        }
    }
    return false;
}

bool hasThreeDimensionalGlobalMaxAllocation(const std::vector<std::string_view>& tokens) {
    bool allocation = false;
    bool group      = false;
    bool batch      = false;
    bool maximum    = false;
    for (const auto token : tokens) {
        allocation = allocation || token == "empty" || token == "zeros" || token == "full" || token == "allocate"
                     || token == "resize" || token == "reshape";
        group   = group || token.find("group") != std::string_view::npos;
        batch   = batch || token.find("batch") != std::string_view::npos;
        maximum = maximum || token.find("global_max") != std::string_view::npos
                  || token.find("max_blocks") != std::string_view::npos
                  || token.find("max_block") != std::string_view::npos
                  || (token.find("max") != std::string_view::npos && token.find("width") != std::string_view::npos);
    }
    return allocation && group && batch && maximum;
}

bool isAuthoritativeProjectionSource(const std::string& relative) {
    return relative == "rtp_llm/cpp/cache/BlockExpression.cc";
}

bool isExactPhase3CpArithmetic(const std::string& statement, const std::vector<std::string_view>& tokens) {
    bool pool_tokens = false;
    bool cp_divisor  = false;
    bool kernel      = false;
    for (const auto token : tokens) {
        pool_tokens = pool_tokens || isPoolTokenSpan(token);
        cp_divisor  = cp_divisor || token == "cp_size" || token == "cp_size_" || token == "prefill_cp_size";
        kernel      = kernel || isKernelTokenSpan(token);
    }
    return pool_tokens && cp_divisor && !kernel
           && (statement.find('/') != std::string::npos || statement.find('%') != std::string::npos);
}

bool matchesRule(const Rule& rule, const std::string& line, const std::vector<std::string_view>& tokens) {
    return rule.identifier_fragment.empty() ? std::regex_search(line, rule.pattern) :
                                              containsIdentifierFragment(tokens, rule.identifier_fragment);
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
    std::string          line;
    int                  line_no = 0;
    std::string          active_record;
    size_t               allowed_boundary_ordinals = 0;
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

struct NormalizedStatementScanState {
    std::string text;
    int         start_line          = 0;
    size_t      line_span           = 0;
    int         bracket_depth       = 0;
    bool        bk_reported         = false;
    bool        allocation_reported = false;
};

void appendNormalizedStatementLine(NormalizedStatementScanState& state, const std::string& sanitized, int line_no) {
    const auto collapsed = collapseWhitespace(sanitized);
    if (collapsed.empty()) {
        return;
    }
    if (state.text.empty()) {
        state.start_line = line_no;
    } else {
        state.text += ' ';
    }
    state.text += collapsed;
    ++state.line_span;
    for (const char c : sanitized) {
        if (c == '(' || c == '[' || c == '{') {
            ++state.bracket_depth;
        } else if ((c == ')' || c == ']' || c == '}') && state.bracket_depth > 0) {
            --state.bracket_depth;
        }
    }
}

void scanBlockExpressionFile(const fs::path& absolute, const std::string& relative, std::vector<Finding>& findings) {
    std::ifstream input(absolute);
    if (!input.is_open()) {
        return;
    }
    std::string                  line;
    int                          line_no = 0;
    IdentifierLexerState         lexer_state;
    IdentifierLexerState         sanitizer_state;
    NormalizedStatementScanState statement_state;
    const bool                   python_source = absolute.extension() == ".py" || absolute.extension() == ".pyi";
    while (std::getline(input, line)) {
        ++line_no;
        const auto tokens = identifierTokens(line, lexer_state, python_source);
        for (const auto& identifier : forbiddenBlockExpressionIdentifiers()) {
            if (std::find(tokens.begin(), tokens.end(), identifier) != tokens.end()) {
                findings.push_back(
                    {relative, line_no, "transitional block-expression representation " + identifier, line});
            }
        }
        if (std::find(tokens.begin(), tokens.end(), "kernelBlocks") != tokens.end()) {
            findings.push_back({relative, line_no, "kernelBlocks() compatibility projection", line});
        }

        const auto sanitized = sanitizedSourceLine(line, sanitizer_state, python_source);
        appendNormalizedStatementLine(statement_state, sanitized, line_no);
        IdentifierLexerState statement_lexer_state;
        const auto statement_tokens = identifierTokens(statement_state.text, statement_lexer_state, python_source);
        if (!statement_state.bk_reported && hasExplicitPoolToKernelArithmetic(statement_state.text, statement_tokens)
            && !isAuthoritativeProjectionSource(relative)
            && !isExactPhase3CpArithmetic(statement_state.text, statement_tokens)) {
            findings.push_back({relative,
                                statement_state.start_line,
                                "pool-to-kernel B/K arithmetic outside the authoritative projection",
                                statement_state.text});
            statement_state.bk_reported = true;
        }
        if (!statement_state.allocation_reported && hasThreeDimensionalGlobalMaxAllocation(statement_tokens)) {
            findings.push_back({relative,
                                statement_state.start_line,
                                "packed block tables must not use a group/batch/global-max-width layout",
                                statement_state.text});
            statement_state.allocation_reported = true;
        }

        const bool statement_complete =
            sanitized.find(';') != std::string::npos
            || (python_source && statement_state.bracket_depth == 0 && (sanitized.empty() || sanitized.back() != '\\'))
            || statement_state.line_span >= 12 || statement_state.text.size() >= 2048;
        if (statement_complete) {
            statement_state = {};
        }
    }
}

}  // namespace

TEST(CacheTagRoutingSourceTest, ProductionSourcesCarryNoForbiddenPhase1AbstractionsOrPositionalGroupRouting) {
    const std::string root = workspaceRoot();
    ASSERT_FALSE(root.empty()) << "TEST_SRCDIR is unset; the source gate cannot locate the staged cache sources";

    std::set<std::string> scanned;
    std::vector<Finding>  findings;

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

TEST(CacheTagRoutingSourceTest, ProductionSourcesCarryNoTransitionalBlockExpressionRepresentations) {
    const std::string root = sourceWorkspaceRoot();
    ASSERT_FALSE(root.empty())
        << "source gate cannot resolve the real checkout from its staged CacheConfig.h anchor; this target must run "
           "locally without sandboxing";

    std::set<std::string> scanned;
    std::set<std::string> canonical_files;
    std::vector<Finding>  findings;
    size_t                public_scanned   = 0;
    size_t                internal_scanned = 0;
    const bool            internal_present = fs::is_directory(fs::path(root) / "internal_source/rtp_llm");
    for (const auto& production_root : blockExpressionRoots(root)) {
        const fs::path  absolute_root = fs::path(root) / production_root.relative;
        std::error_code ec;
        for (fs::recursive_directory_iterator it(absolute_root,
                                                 fs::directory_options::follow_directory_symlink
                                                     | fs::directory_options::skip_permission_denied,
                                                 ec);
             it != fs::recursive_directory_iterator();
             it.increment(ec)) {
            if (ec || !it->is_regular_file(ec) || ec || !isScannedExtension(it->path())) {
                continue;
            }
            const std::string relative  = it->path().lexically_relative(fs::path(root)).generic_string();
            const auto        canonical = fs::weakly_canonical(it->path(), ec).generic_string();
            if (ec || relative.empty() || relative.compare(0, 2, "..") == 0 || !isProductionSource(fs::path(relative))
                || !canonical_files.insert(canonical).second) {
                ec.clear();
                continue;
            }
            scanned.insert(relative);
            if (production_root.internal) {
                ++internal_scanned;
            } else {
                ++public_scanned;
            }
            scanBlockExpressionFile(it->path(), relative, findings);
        }
    }
    const auto coverage_failures =
        blockExpressionCoverageFailures(scanned, public_scanned, internal_scanned, internal_present);
    for (const auto& failure : coverage_failures) {
        ADD_FAILURE() << failure;
    }
    if (!findings.empty()) {
        std::ostringstream report;
        report << "transitional block-expression representations survive in " << findings.size() << " place(s):\n";
        for (const auto& finding : findings) {
            report << "  " << finding.relative_path << ":" << finding.line_no << ": " << finding.why << "\n"
                   << "      " << finding.line << "\n";
        }
        FAIL() << report.str();
    }
}

TEST(CacheTagRoutingSourceTest, ForbiddenIdentifierRulesCatchPrivateAndPrefixedSpellings) {
    const auto& all       = rules();
    const auto  find_rule = [&all](const std::string& fragment) -> const Rule& {
        const auto it = std::find_if(
            all.begin(), all.end(), [&fragment](const Rule& rule) { return rule.identifier_fragment == fragment; });
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
    const auto  it  = std::find_if(all.begin(), all.end(), [](const Rule& rule) {
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
        "rtp_llm/cpp/cache/CoordinatorCacheManager.cc", "", identifierTokens("size_t group_ordinal = 0;", state)));
    state = {};
    EXPECT_TRUE(isAllowedFinalBoundaryOrdinal(
        "rtp_llm/cpp/cache/Types.h", "GroupOrdinalBlockIdPair", identifierTokens("int32_t group_ordinal;", state)));
    state = {};
    EXPECT_FALSE(isAllowedFinalBoundaryOrdinal(
        "rtp_llm/cpp/cache/Types.h", "OtherRecord", identifierTokens("int32_t group_ordinal;", state)));
    state = {};
    EXPECT_FALSE(isAllowedFinalBoundaryOrdinal("rtp_llm/cpp/cache/Types.h",
                                               "GroupOrdinalBlockIdPair",
                                               identifierTokens("int32_t extra_group_ordinal;", state)));
}

TEST(CacheTagRoutingSourceTest, ProjectedAndPrefixedCacheGroupSlotsAreForbidden) {
    IdentifierLexerState state;
    EXPECT_TRUE(hasProjectedCacheGroupSlotIdentity(
        identifierTokens("std::unordered_map<std::string, size_t> tag_to_projected_slot;", state)));
    state = {};
    EXPECT_TRUE(
        hasProjectedCacheGroupSlotIdentity(identifierTokens("size_t private_tag_to_reordered_slot = 0;", state)));
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

TEST(CacheTagRoutingSourceTest, TerminalBlockExpressionRulesRejectTransitionalRepresentationsAndGlobalWidth) {
    const char* tmpdir = std::getenv("TEST_TMPDIR");
    ASSERT_NE(tmpdir, nullptr);

    const fs::path mutation = fs::path(tmpdir) / "block_expression_mutation.h";
    {
        std::ofstream output(mutation);
        ASSERT_TRUE(output.is_open());
        output << "BlockIds ids;\n"
                  "std::vector<std::string> kv_cache_group_tags;\n"
                  "auto table = torch::empty({\n"
                  "    group_count,\n"
                  "    batch_size,\n"
                  "    global_max_width});\n"
                  "auto pages = layout.kernelBlocks();\n"
                  "auto ratio = group.seq_size_per_block / group.kernel_seq_size_per_block;\n";
    }
    std::vector<Finding> findings;
    scanBlockExpressionFile(mutation, "rtp_llm/cpp/models/Mutation.h", findings);
    ASSERT_EQ(findings.size(), 4u);
    EXPECT_NE(findings[0].why.find("BlockIds"), std::string::npos);
    EXPECT_NE(findings[1].why.find("global-max-width"), std::string::npos);
    EXPECT_NE(findings[2].why.find("kernelBlocks"), std::string::npos);
    EXPECT_NE(findings[3].why.find("B/K"), std::string::npos);

    std::error_code ec;
    fs::remove(mutation, ec);
}

TEST(CacheTagRoutingSourceTest, PoolToKernelAuthorityRuleRejectsOpDefsAndAllowsProjectionImplementation) {
    const char* tmpdir = std::getenv("TEST_TMPDIR");
    ASSERT_NE(tmpdir, nullptr);

    const fs::path mutation = fs::path(tmpdir) / "pool_to_kernel_authority_mutation.h";
    {
        std::ofstream output(mutation);
        ASSERT_TRUE(output.is_open());
        output << "return group.seq_size_per_block / group.kernel_seq_size_per_block;\n";
    }

    std::vector<Finding> op_defs_findings;
    scanBlockExpressionFile(mutation, "rtp_llm/models_py/bindings/OpDefs.h", op_defs_findings);
    ASSERT_EQ(op_defs_findings.size(), 1u);
    EXPECT_NE(op_defs_findings.front().why.find("B/K"), std::string::npos);

    std::vector<Finding> authority_findings;
    scanBlockExpressionFile(mutation, "rtp_llm/cpp/cache/BlockExpression.cc", authority_findings);
    EXPECT_TRUE(authority_findings.empty());

    std::error_code ec;
    fs::remove(mutation, ec);
}

TEST(CacheTagRoutingSourceTest, TerminalRulesNormalizeMultilineArithmeticAndMaxBlockShapes) {
    const char* tmpdir = std::getenv("TEST_TMPDIR");
    ASSERT_NE(tmpdir, nullptr);

    const fs::path mutation = fs::path(tmpdir) / "multiline_block_expression_mutation.cc";
    {
        std::ofstream output(mutation);
        ASSERT_TRUE(output.is_open());
        output << "auto ratio = physical_tokens_per_block\n"
                  "             / kernel_tokens_per_block;\n"
                  "auto table = torch::empty({group_num, batch_size, max_blocks_num});\n"
                  "auto flat = allocate(group_count * batch_size * max_blocks_num);\n"
                  "const char* ignored = \"physical_tokens_per_block / kernel_tokens_per_block\";\n"
                  "/* physical_tokens_per_block\n"
                  "   / kernel_tokens_per_block; */\n";
    }
    std::vector<Finding> findings;
    scanBlockExpressionFile(mutation, "rtp_llm/cpp/engine_base/Mutation.cc", findings);
    ASSERT_EQ(findings.size(), 3u);
    EXPECT_NE(findings[0].why.find("B/K"), std::string::npos);
    EXPECT_NE(findings[1].why.find("global-max-width"), std::string::npos);
    EXPECT_NE(findings[2].why.find("global-max-width"), std::string::npos);

    std::error_code ec;
    fs::remove(mutation, ec);
}

TEST(CacheTagRoutingSourceTest, CpArithmeticAllowlistIsExpressionLevelNotFileWide) {
    const char* tmpdir = std::getenv("TEST_TMPDIR");
    ASSERT_NE(tmpdir, nullptr);

    const fs::path cp_mutation = fs::path(tmpdir) / "cp_arithmetic_mutation.cc";
    {
        std::ofstream output(cp_mutation);
        ASSERT_TRUE(output.is_open());
        output << "auto canonical = physical_tokens_per_block / prefill_cp_size;\n";
    }
    std::vector<Finding> cp_findings;
    scanBlockExpressionFile(cp_mutation, "rtp_llm/cpp/model_rpc/DecodeRpcServer.cc", cp_findings);
    EXPECT_TRUE(cp_findings.empty());
    IdentifierLexerState cp_lexer_state;
    const std::string    cp_statement = "physical_tokens_per_block / prefill_cp_size";
    EXPECT_TRUE(isExactPhase3CpArithmetic(cp_statement, identifierTokens(cp_statement, cp_lexer_state)));

    const fs::path bk_mutation = fs::path(tmpdir) / "mixed_purpose_bk_mutation.cc";
    {
        std::ofstream output(bk_mutation);
        ASSERT_TRUE(output.is_open());
        output << "auto ratio = physical_tokens_per_block\n"
                  "             % kernel_tokens_per_block;\n";
    }
    std::vector<Finding> bk_findings;
    scanBlockExpressionFile(bk_mutation, "rtp_llm/cpp/model_rpc/DecodeRpcServer.cc", bk_findings);
    ASSERT_EQ(bk_findings.size(), 1u);
    EXPECT_NE(bk_findings.front().why.find("B/K"), std::string::npos);
    IdentifierLexerState bk_lexer_state;
    const std::string    bk_statement = "physical_tokens_per_block % kernel_tokens_per_block";
    EXPECT_FALSE(isExactPhase3CpArithmetic(bk_statement, identifierTokens(bk_statement, bk_lexer_state)));

    std::error_code ec;
    fs::remove(cp_mutation, ec);
    ec.clear();
    fs::remove(bk_mutation, ec);
}

TEST(CacheTagRoutingSourceTest, StandalonePublicRootDoesNotRequireInternalSource) {
    const char* tmpdir = std::getenv("TEST_TMPDIR");
    ASSERT_NE(tmpdir, nullptr);

    const fs::path  standalone = fs::path(tmpdir) / "standalone_root_discovery";
    std::error_code ec;
    fs::remove_all(standalone, ec);
    ec.clear();
    ASSERT_TRUE(fs::create_directories(standalone / "rtp_llm/cpp", ec));
    ASSERT_FALSE(ec);
    fs::create_directory_symlink(standalone / "missing_internal", standalone / "internal_source", ec);
    ASSERT_FALSE(ec);

    const auto standalone_roots = blockExpressionRoots(standalone);
    ASSERT_EQ(standalone_roots.size(), 1u);
    EXPECT_EQ(standalone_roots.front().relative, "rtp_llm");
    EXPECT_FALSE(standalone_roots.front().internal);

    std::set<std::string> standalone_scanned(requiredPublicBlockExpressionFiles().begin(),
                                             requiredPublicBlockExpressionFiles().end());
    EXPECT_TRUE(blockExpressionCoverageFailures(standalone_scanned, 900, 0, false).empty());
    EXPECT_FALSE(blockExpressionCoverageFailures(standalone_scanned, 900, 0, true).empty());

    ASSERT_TRUE(fs::create_directories(standalone / "real_internal/rtp_llm", ec));
    ASSERT_FALSE(ec);
    fs::remove(standalone / "internal_source", ec);
    ASSERT_FALSE(ec);
    fs::create_directory_symlink(standalone / "real_internal", standalone / "internal_source", ec);
    ASSERT_FALSE(ec);
    const auto internal_roots = blockExpressionRoots(standalone);
    ASSERT_EQ(internal_roots.size(), 2u);
    EXPECT_EQ(internal_roots[1].relative, "internal_source/rtp_llm");
    EXPECT_TRUE(internal_roots[1].internal);

    fs::remove_all(standalone, ec);
}

}  // namespace rtp_llm
