#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

using RequestPrefixKey = int64_t;

// Immutable, tagless view passed to connector match implementations.  It is
// intentionally independent of native group cache keys and block tables.
class RequestPrefixMatchView {
public:
    RequestPrefixMatchView(const std::vector<RequestPrefixKey>& keys,
                           size_t                               match_span_tokens,
                           size_t                               token_extent,
                           size_t                               match_limit_tokens,
                           size_t                               write_limit_tokens,
                           size_t                               reuse_tokens):
        keys_(&keys),
        match_span_tokens_(match_span_tokens),
        token_extent_(token_extent),
        match_limit_tokens_(match_limit_tokens),
        write_limit_tokens_(write_limit_tokens),
        reuse_tokens_(reuse_tokens) {}

    const std::vector<RequestPrefixKey>& keys() const {
        return *keys_;
    }
    size_t matchSpanTokens() const {
        return match_span_tokens_;
    }
    size_t tokenExtent() const {
        return token_extent_;
    }
    size_t matchLimitTokens() const {
        return match_limit_tokens_;
    }
    size_t writeLimitTokens() const {
        return write_limit_tokens_;
    }
    size_t reuseTokens() const {
        return reuse_tokens_;
    }

private:
    const std::vector<RequestPrefixKey>* keys_;
    size_t                               match_span_tokens_;
    size_t                               token_extent_;
    size_t                               match_limit_tokens_;
    size_t                               write_limit_tokens_;
    size_t                               reuse_tokens_;
};

// Request-owned connector control plane. Native tag-local cache keys and
// physical block resources remain in CacheGroupResource.
class RequestPrefixResource {
public:
    void configure(const CacheTopology& topology) {
        size_t span     = 1;
        bool   reusable = false;
        for (const auto& group : topology.groups()) {
            if (!group.policy.enable_prefix_reuse) {
                continue;
            }
            reusable         = true;
            const size_t gcd = std::gcd(span, group.seq_size_per_block);
            RTP_LLM_CHECK_WITH_INFO(span / gcd <= std::numeric_limits<size_t>::max() / group.seq_size_per_block,
                                    "request prefix match-span LCM overflow");
            span = span / gcd * group.seq_size_per_block;
        }
        match_span_tokens_ = reusable ? span : 1;
        reset();
    }

    void reset() {
        keys_.clear();
        token_extent_        = 0;
        match_limit_tokens_  = 0;
        write_limit_tokens_  = 0;
        device_reuse_tokens_ = 0;
        memory_reuse_tokens_ = 0;
        remote_reuse_tokens_ = 0;
    }

    void rebuild(int32_t* tokens, size_t token_count) {
        RTP_LLM_CHECK_WITH_INFO(tokens != nullptr || token_count == 0,
                                "RequestPrefixResource::rebuild received null tokens");
        keys_.clear();
        token_extent_       = token_count;
        match_limit_tokens_ = token_count == 0 ? 0 : ((token_count - 1) / match_span_tokens_) * match_span_tokens_;
        write_limit_tokens_ = (token_count / match_span_tokens_) * match_span_tokens_;
        keys_.reserve(write_limit_tokens_ / match_span_tokens_);
        RequestPrefixKey hash = 0;
        for (size_t begin = 0; begin < write_limit_tokens_; begin += match_span_tokens_) {
            hash = hashInt64Array(hash, tokens + begin, tokens + begin + match_span_tokens_);
            keys_.push_back(hash);
        }
    }

    RequestPrefixMatchView matchView() const {
        return RequestPrefixMatchView(
            keys_, match_span_tokens_, token_extent_, match_limit_tokens_, write_limit_tokens_, reuseTokens());
    }

    const std::vector<RequestPrefixKey>& keys() const {
        return keys_;
    }
    size_t matchSpanTokens() const {
        return match_span_tokens_;
    }
    size_t tokenExtent() const {
        return token_extent_;
    }
    size_t completePrefixEndpoint() const {
        return write_limit_tokens_;
    }
    size_t matchLimitTokens() const {
        return match_limit_tokens_;
    }
    size_t writeLimitTokens() const {
        return write_limit_tokens_;
    }

    size_t deviceReuseTokens() const {
        return device_reuse_tokens_;
    }
    size_t memoryReuseTokens() const {
        return memory_reuse_tokens_;
    }
    size_t remoteReuseTokens() const {
        return remote_reuse_tokens_;
    }
    size_t reuseTokens() const {
        return device_reuse_tokens_ + memory_reuse_tokens_ + remote_reuse_tokens_;
    }

    void setDeviceReuseTokens(size_t tokens) {
        device_reuse_tokens_ = tokens;
    }
    void setMemoryReuseTokens(size_t tokens) {
        memory_reuse_tokens_ = tokens;
    }
    void setRemoteReuseTokens(size_t tokens) {
        remote_reuse_tokens_ = tokens;
    }

private:
    size_t                        match_span_tokens_{1};
    std::vector<RequestPrefixKey> keys_;
    size_t                        token_extent_{0};
    size_t                        match_limit_tokens_{0};
    size_t                        write_limit_tokens_{0};
    size_t                        device_reuse_tokens_{0};
    size_t                        memory_reuse_tokens_{0};
    size_t                        remote_reuse_tokens_{0};
};

}  // namespace rtp_llm
