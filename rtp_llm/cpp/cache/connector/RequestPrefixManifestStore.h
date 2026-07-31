#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/RequestPrefixResource.h"

namespace rtp_llm {

struct RequestPrefixManifestKey {
    RequestPrefixKey prefix_hash{0};
    size_t           token_end{0};

    bool operator==(const RequestPrefixManifestKey& other) const {
        return prefix_hash == other.prefix_hash && token_end == other.token_end;
    }
    bool operator!=(const RequestPrefixManifestKey& other) const {
        return !(*this == other);
    }
};

struct RequestPrefixManifestKeyHash {
    size_t operator()(const RequestPrefixManifestKey& key) const {
        const size_t h1 = std::hash<RequestPrefixKey>()(key.prefix_hash);
        const size_t h2 = std::hash<size_t>()(key.token_end);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

enum class NativeCacheItemKind : uint8_t {
    FULL_INTERVAL = 0,
    ACTIVE_TAIL   = 1,
    STATE         = 2,
};

struct NativeCacheItemRef {
    std::string         tag;
    int64_t             native_cache_key{0};
    uint32_t            physical_ordinal{0};
    NativeCacheItemKind kind{NativeCacheItemKind::FULL_INTERVAL};
    uint32_t            slot{0};
};

struct RequestPrefixManifest {
    RequestPrefixManifestKey                key;
    std::optional<RequestPrefixManifestKey> parent;
    std::vector<NativeCacheItemRef>         native_items;
    // Keeps native backing leases alive for exactly the manifest lifetime.
    std::vector<std::shared_ptr<void>> native_backing_holds;
};

class RequestPrefixManifestStore: public std::enable_shared_from_this<RequestPrefixManifestStore> {
public:
    class PinnedChain {
    public:
        ~PinnedChain();
        PinnedChain(const PinnedChain&)            = delete;
        PinnedChain& operator=(const PinnedChain&) = delete;

        size_t                                    matchedTokenCount() const;
        const std::vector<RequestPrefixManifest>& manifests() const {
            return manifests_;
        }

    private:
        friend class RequestPrefixManifestStore;
        PinnedChain(std::weak_ptr<RequestPrefixManifestStore> store,
                    std::vector<RequestPrefixManifestKey>     keys,
                    std::vector<RequestPrefixManifest>        manifests):
            store_(std::move(store)), keys_(std::move(keys)), manifests_(std::move(manifests)) {}

        std::weak_ptr<RequestPrefixManifestStore> store_;
        std::vector<RequestPrefixManifestKey>     keys_;
        std::vector<RequestPrefixManifest>        manifests_;
    };

    bool                         publish(RequestPrefixManifest manifest);
    std::shared_ptr<PinnedChain> match(const RequestPrefixMatchView& view, size_t start_token);
    bool                         evict(const RequestPrefixManifestKey& key);
    size_t                       evictAllUnpinned();
    size_t                       visibleSize() const;
    size_t                       pinCount(const RequestPrefixManifestKey& key) const;

private:
    struct Entry {
        RequestPrefixManifest manifest;
        size_t                pins{0};
        bool                  visible{true};
    };

    void release(const std::vector<RequestPrefixManifestKey>& keys);

    mutable std::mutex                                                                mutex_;
    std::unordered_map<RequestPrefixManifestKey, Entry, RequestPrefixManifestKeyHash> entries_;
};

}  // namespace rtp_llm
