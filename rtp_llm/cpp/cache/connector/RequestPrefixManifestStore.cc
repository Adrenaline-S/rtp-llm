#include "rtp_llm/cpp/cache/connector/RequestPrefixManifestStore.h"

#include <algorithm>

namespace rtp_llm {

RequestPrefixManifestStore::PinnedChain::~PinnedChain() {
    if (auto store = store_.lock()) {
        store->release(keys_);
    }
}

size_t RequestPrefixManifestStore::PinnedChain::matchedTokenCount() const {
    return manifests_.empty() ? 0 : manifests_.back().key.token_end;
}

bool RequestPrefixManifestStore::publish(RequestPrefixManifest manifest) {
    if (manifest.key.token_end == 0 || manifest.native_items.empty()) {
        return false;
    }
    if (manifest.parent.has_value() && manifest.parent->token_end >= manifest.key.token_end) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (manifest.parent.has_value()) {
        const auto parent = entries_.find(*manifest.parent);
        if (parent == entries_.end() || !parent->second.visible) {
            return false;
        }
    }
    auto found = entries_.find(manifest.key);
    if (found != entries_.end() && found->second.pins != 0) {
        return false;
    }
    entries_.insert_or_assign(manifest.key, Entry{std::move(manifest), 0, true});
    return true;
}

std::shared_ptr<RequestPrefixManifestStore::PinnedChain>
RequestPrefixManifestStore::match(const RequestPrefixMatchView& view, size_t start_token) {
    const size_t span = view.matchSpanTokens();
    if (span == 0 || start_token % span != 0 || start_token >= view.matchLimitTokens()) {
        return nullptr;
    }

    std::vector<RequestPrefixManifestKey>   pinned_keys;
    std::vector<RequestPrefixManifest>      manifests;
    std::lock_guard<std::mutex>             lock(mutex_);
    std::optional<RequestPrefixManifestKey> expected_parent;
    if (start_token > 0) {
        const size_t parent_index = start_token / span - 1;
        if (parent_index >= view.keys().size()) {
            return nullptr;
        }
        expected_parent = RequestPrefixManifestKey{view.keys()[parent_index], start_token};
    }
    for (size_t endpoint = start_token + span; endpoint <= view.matchLimitTokens(); endpoint += span) {
        const size_t index = endpoint / span - 1;
        if (index >= view.keys().size()) {
            break;
        }
        const RequestPrefixManifestKey key{view.keys()[index], endpoint};
        auto                           found = entries_.find(key);
        if (found == entries_.end() || !found->second.visible || found->second.manifest.parent != expected_parent) {
            break;
        }
        ++found->second.pins;
        pinned_keys.push_back(key);
        manifests.push_back(found->second.manifest);
        expected_parent = key;
    }
    if (manifests.empty()) {
        return nullptr;
    }
    return std::shared_ptr<PinnedChain>(
        new PinnedChain(weak_from_this(), std::move(pinned_keys), std::move(manifests)));
}

bool RequestPrefixManifestStore::evict(const RequestPrefixManifestKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        found = entries_.find(key);
    if (found == entries_.end()) {
        return false;
    }
    found->second.visible = false;
    if (found->second.pins == 0) {
        entries_.erase(found);
    }
    return true;
}

size_t RequestPrefixManifestStore::evictAllUnpinned() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      evicted = 0;
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.visible) {
            it->second.visible = false;
            ++evicted;
        }
        if (it->second.pins == 0) {
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
    return evicted;
}

size_t RequestPrefixManifestStore::visibleSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::count_if(entries_.begin(), entries_.end(), [](const auto& item) { return item.second.visible; });
}

size_t RequestPrefixManifestStore::pinCount(const RequestPrefixManifestKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  found = entries_.find(key);
    return found == entries_.end() ? 0 : found->second.pins;
}

void RequestPrefixManifestStore::release(const std::vector<RequestPrefixManifestKey>& keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& key : keys) {
        auto found = entries_.find(key);
        if (found == entries_.end()) {
            continue;
        }
        if (found->second.pins > 0) {
            --found->second.pins;
        }
        if (!found->second.visible && found->second.pins == 0) {
            entries_.erase(found);
        }
    }
}

}  // namespace rtp_llm
