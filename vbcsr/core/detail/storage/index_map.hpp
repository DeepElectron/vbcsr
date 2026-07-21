#ifndef VBCSR_DETAIL_STORAGE_INDEX_MAP_HPP
#define VBCSR_DETAIL_STORAGE_INDEX_MAP_HPP

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vbcsr::detail {

// Flat, open-addressed map from a non-negative integer key to an int value,
// used for the global-index -> local-index tables that every distributed
// operation probes.
//
// API: a deliberate subset of std::map<int, int> — at(), find()/end(),
// count(), operator[], size(), empty(), clear(), reserve() all keep their
// standard meaning, and find() yields a handle with ->first / ->second. Call
// sites therefore read exactly as they did with the map they replace.
//
// What it intentionally does NOT provide is iteration. This container has no
// key order, and leaving begin()/range-for out turns "someone later depends on
// the order" into a compile error. (Swapping in std::unordered_map instead
// would have compiled and changed iteration order silently.)
//
// Storage is one contiguous array, so building the table costs no per-entry
// allocation and probing stays in cache — which is the point: the tree-based
// map it replaces dominated distributed result-graph construction.
//
// Contract: keys must be >= 0 (kEmptyKey marks a free slot). References
// returned by operator[] are invalidated by any later insertion that grows the
// table, exactly like std::unordered_map's rehash rule.
class IndexMap {
public:
    struct Entry {
        int first;   // key; kEmptyKey when the slot is free
        int second;  // value
    };

    using iterator = Entry*;
    using const_iterator = const Entry*;

    static constexpr int kEmptyKey = -1;

    IndexMap() = default;

    void clear() {
        for (auto& slot : slots_) {
            slot.first = kEmptyKey;
        }
        size_ = 0;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    // Sentinel only: comparing against end() is how absence is reported. There
    // is deliberately no begin() to pair it with.
    iterator end() { return nullptr; }
    const_iterator end() const { return nullptr; }

    void reserve(size_t entries) {
        // Keep the load factor at or below 0.7.
        size_t needed = 1;
        while (needed * 7 < entries * 10) {
            needed <<= 1;
        }
        if (needed > slots_.size()) {
            rehash(needed);
        }
    }

    iterator find(int key) {
        if (slots_.empty()) {
            return end();
        }
        size_t slot = probe(key);
        return slots_[slot].first == key ? &slots_[slot] : end();
    }

    const_iterator find(int key) const {
        if (slots_.empty()) {
            return end();
        }
        size_t slot = probe(key);
        return slots_[slot].first == key ? &slots_[slot] : end();
    }

    size_t count(int key) const { return find(key) != end() ? 1 : 0; }

    int& at(int key) {
        iterator it = find(key);
        if (it == end()) {
            throw std::out_of_range("IndexMap::at: key not present");
        }
        return it->second;
    }

    const int& at(int key) const {
        const_iterator it = find(key);
        if (it == end()) {
            throw std::out_of_range("IndexMap::at: key not present");
        }
        return it->second;
    }

    int& operator[](int key) {
        if (needs_growth()) {
            rehash(slots_.empty() ? 16 : slots_.size() * 2);
        }
        size_t slot = probe(key);
        if (slots_[slot].first != key) {
            slots_[slot].first = key;
            slots_[slot].second = 0;
            ++size_;
        }
        return slots_[slot].second;
    }

private:
    static size_t hash(int key) {
        // Multiplicative mixing, high bits taken: sequential and scattered
        // global indices both spread evenly over a power-of-two table.
        const uint64_t mixed =
            static_cast<uint64_t>(static_cast<uint32_t>(key)) * 0x9E3779B97F4A7C15ull;
        return static_cast<size_t>(mixed >> 32);
    }

    bool needs_growth() const { return (size_ + 1) * 10 >= slots_.size() * 7; }

    // Returns the slot holding `key`, or the first free slot where it belongs.
    size_t probe(int key) const {
        const size_t mask = slots_.size() - 1;
        size_t slot = hash(key) & mask;
        while (slots_[slot].first != kEmptyKey && slots_[slot].first != key) {
            slot = (slot + 1) & mask;
        }
        return slot;
    }

    void rehash(size_t capacity) {
        std::vector<Entry> moved(capacity, Entry{kEmptyKey, 0});
        moved.swap(slots_);
        size_ = 0;
        for (const auto& entry : moved) {
            if (entry.first != kEmptyKey) {
                (*this)[entry.first] = entry.second;
            }
        }
    }

    std::vector<Entry> slots_;
    size_t size_ = 0;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_STORAGE_INDEX_MAP_HPP
