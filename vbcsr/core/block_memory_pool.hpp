#ifndef VBCSR_BLOCK_MEMORY_POOL_HPP
#define VBCSR_BLOCK_MEMORY_POOL_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace vbcsr {

template <typename T>
struct PayloadPageView {
    T* data = nullptr;
    size_t size = 0;
    size_t page_index = 0;
};

template <typename T>
class BlockArena {
public:
    // Default page size: 1M elements (approx 8MB for double)
    // This is small enough for tests but large enough to amortize allocation.
    static constexpr size_t DEFAULT_PAGE_SIZE = 1ULL << 22; 

    BlockArena() : current_page_idx(0), current_offset(0) {
        // Initialize with one empty page? 
        // Or wait for first allocation. Let's wait.
    }

    // Disable copy
    BlockArena(const BlockArena&) = delete;
    BlockArena& operator=(const BlockArena&) = delete;

    // Move constructor
    BlockArena(BlockArena&& other) noexcept 
        : pages(std::move(other.pages)), 
          free_blocks(std::move(other.free_blocks)),
          current_page_idx(other.current_page_idx),
          current_offset(other.current_offset) {
        other.current_page_idx = 0;
        other.current_offset = 0;
    }

    // Move assignment
    BlockArena& operator=(BlockArena&& other) noexcept {
        if (this != &other) {
            pages = std::move(other.pages);
            free_blocks = std::move(other.free_blocks);
            current_page_idx = other.current_page_idx;
            current_offset = other.current_offset;
            other.current_page_idx = 0;
            other.current_offset = 0;
        }
        return *this;
    }

    // Allocate a block of 'size' elements
    // Returns a handle (uint64_t)
    uint64_t allocate(size_t size) {
        // 1. Check freelist for exact match
        auto it = free_blocks.find(size);
        if (it != free_blocks.end() && !it->second.empty()) {
            uint64_t handle = it->second.back();
            it->second.pop_back();
            return handle;
        }

        // 2. Allocate from current page
        if (pages.empty()) {
            allocate_new_page(std::max(size, DEFAULT_PAGE_SIZE));
        }

        // Check if fits in current page
        if (current_offset + size > pages[current_page_idx].size) {
            // Move to new page
            allocate_new_page(std::max(size, DEFAULT_PAGE_SIZE));
        }

        uint64_t handle = (static_cast<uint64_t>(current_page_idx) << 32) | static_cast<uint64_t>(current_offset);
        current_offset += size;
        return handle;
    }

    // Get pointer from handle
    // Inline for performance
    inline T* get_ptr(uint64_t handle) const {
        uint32_t page_idx = static_cast<uint32_t>(handle >> 32);
        uint32_t offset = static_cast<uint32_t>(handle & 0xFFFFFFFF);
        return pages[page_idx].data.get() + offset;
    }

    // Free a block
    void free(uint64_t handle, size_t size) {
        // zero out the block
        T* ptr = get_ptr(handle);
        std::fill(ptr, ptr + size, T(0));
        free_blocks[size].push_back(handle);
    }

    // Clear all memory
    void clear() {
        pages.clear();
        free_blocks.clear();
        current_page_idx = 0;
        current_offset = 0;
    }

    // Reserve memory for at least 'total_elements'
    void reserve(unsigned long long total_elements) {
        size_t current_capacity = 0;
        for (const auto& page : pages) {
            current_capacity += page.size;
        }
        
        if (total_elements <= current_capacity) return;
        
        unsigned long long needed = total_elements - current_capacity;
        
        // Allocate full pages
        while (needed > 0) {
            size_t alloc_size = std::min((unsigned long long)DEFAULT_PAGE_SIZE, needed);
            // Actually, we should just allocate DEFAULT_PAGE_SIZE chunks until covered
            // But to avoid fragmentation if needed is small? No, reserve implies big.
            // Let's just allocate DEFAULT_PAGE_SIZE pages.
            // Wait, if needed > DEFAULT_PAGE_SIZE, we loop.
            
            allocate_new_page(DEFAULT_PAGE_SIZE);
            if (needed < DEFAULT_PAGE_SIZE) needed = 0;
            else needed -= DEFAULT_PAGE_SIZE;
        }
    }

    size_t total_allocated_bytes() const {
        size_t total = 0;
        for (const auto& page : pages) {
            total += page.size * sizeof(T);
        }
        return total;
    }

    size_t page_count() const {
        return pages.size();
    }

    PayloadPageView<T> page_view(size_t page_idx) {
        if (page_idx >= pages.size()) {
            throw std::out_of_range("BlockArena::page_view index out of range");
        }
        return PayloadPageView<T>{pages[page_idx].data.get(), pages[page_idx].size, page_idx};
    }

    PayloadPageView<const T> page_view(size_t page_idx) const {
        if (page_idx >= pages.size()) {
            throw std::out_of_range("BlockArena::page_view index out of range");
        }
        return PayloadPageView<const T>{pages[page_idx].data.get(), pages[page_idx].size, page_idx};
    }

    template <typename Fn>
    void for_each_page(Fn&& fn) {
        for (size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
            fn(page_view(page_idx));
        }
    }

    template <typename Fn>
    void for_each_page(Fn&& fn) const {
        for (size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
            fn(page_view(page_idx));
        }
    }

private:
    struct Page {
        std::unique_ptr<T[]> data;
        size_t size;
    };

    std::vector<Page> pages;
    std::unordered_map<size_t, std::vector<uint64_t>> free_blocks;
    
    uint32_t current_page_idx;
    uint32_t current_offset;

    void allocate_new_page(size_t size) {
        if (!pages.empty()) {
            current_page_idx++;
        }
        Page new_page;
        new_page.size = size;
        new_page.data = std::make_unique<T[]>(size);
        std::fill(new_page.data.get(), new_page.data.get() + size, T(0));
        pages.push_back(std::move(new_page));
        current_offset = 0;
    }
};

} // namespace vbcsr

#endif // VBCSR_BLOCK_MEMORY_POOL_HPP
