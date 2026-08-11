#ifndef VBCSR_DETAIL_STORAGE_PAGED_ARRAY_HPP
#define VBCSR_DETAIL_STORAGE_PAGED_ARRAY_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "numa_buffer.hpp"

namespace vbcsr::detail {

template <typename T>
struct BSRMatrixBackend;

template <typename T>
struct CSRMatrixBackend;

template <typename T>
class ShapeBlockStore;

template <typename T>
struct PageSlice {
    T* data = nullptr;
    uint32_t count = 0;
    uint32_t page_index = 0;
    uint64_t first_element = 0;
};

template <typename T>
class PagedBuffer {
    friend struct BSRMatrixBackend<T>;
    friend struct CSRMatrixBackend<T>;
    friend class ShapeBlockStore<T>;

public:
    // Page payloads use the shared fresh-buffer allocator (numa_buffer.hpp):
    // mmap-backed for trivial scalars so the zero-filling domain thread is
    // always the first toucher; plain new[] otherwise.
    struct Page {
        FreshBufferOwner<T> data;
        uint32_t used = 0;
    };

    PagedBuffer()
        : elements_per_page_(default_elements_per_page()) {}

    explicit PagedBuffer(uint32_t elements_per_page)
        : elements_per_page_(std::max<uint32_t>(1, elements_per_page)) {}

    uint64_t size() const {
        return size_;
    }

    uint64_t capacity() const {
        return static_cast<uint64_t>(pages_.size()) * static_cast<uint64_t>(elements_per_page_);
    }

    bool empty() const {
        return size_ == 0;
    }

    uint32_t page_count() const {
        return static_cast<uint32_t>(pages_.size());
    }

    uint32_t elements_per_page() const {
        return elements_per_page_;
    }

    void clear() {
        pages_.clear();
        size_ = 0;
        released_prefix_ = false;
        released_through_ = 0;
    }

    // Standard reserve semantics: capacity beyond size() is not observable,
    // so the new pages are not zero-filled (resize() zero-fills the live
    // range on growth itself).
    void reserve(uint64_t element_capacity) {
        assert_growable("reserve");
        ensure_capacity_uninitialized(element_capacity);
    }

    void resize(uint64_t element_count) {
        assert_growable("resize");
        const uint64_t old_capacity = capacity();
        if (element_count > capacity()) {
            ensure_capacity(element_count);
        }
        if (element_count > size_) {
            // Only the part of the new range that lands in pages which ALREADY
            // existed can hold stale values. Pages appended just now are zero
            // on arrival -- anonymous mmap is kernel-zeroed, and the heap path
            // fills on append -- so clearing them again is a second full pass
            // over the whole buffer, and worse, it faults in every page of an
            // allocation whose entire point was to stay untouched until
            // written. On a matrix grown from empty this was zeroing tens of
            // GB twice before a single value existed.
            const uint64_t stale_end = std::min(element_count, old_capacity);
            if (stale_end > size_) {
                zero_fill_range(size_, stale_end);
            }
        }
        size_ = element_count;
        refresh_page_usage();
    }

    T& operator[](uint64_t index) {
        return *element_ptr(index);
    }

    const T& operator[](uint64_t index) const {
        return *element_ptr(index);
    }

    T* element_ptr(uint64_t index) {
        auto [page_index, offset] = locate(index);
        return pages_[page_index].data.get() + offset;
    }

    const T* element_ptr(uint64_t index) const {
        auto [page_index, offset] = locate(index);
        return pages_[page_index].data.get() + offset;
    }

    PageSlice<T> page(uint32_t page_index) {
        const auto& storage_page = require_page(page_index);
        return PageSlice<T>{
            storage_page.data.get(),
            storage_page.used,
            page_index,
            static_cast<uint64_t>(page_index) * static_cast<uint64_t>(elements_per_page_)};
    }

    PageSlice<const T> page(uint32_t page_index) const {
        const auto& storage_page = require_page(page_index);
        return PageSlice<const T>{
            storage_page.data.get(),
            storage_page.used,
            page_index,
            static_cast<uint64_t>(page_index) * static_cast<uint64_t>(elements_per_page_)};
    }

    template <typename Fn>
    void for_each_range(uint64_t begin, uint64_t end, Fn&& fn) {
        walk_range(*this, begin, end, std::forward<Fn>(fn));
    }

    template <typename Fn>
    void for_each_range(uint64_t begin, uint64_t end, Fn&& fn) const {
        walk_range(*this, begin, end, std::forward<Fn>(fn));
    }

    template <typename U, typename Fn>
    void for_each_zipped_range(PagedBuffer<U>& other, uint64_t begin, uint64_t end, Fn&& fn) {
        walk_zipped_range(*this, other, begin, end, std::forward<Fn>(fn));
    }

    template <typename U, typename Fn>
    void for_each_zipped_range(const PagedBuffer<U>& other, uint64_t begin, uint64_t end, Fn&& fn) {
        walk_zipped_range(*this, other, begin, end, std::forward<Fn>(fn));
    }

    template <typename U, typename Fn>
    void for_each_zipped_range(const PagedBuffer<U>& other, uint64_t begin, uint64_t end, Fn&& fn) const {
        walk_zipped_range(*this, other, begin, end, std::forward<Fn>(fn));
    }

    template <typename U>
    void copy_prefix_from(const PagedBuffer<U>& other, uint64_t count) {
        if (count > other.size()) {
            throw std::out_of_range("PagedBuffer::copy_prefix_from count out of bounds");
        }
        if (size() < count) {
            resize(count);
        }
        if (count == 0) {
            return;
        }
        if constexpr (std::is_same_v<T, U>) {
            if (elements_per_page() == other.elements_per_page()) {
            for_each_zipped_range(other, 0, count, [](auto dst, auto src) {
                std::memcpy(dst.data, src.data, static_cast<size_t>(dst.count) * sizeof(T));
            });
            return;
        }
        }
        for (uint64_t idx = 0; idx < count; ++idx) {
            *element_ptr(idx) = static_cast<T>(*other.element_ptr(idx));
        }
    }

private:
    // Frees every page lying entirely below `element_index`, and returns the
    // bytes handed back. The page SLOTS stay, so every element index above the
    // boundary keeps its address; only the storage under the released prefix
    // goes away. Reading a released element is a programming error, not a
    // resize -- release builds cannot afford to detect it, so this is private
    // and reachable only from a backend that owns the traversal order, and
    // debug builds trip assert_not_released on the read paths.
    //
    // This is what lets an in-place product shrink its input as it consumes it:
    // pages are independent allocations, and on the mmap path the release is a
    // munmap, so resident memory actually drops instead of returning to an
    // allocator free list where it would still count against the job.
    uint64_t release_pages_before(uint64_t element_index) {
        if (element_index > size_) {
            element_index = size_;
        }
        if (element_index == 0) {
            return 0;
        }
        uint64_t freed_bytes = 0;
#ifdef __linux__
        // MADV_DONTNEED, not munmap, and it works on the OS page rather than
        // this buffer's page. That distinction is the whole reason this
        // function does anything: a store sizes its pages to hold the entire
        // matrix where it can -- BSR's cap is UINT32_MAX/block_values, some 25
        // million 13x13 blocks -- so "free the pages lying entirely below the
        // boundary" frees nothing at all until the last one, which is too late
        // to be worth doing. Advising the aligned prefix hands back physical
        // memory at 4 KB granularity while the mapping and every pointer into
        // it stay exactly as they were.
        //
        // Only anonymous mmap pages qualify. On the heap path there is no way
        // to return part of an allocation, so nothing happens and the caller
        // simply keeps the memory -- correct, just not smaller.
        const long os_page = ::sysconf(_SC_PAGESIZE);
        if (os_page <= 0) return 0;
        const uintptr_t mask = static_cast<uintptr_t>(os_page) - 1;
        // Only the NEWLY dead range. A consuming product calls this once per
        // chunk with a growing boundary, and starting from page 0 every time
        // re-advised every previously released page -- quadratic syscall work
        // over memory that was already gone, and it reported those bytes as
        // freed again.
        if (element_index <= released_through_) return 0;
        uint64_t consumed = 0;
        for (uint32_t page = 0; page < pages_.size() && consumed < element_index; ++page) {
            const uint64_t page_begin = consumed;
            const uint64_t page_end = consumed + elements_per_page_;
            consumed = page_end;
            const uint64_t lo = std::max<uint64_t>(released_through_, page_begin);
            const uint64_t hi = std::min<uint64_t>(element_index, page_end);
            if (lo >= hi) continue;
            Page& storage_page = pages_[page];
            if (storage_page.data && storage_page.data.get_deleter().mmap_bytes != 0) {
                const uintptr_t base =
                    reinterpret_cast<uintptr_t>(storage_page.data.get());
                const uintptr_t begin = base + static_cast<size_t>(lo - page_begin) * sizeof(T);
                const uintptr_t end = base + static_cast<size_t>(hi - page_begin) * sizeof(T);
                const uintptr_t aligned_begin = (begin + mask) & ~mask;
                const uintptr_t aligned_end = end & ~mask;
                if (aligned_end > aligned_begin) {
                    ::madvise(reinterpret_cast<void*>(aligned_begin),
                              static_cast<size_t>(aligned_end - aligned_begin),
                              MADV_DONTNEED);
                    freed_bytes += aligned_end - aligned_begin;
                }
            }
        }
        released_through_ = element_index;
#else
        (void)element_index;
#endif
        released_prefix_ = true;
        return freed_bytes;
    }

    // A buffer with released pages still REPORTS the capacity of the slots it
    // kept, and the mapping is still there (MADV_DONTNEED keeps it; only the
    // physical pages go), so a later grow would write into memory whose
    // contents have silently reverted to zero-fill
    // somewhere unrelated. Growing one is always a bug -- release is for
    // storage about to be dropped -- so say that here rather than let it
    // become a crash in whatever touched it next.
    void assert_growable(const char* what) const {
        if (released_prefix_) {
            throw std::logic_error(
                std::string("PagedBuffer::") + what +
                ": pages were released from this buffer; it can only be cleared or destroyed");
        }
    }

    // Deliberately private: this is only for backend code paths that prove every
    // element is overwritten before the buffer can be observed by matrix code.
    void resize_uninitialized(uint64_t element_count) {
        assert_growable("resize_uninitialized");
        if (element_count > capacity()) {
            ensure_capacity_uninitialized(element_count);
        }
        size_ = element_count;
        refresh_page_usage();
    }

    template <typename BufferLike, typename Fn>
    static void walk_range(BufferLike& buffer, uint64_t begin, uint64_t end, Fn&& fn) {
        if (begin > end || end > buffer.size()) {
            throw std::out_of_range("PagedBuffer::for_each_range range out of bounds");
        }
        if (begin == end) {
            return;
        }

        uint64_t current = begin;
        while (current < end) {
            auto slice = buffer.trimmed_page(current, end);
            fn(slice);
            current = slice.first_element + slice.count;
        }
    }

    template <typename Lhs, typename Rhs, typename Fn>
    static void walk_zipped_range(Lhs& lhs, Rhs& rhs, uint64_t begin, uint64_t end, Fn&& fn) {
        if (lhs.elements_per_page() != rhs.elements_per_page()) {
            throw std::logic_error("PagedBuffer::for_each_zipped_range requires matching page sizes");
        }
        if (begin > end || end > lhs.size() || end > rhs.size()) {
            throw std::out_of_range("PagedBuffer::for_each_zipped_range range out of bounds");
        }
        if (begin == end) {
            return;
        }

        uint64_t current = begin;
        while (current < end) {
            auto lhs_slice = lhs.trimmed_page(current, end);
            auto rhs_slice = rhs.trimmed_page(current, end);
            const uint32_t chunk = std::min(lhs_slice.count, rhs_slice.count);
            lhs_slice.count = chunk;
            rhs_slice.count = chunk;
            fn(lhs_slice, rhs_slice);
            current += chunk;
        }
    }

    static constexpr uint32_t default_elements_per_page() {
        constexpr size_t kTargetBytes = 1u << 20;
        constexpr size_t elems = kTargetBytes / sizeof(T);
        return static_cast<uint32_t>(elems > 0 ? elems : 1);
    }

    void ensure_capacity(uint64_t element_capacity) {
        while (capacity() < element_capacity) {
            append_page();
        }
    }

    void ensure_capacity_uninitialized(uint64_t element_capacity) {
        while (capacity() < element_capacity) {
            append_page_uninitialized();
        }
    }

    // Guarantees a zeroed page, and skips the write when the allocator already
    // guaranteed it. An anonymous mmap is kernel-zeroed by definition, so the
    // fill would only serve to fault in every page of a buffer that may never
    // be fully written. The heap path still needs it: `new T[n]` DEFAULT-
    // initializes, which leaves arithmetic types indeterminate (class types
    // like std::complex do run their constructor, but this has to be right for
    // both).
    void append_page() {
        Page storage_page;
        storage_page.data = allocate_fresh_buffer<T>(elements_per_page_);
        if (storage_page.data.get_deleter().mmap_bytes == 0) {
            std::fill(storage_page.data.get(),
                      storage_page.data.get() + elements_per_page_, T(0));
        }
        pages_.push_back(std::move(storage_page));
    }

    void append_page_uninitialized() {
        Page storage_page;
        storage_page.data = allocate_fresh_buffer<T>(elements_per_page_);
        pages_.push_back(std::move(storage_page));
    }

    void refresh_page_usage() {
        uint64_t remaining = size_;
        for (auto& storage_page : pages_) {
            storage_page.used = static_cast<uint32_t>(std::min<uint64_t>(remaining, elements_per_page_));
            remaining -= storage_page.used;
        }
    }

    void zero_fill_range(uint64_t begin, uint64_t end) {
        if (begin > end || end > capacity()) {
            throw std::out_of_range("PagedBuffer::zero_fill_range range out of bounds");
        }
        uint64_t current = begin;
        while (current < end) {
            const uint32_t page_index = static_cast<uint32_t>(current / elements_per_page_);
            const uint32_t offset = static_cast<uint32_t>(current % elements_per_page_);
            const uint32_t chunk = static_cast<uint32_t>(
                std::min<uint64_t>(static_cast<uint64_t>(elements_per_page_ - offset), end - current));
            auto& storage_page = pages_.at(page_index);
            std::fill(storage_page.data.get() + offset, storage_page.data.get() + offset + chunk, T(0));
            current += chunk;
        }
    }

    // Debug-only use-after-release tripwire. An element below the watermark
    // has a valid address (MADV_DONTNEED keeps the mapping) whose contents
    // have silently reverted to zero, so nothing downstream ever announces
    // the bug -- a consuming kernel that reads behind its own release just
    // computes with zeros. The watermark is already maintained for the
    // advise-once bookkeeping; checking it costs one compare, and only in
    // builds that ask for checking.
    void assert_not_released(uint64_t index) const {
#ifndef NDEBUG
        if (index < released_through_) {
            throw std::logic_error(
                "PagedBuffer: element read below the release watermark (use-after-release)");
        }
#else
        (void)index;
#endif
    }

    std::pair<uint32_t, uint32_t> locate(uint64_t index) const {
        if (index >= size_) {
            throw std::out_of_range("PagedBuffer element index out of bounds");
        }
        assert_not_released(index);
        return {
            static_cast<uint32_t>(index / elements_per_page_),
            static_cast<uint32_t>(index % elements_per_page_)};
    }

    PageSlice<T> trimmed_page(uint64_t begin, uint64_t end) {
        assert_not_released(begin);
        auto slice = page(static_cast<uint32_t>(begin / elements_per_page_));
        const uint32_t offset = static_cast<uint32_t>(begin % elements_per_page_);
        const uint32_t available = slice.count - offset;
        const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(available, end - begin));
        slice.data += offset;
        slice.count = chunk;
        slice.first_element = begin;
        return slice; // if end is greater than the page end, this function returns the page slice end at the page end, and the caller is expected to call it again with the next page until the full range is covered
    }

    PageSlice<const T> trimmed_page(uint64_t begin, uint64_t end) const {
        assert_not_released(begin);
        auto slice = page(static_cast<uint32_t>(begin / elements_per_page_));
        const uint32_t offset = static_cast<uint32_t>(begin % elements_per_page_);
        const uint32_t available = slice.count - offset;
        const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(available, end - begin));
        slice.data += offset;
        slice.count = chunk;
        slice.first_element = begin;
        return slice;
    }

    Page& require_page(uint32_t page_index) {
        if (page_index >= pages_.size()) {
            throw std::out_of_range("PagedBuffer::page page index out of bounds");
        }
        return pages_[page_index];
    }

    const Page& require_page(uint32_t page_index) const {
        if (page_index >= pages_.size()) {
            throw std::out_of_range("PagedBuffer::page page index out of bounds");
        }
        return pages_[page_index];
    }

    uint64_t size_ = 0;
    uint32_t elements_per_page_ = 1;
    // Set once any page has been handed back: capacity() still counts the slot,
    // so growing afterwards would write into a range whose contents were
    // silently discarded. The mapping survives -- MADV_DONTNEED, not munmap --
    // which is exactly why the guard is needed: nothing faults to announce it.
    bool released_prefix_ = false;
    // Elements already handed back, so a repeated call advises only what is
    // newly dead rather than the whole prefix again.
    uint64_t released_through_ = 0;
    std::vector<Page> pages_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_STORAGE_PAGED_ARRAY_HPP
