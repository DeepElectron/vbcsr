#ifndef VBCSR_DETAIL_PAGED_ARRAY_HPP
#define VBCSR_DETAIL_PAGED_ARRAY_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace vbcsr::detail {

template <typename T>
struct PageSlice {
    T* data = nullptr;
    uint32_t count = 0;
    uint32_t page_index = 0;
    uint64_t first_element = 0;
};

template <typename T>
class PagedBuffer {
public:
    struct Page {
        std::unique_ptr<T[]> data;
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
    }

    void reserve(uint64_t element_capacity) {
        ensure_capacity(element_capacity);
    }

    void resize(uint64_t element_count) {
        if (element_count > capacity()) {
            ensure_capacity(element_count);
        }
        if (element_count > size_) {
            zero_fill_range(size_, element_count);
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

    void append_page() {
        Page storage_page;
        storage_page.data = std::make_unique<T[]>(elements_per_page_);
        std::fill(storage_page.data.get(), storage_page.data.get() + elements_per_page_, T(0));
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

    std::pair<uint32_t, uint32_t> locate(uint64_t index) const {
        if (index >= size_) {
            throw std::out_of_range("PagedBuffer element index out of bounds");
        }
        return {
            static_cast<uint32_t>(index / elements_per_page_),
            static_cast<uint32_t>(index % elements_per_page_)};
    }

    PageSlice<T> trimmed_page(uint64_t begin, uint64_t end) {
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
    std::vector<Page> pages_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_PAGED_ARRAY_HPP
