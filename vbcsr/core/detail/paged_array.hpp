#ifndef VBCSR_DETAIL_PAGED_ARRAY_HPP
#define VBCSR_DETAIL_PAGED_ARRAY_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace vbcsr::detail {

template <typename T>
struct PageSpan {
    T* data = nullptr;
    uint32_t length = 0;
    uint32_t page_id = 0;
    uint64_t global_begin = 0;
};

template <typename T>
class PagedArray {
public:
    struct Page {
        std::unique_ptr<T[]> data;
        uint32_t used = 0;
    };

    PagedArray()
        : page_capacity_(default_page_capacity()) {}

    explicit PagedArray(uint32_t page_capacity)
        : page_capacity_(std::max<uint32_t>(1, page_capacity)) {}

    uint64_t size() const {
        return logical_size_;
    }

    bool empty() const {
        return logical_size_ == 0;
    }

    uint32_t page_count() const {
        return static_cast<uint32_t>(pages_.size());
    }

    uint32_t page_capacity() const {
        return page_capacity_;
    }

    void clear() {
        pages_.clear();
        logical_size_ = 0;
    }

    void resize(uint64_t logical_size) {
        clear();
        logical_size_ = logical_size;
        if (logical_size_ == 0) {
            return;
        }

        uint64_t remaining = logical_size_;
        while (remaining > 0) {
            const uint32_t used = static_cast<uint32_t>(
                std::min<uint64_t>(remaining, page_capacity_));
            append_page(used);
            remaining -= used;
        }
    }

    T& operator[](uint64_t index) {
        return *ptr(index);
    }

    const T& operator[](uint64_t index) const {
        return *ptr(index);
    }

    T* ptr(uint64_t index) {
        auto [page_id, offset] = locate(index);
        return pages_[page_id].data.get() + offset;
    }

    const T* ptr(uint64_t index) const {
        auto [page_id, offset] = locate(index);
        return pages_[page_id].data.get() + offset;
    }

    PageSpan<T> page_span(uint32_t page_id) {
        const auto& page = require_page(page_id);
        return PageSpan<T>{
            page.data.get(),
            page.used,
            page_id,
            static_cast<uint64_t>(page_id) * static_cast<uint64_t>(page_capacity_)};
    }

    PageSpan<const T> page_span(uint32_t page_id) const {
        const auto& page = require_page(page_id);
        return PageSpan<const T>{
            page.data.get(),
            page.used,
            page_id,
            static_cast<uint64_t>(page_id) * static_cast<uint64_t>(page_capacity_)};
    }

    template <typename Fn>
    void for_each_span(uint64_t begin, uint64_t end, Fn&& fn) {
        for_each_span_impl(*this, begin, end, std::forward<Fn>(fn));
    }

    template <typename Fn>
    void for_each_span(uint64_t begin, uint64_t end, Fn&& fn) const {
        for_each_span_impl(*this, begin, end, std::forward<Fn>(fn));
    }

    template <typename U, typename Fn>
    void for_each_zip_span(PagedArray<U>& other, uint64_t begin, uint64_t end, Fn&& fn) {
        for_each_zip_span_impl(*this, other, begin, end, std::forward<Fn>(fn));
    }

    template <typename U, typename Fn>
    void for_each_zip_span(const PagedArray<U>& other, uint64_t begin, uint64_t end, Fn&& fn) const {
        for_each_zip_span_impl(*this, other, begin, end, std::forward<Fn>(fn));
    }

private:
    template <typename ArrayLike, typename Fn>
    static void for_each_span_impl(ArrayLike& array, uint64_t begin, uint64_t end, Fn&& fn) {
        if (begin > end || end > array.size()) {
            throw std::out_of_range("PagedArray::for_each_span range out of bounds");
        }
        if (begin == end) {
            return;
        }

        uint64_t current = begin;
        while (current < end) {
            const uint32_t page_id = static_cast<uint32_t>(current / array.page_capacity_);
            const uint32_t offset = static_cast<uint32_t>(current % array.page_capacity_);
            auto span = array.page_span(page_id);
            const uint32_t available = span.length - offset;
            const uint32_t chunk = static_cast<uint32_t>(
                std::min<uint64_t>(available, end - current));
            span.data += offset;
            span.length = chunk;
            span.global_begin = current;
            fn(span);
            current += chunk;
        }
    }

    template <typename ArrayA, typename ArrayB, typename Fn>
    static void for_each_zip_span_impl(ArrayA& lhs, ArrayB& rhs, uint64_t begin, uint64_t end, Fn&& fn) {
        if (lhs.page_capacity() != rhs.page_capacity()) {
            throw std::logic_error("PagedArray::for_each_zip_span requires matching page capacities");
        }
        if (begin > end || end > lhs.size() || end > rhs.size()) {
            throw std::out_of_range("PagedArray::for_each_zip_span range out of bounds");
        }
        if (begin == end) {
            return;
        }

        uint64_t current = begin;
        while (current < end) {
            const uint32_t page_id = static_cast<uint32_t>(current / lhs.page_capacity());
            const uint32_t offset = static_cast<uint32_t>(current % lhs.page_capacity());
            auto lhs_span = lhs.page_span(page_id);
            auto rhs_span = rhs.page_span(page_id);
            const uint32_t available = std::min(lhs_span.length, rhs_span.length) - offset;
            const uint32_t chunk = static_cast<uint32_t>(
                std::min<uint64_t>(available, end - current));
            lhs_span.data += offset;
            lhs_span.length = chunk;
            lhs_span.global_begin = current;
            rhs_span.data += offset;
            rhs_span.length = chunk;
            rhs_span.global_begin = current;
            fn(lhs_span, rhs_span);
            current += chunk;
        }
    }

    static constexpr uint32_t default_page_capacity() {
        constexpr size_t kTargetBytes = 1u << 20;
        constexpr size_t elems = kTargetBytes / sizeof(T);
        return static_cast<uint32_t>(elems > 0 ? elems : 1);
    }

    const Page& require_page(uint32_t page_id) const {
        if (page_id >= pages_.size()) {
            throw std::out_of_range("PagedArray::page_span page_id out of bounds");
        }
        return pages_[page_id];
    }

    std::pair<uint32_t, uint32_t> locate(uint64_t index) const {
        if (index >= logical_size_) {
            throw std::out_of_range("PagedArray index out of bounds");
        }
        const uint32_t page_id = static_cast<uint32_t>(index / page_capacity_);
        const uint32_t offset = static_cast<uint32_t>(index % page_capacity_);
        return {page_id, offset};
    }

    void append_page(uint32_t used) {
        Page page;
        page.used = used;
        page.data = std::make_unique<T[]>(page_capacity_);
        std::fill(page.data.get(), page.data.get() + page_capacity_, T(0));
        pages_.push_back(std::move(page));
    }

    uint64_t logical_size_ = 0;
    uint32_t page_capacity_ = 1;
    std::vector<Page> pages_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_PAGED_ARRAY_HPP
