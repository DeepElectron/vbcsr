#ifndef VBCSR_DETAIL_STORAGE_NUMA_BUFFER_HPP
#define VBCSR_DETAIL_STORAGE_NUMA_BUFFER_HPP

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vbcsr {
namespace detail {

// Fresh-buffer allocation shared by NumaVector and PagedBuffer.
//
// Heap-recycled memory keeps whatever NUMA placement its previous owner's
// writes gave it, which silently defeats first-touch placement (caught by
// test_numa_locality). Anonymous mmap pages are guaranteed untouched, so the
// thread that zero-fills a range is always the first toucher. Non-trivial T
// (std::complex: array-new zero-writes on the allocating thread) and
// non-Linux keep plain new[]; buffers below kFreshMmapMinBytes stay on the
// heap (cache-resident anyway; the mmap round-trip would not pay off).
inline constexpr size_t kFreshMmapMinBytes = 256u * 1024u;

template <typename T>
struct FreshBufferDeleter {
    size_t mmap_bytes = 0;  // 0 => delete[]
    void operator()(T* ptr) const {
        if (ptr == nullptr) {
            return;
        }
#ifdef __linux__
        if (mmap_bytes != 0) {
            munmap(static_cast<void*>(ptr), mmap_bytes);
            return;
        }
#endif
        delete[] ptr;
    }
};

template <typename T>
using FreshBufferOwner = std::unique_ptr<T[], FreshBufferDeleter<T>>;

template <typename T>
inline FreshBufferOwner<T> allocate_fresh_buffer(size_t count) {
    if (count == 0) {
        return FreshBufferOwner<T>(nullptr, FreshBufferDeleter<T>{});
    }
#ifdef __linux__
    if constexpr (std::is_trivially_default_constructible_v<T> &&
                  std::is_trivially_destructible_v<T>) {
        const size_t bytes = count * sizeof(T);
        if (bytes >= kFreshMmapMinBytes) {
            void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mem != MAP_FAILED) {
                return FreshBufferOwner<T>(static_cast<T*>(mem),
                                           FreshBufferDeleter<T>{bytes});
            }
        }
    }
#endif
    return FreshBufferOwner<T>(new T[count], FreshBufferDeleter<T>{});
}

// Contiguous vector-like buffer whose pages spread across NUMA nodes.
//
// std::vector value-initializes on resize, so the ALLOCATING thread touches
// every page and the whole buffer lands on its node — on a dual-socket host
// that caps multi-threaded streaming at one socket's bandwidth (the same
// defect doc/numa_locality_plan.md fixes for matrix storage). NumaVector
// instead:
//   1. allocates FRESH memory (mmap for trivially-constructible scalars —
//      heap-recycled memory keeps its previous placement),
//   2. zero-fills (and prefix-copies on growth) inside a parallel region,
//      each thread touching an even slice, so pages distribute across the
//      executing team's nodes.
// The even slice approximates the applies' nnz-weighted row split; per-domain
// exactness is not required because the apply's own in-region writes never
// move already-placed pages. Semantics match std::vector<T>: elements are
// zeroed, resize preserves the prefix.
template <typename T>
class NumaVector {
    using Owner = FreshBufferOwner<T>;

    static Owner allocate(size_t count) { return allocate_fresh_buffer<T>(count); }

    Owner buffer_;
    size_t size_ = 0;

    // First touch: each thread writes its even slice — prefix copied from
    // `old_data`, tail zeroed. Runs serially inside an enclosing parallel
    // region (team of one), which is the correct degenerate behavior.
    static void parallel_fill(T* dst, const T* old_data, size_t old_count, size_t count) {
        #pragma omp parallel
        {
#ifdef _OPENMP
            const size_t threads = static_cast<size_t>(omp_get_num_threads());
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
#else
            const size_t threads = 1;
            const size_t tid = 0;
#endif
            const size_t begin = count * tid / threads;
            const size_t end = count * (tid + 1) / threads;
            const size_t copy_end = std::min(end, old_count);
            if (begin < copy_end) {
                std::copy(old_data + begin, old_data + copy_end, dst + begin);
            }
            const size_t zero_begin = std::max(begin, copy_end);
            if (zero_begin < end) {
                std::fill(dst + zero_begin, dst + end, T(0));
            }
        }
    }

public:
    using value_type = T;

    NumaVector() = default;
    explicit NumaVector(size_t count) { resize(count); }

    NumaVector(const NumaVector& other) { assign_from(other); }
    NumaVector& operator=(const NumaVector& other) {
        if (this != &other) {
            assign_from(other);
        }
        return *this;
    }
    NumaVector(NumaVector&&) noexcept = default;
    NumaVector& operator=(NumaVector&&) noexcept = default;

    // std::vector semantics: prefix preserved, new elements zeroed.
    void resize(size_t count) {
        if (count == size_) {
            return;
        }
        if (count == 0) {
            buffer_.reset();
            size_ = 0;
            return;
        }
        Owner fresh = allocate(count);
        parallel_fill(fresh.get(), buffer_.get(), std::min(size_, count), count);
        buffer_ = std::move(fresh);
        size_ = count;
    }

    void clear() {
        buffer_.reset();
        size_ = 0;
    }

    // Bulk assign: fresh buffer, parallel first-touch copy of [src, src+count).
    void assign(const T* src, size_t count) {
        if (count == 0) {
            clear();
            return;
        }
        Owner fresh = allocate(count);
        parallel_fill(fresh.get(), src, count, count);
        buffer_ = std::move(fresh);
        size_ = count;
    }

    friend bool operator==(const NumaVector& a, const NumaVector& b) {
        return a.size_ == b.size_ && std::equal(a.begin(), a.end(), b.begin());
    }
    friend bool operator!=(const NumaVector& a, const NumaVector& b) {
        return !(a == b);
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    T* data() { return buffer_.get(); }
    const T* data() const { return buffer_.get(); }
    T* begin() { return buffer_.get(); }
    T* end() { return buffer_.get() + size_; }
    const T* begin() const { return buffer_.get(); }
    const T* end() const { return buffer_.get() + size_; }

    T& operator[](size_t index) { return buffer_[index]; }
    const T& operator[](size_t index) const { return buffer_[index]; }

private:
    void assign_from(const NumaVector& other) {
        Owner fresh = allocate(other.size_);
        parallel_fill(fresh.get(), other.buffer_.get(), other.size_, other.size_);
        buffer_ = std::move(fresh);
        size_ = other.size_;
    }
};

template <typename T>
inline bool operator==(const NumaVector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template <typename T>
inline bool operator==(const std::vector<T>& a, const NumaVector<T>& b) {
    return b == a;
}

template <typename T>
inline bool operator!=(const NumaVector<T>& a, const std::vector<T>& b) {
    return !(a == b);
}

template <typename T>
inline bool operator!=(const std::vector<T>& a, const NumaVector<T>& b) {
    return !(a == b);
}

// Read-only view over a contiguous int array. Backends and kernels only read
// column/adjacency indices, so their signatures take this span and accept
// both std::vector<int> (staging, tests) and NumaVector<int> (graph storage)
// without conversion or copies.
class IndexSpan {
public:
    IndexSpan() = default;
    IndexSpan(const std::vector<int>& v) : ptr_(v.data()), count_(v.size()) {}
    IndexSpan(const NumaVector<int>& v) : ptr_(v.data()), count_(v.size()) {}

    const int* data() const { return ptr_; }
    size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    const int* begin() const { return ptr_; }
    const int* end() const { return ptr_ + count_; }
    int operator[](size_t index) const { return ptr_[index]; }

private:
    const int* ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace detail
} // namespace vbcsr

#endif // VBCSR_DETAIL_STORAGE_NUMA_BUFFER_HPP
