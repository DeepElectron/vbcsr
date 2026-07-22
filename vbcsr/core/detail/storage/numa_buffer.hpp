#ifndef VBCSR_DETAIL_STORAGE_NUMA_BUFFER_HPP
#define VBCSR_DETAIL_STORAGE_NUMA_BUFFER_HPP

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>

#ifdef __linux__
#include <sys/mman.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vbcsr {
namespace detail {

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
    static constexpr bool kFreshMmap =
        std::is_trivially_default_constructible_v<T> &&
        std::is_trivially_destructible_v<T>;
    // Below this the mmap round-trip outweighs placement (small buffers are
    // cache-resident anyway).
    static constexpr size_t kMinMmapBytes = 256u * 1024u;

    struct Deleter {
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
    using Owner = std::unique_ptr<T[], Deleter>;

    static Owner allocate(size_t count) {
        if (count == 0) {
            return Owner(nullptr, Deleter{});
        }
#ifdef __linux__
        if constexpr (kFreshMmap) {
            const size_t bytes = count * sizeof(T);
            if (bytes >= kMinMmapBytes) {
                void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (mem != MAP_FAILED) {
                    return Owner(static_cast<T*>(mem), Deleter{bytes});
                }
            }
        }
#endif
        return Owner(new T[count], Deleter{});
    }

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

} // namespace detail
} // namespace vbcsr

#endif // VBCSR_DETAIL_STORAGE_NUMA_BUFFER_HPP
