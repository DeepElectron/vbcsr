#ifndef VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_TYPES_HPP
#define VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_TYPES_HPP

#include <cstdlib>
#include <map>
#include <unistd.h>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace vbcsr {

struct BlockID {
    int row;
    int col;

    bool operator<(const BlockID& other) const {
        if (row != other.row) {
            return row < other.row;
        }
        return col < other.col;
    }
};

namespace detail {

/// Hand a staged row's physical pages back to the OS, then drop the vector.
///
/// `std::vector<T>().swap(v)` releases the capacity to the ALLOCATOR, which is
/// not the same as releasing it to the kernel, and on the staged rows of a
/// fused product the difference is most of the peak. glibc returns a large
/// block by munmap only while it is above the mmap threshold, and that
/// threshold is DYNAMIC: freeing one mmapped chunk raises it to that chunk's
/// size, so the first staged row returned is also the last one returned --
/// every row after it is served from the arena and its free() is bookkeeping.
/// Measured on a 4096-block product with 200 neighbours per row: the copy pass
/// wrote a 5.72 GB result while freeing 5.72 GB of staging and RSS still rose
/// 3.9 GB.
///
/// MADV_DONTNEED is the same instrument PagedBuffer::release_pages_before uses,
/// for the same reason, and it is a statement to the kernel rather than to
/// malloc: the mapping and the pointer stay valid, the physical pages go, and a
/// later touch faults in zeroes. That is exactly the contract free() needs --
/// it writes its own metadata into the chunk afterwards (a page or two, faulted
/// straight back) and never reads what the caller left there.
///
/// Interior pages only: the partial pages at either end may be shared with a
/// neighbouring live allocation, and MADV_DONTNEED would discard that too.
template <typename T>
inline void release_and_drop(std::vector<T>& v) {
#ifdef __linux__
    if (!v.empty()) {
        static const long os_page = ::sysconf(_SC_PAGESIZE);
        if (os_page > 0) {
            const uintptr_t mask = static_cast<uintptr_t>(os_page) - 1;
            const uintptr_t begin = reinterpret_cast<uintptr_t>(v.data());
            const uintptr_t end = begin + v.size() * sizeof(T);
            const uintptr_t lo = (begin + mask) & ~mask;
            const uintptr_t hi = end & ~mask;
            if (hi > lo) {
                ::madvise(reinterpret_cast<void*>(lo), static_cast<size_t>(hi - lo),
                          MADV_DONTNEED);
            }
        }
    }
#endif
    std::vector<T>().swap(v);
}


// An owning block record: block_csr's assembly staging builds these to SEND.
template <typename T>
struct FetchedBlock {
    int global_row;
    int global_col;
    int r_dim;
    int c_dim;
    std::vector<T> data;
};

// A fetched block as the payload exchange RETURNS it: a view, not an owner.
// Remote payloads live in the context's arena below; blocks the rank already
// owned point straight into the source matrix's storage (no copy at all), so
// the SOURCE MATRIX MUST OUTLIVE THE CONTEXT -- true of every caller, which
// holds the operand it is fetching from for the duration of its product.
template <typename T>
struct FetchedBlockRef {
    int global_row;
    int global_col;
    int r_dim;
    int c_dim;
    const T* data;
};

template <typename T>
struct FetchedBlockContext {
    std::vector<FetchedBlockRef<T>> blocks;
    std::map<int, int> row_sizes;
    // One allocation owning every REMOTE payload the refs point into. This
    // replaces one heap vector per fetched block -- at halo scale, tens of
    // millions of allocations and a serial copy loop -- with a single sized
    // arena filled by a parallel copy.
    std::vector<T> arena;
};

} // namespace detail

} // namespace vbcsr

#endif // VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_TYPES_HPP
