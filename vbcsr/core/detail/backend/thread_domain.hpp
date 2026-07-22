#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vbcsr {
namespace detail {

inline int thread_domain_max_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

// A work-balanced split of a rank's owned block rows into contiguous
// per-thread domains. This is the thread-level analogue of the MPI row
// partition: computed once from the structure, stored, and then used both by
// the apply kernels (each thread iterates its own domain) and by storage
// first-touch (each thread zero-touches its own domain's pages), so memory
// placement and access cannot disagree on a NUMA host.
//
// A partition is only honored when the executing parallel region has exactly
// `thread_count` threads (see `matches`); otherwise callers fall back to
// their dynamic split. Correctness never depends on the partition — only
// locality does.
struct ThreadDomainPartition {
    int thread_count = 0;
    // Row boundaries, size thread_count + 1: domain t owns rows
    // [row_bounds[t], row_bounds[t + 1]).
    std::vector<int> row_bounds;

    bool empty() const { return row_bounds.empty(); }

    bool matches(int threads) const {
        return thread_count == threads &&
            static_cast<int>(row_bounds.size()) == thread_count + 1;
    }

    int domain_begin(int tid) const { return row_bounds[static_cast<size_t>(tid)]; }
    int domain_end(int tid) const { return row_bounds[static_cast<size_t>(tid) + 1]; }
};

// Split n_rows into `thread_count` contiguous domains with (approximately)
// equal cumulative weight. `weight(row)` is any monotone per-row cost proxy —
// nnz for CSR/BSR, scalar work units for VBCSR. Boundary t is placed at the
// first row whose cumulative weight reaches t/thread_count of the total, so
// every domain is non-degenerate whenever n_rows >= thread_count and no
// single row dominates.
template <typename WeightFn>
inline ThreadDomainPartition build_thread_domain_partition(
    int n_rows,
    int thread_count,
    WeightFn&& weight) {
    ThreadDomainPartition part;
    part.thread_count = std::max(1, thread_count);
    part.row_bounds.assign(static_cast<size_t>(part.thread_count) + 1, 0);
    if (n_rows <= 0) {
        return part;
    }

    uint64_t total = 0;
    for (int row = 0; row < n_rows; ++row) {
        total += static_cast<uint64_t>(weight(row));
    }

    if (total == 0) {
        // Structureless rows: fall back to an even row split.
        for (int t = 0; t <= part.thread_count; ++t) {
            part.row_bounds[static_cast<size_t>(t)] =
                static_cast<int>((static_cast<int64_t>(n_rows) * t) / part.thread_count);
        }
        return part;
    }

    uint64_t cumulative = 0;
    int next_domain = 1;
    for (int row = 0; row < n_rows && next_domain < part.thread_count; ++row) {
        cumulative += static_cast<uint64_t>(weight(row));
        // Close every domain whose weight quantile this row crossed.
        while (next_domain < part.thread_count &&
               cumulative * static_cast<uint64_t>(part.thread_count) >=
                   total * static_cast<uint64_t>(next_domain)) {
            part.row_bounds[static_cast<size_t>(next_domain)] = row + 1;
            ++next_domain;
        }
    }
    for (int t = next_domain; t <= part.thread_count; ++t) {
        part.row_bounds[static_cast<size_t>(t)] = n_rows;
    }
    return part;
}

} // namespace detail
} // namespace vbcsr
