// Locality invariant test (doc/numa_locality_plan.md stage D).
//
// The first-touch construction path places each thread domain's value pages
// on the NUMA node of the owning thread. This test makes that a CI-enforced
// invariant instead of something that silently rots the next time allocation
// changes — which is exactly how the original defect arose: for CSR, BSR and
// VBCSR it builds a matrix large enough for several MiB per thread domain,
// then each thread samples value-block addresses from the interior of its own
// domain and asserts they reside on its node (get_mempolicy MPOL_F_ADDR).
//
// The invariant is only meaningful when the OS, OpenMP runtime, and topology
// cooperate, so the test SKIPs (passes with a notice) rather than fails when:
// not Linux, OpenMP disabled, a single thread, threads not spanning >1 NUMA
// node (single-node host or packed binding), or the syscalls are unavailable.
// Run it with OMP_PROC_BIND=spread OMP_PLACES=cores so a multi-node host
// actually exercises it (the ctest registration sets this).

// Test assertions must stay active in Release builds.
#undef NDEBUG

#include "../block_csr.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#ifndef MPOL_F_NODE
#define MPOL_F_NODE (1 << 0)
#endif
#ifndef MPOL_F_ADDR
#define MPOL_F_ADDR (1 << 1)
#endif

namespace {

int node_of_address(const void* addr) {
    static const uintptr_t page_mask =
        ~static_cast<uintptr_t>(sysconf(_SC_PAGESIZE) - 1);
    int node = -1;
    const void* base =
        reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(addr) & page_mask);
    if (syscall(SYS_get_mempolicy, &node, nullptr, 0, base,
                MPOL_F_NODE | MPOL_F_ADDR) != 0) {
        return -1;
    }
    return node;
}

int current_node() {
    unsigned cpu = 0;
    unsigned node = 0;
    if (syscall(SYS_getcpu, &cpu, &node, nullptr) != 0) {
        return -1;
    }
    return static_cast<int>(node);
}

} // namespace
#endif // __linux__

using namespace vbcsr;

namespace {

// Deterministic degree-13-ish adjacency: diagonal plus pseudo-random
// neighbors. Structure only — placement happens at construction, no assembly
// needed.
std::vector<std::vector<int>> make_adjacency(int n_rows) {
    std::vector<std::vector<int>> adjacency(static_cast<size_t>(n_rows));
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int row = 0; row < n_rows; ++row) {
        auto& cols = adjacency[static_cast<size_t>(row)];
        cols.push_back(row);
        for (int k = 0; k < 12; ++k) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            cols.push_back(static_cast<int>((state >> 33) % static_cast<uint64_t>(n_rows)));
        }
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    }
    return adjacency;
}

#if defined(__linux__) && defined(_OPENMP)

// Returns true when the invariant was actually exercised (threads spanned
// more than one node and the samples resolved).
bool check_locality(const char* label, BlockSpMat<double>& mat) {
    const auto& domains = mat.thread_domain_partition();
    const auto& adj_ptr = mat.graph->adj_ptr;
    const int max_threads = omp_get_max_threads();
    const int check_rows = static_cast<int>(adj_ptr.size()) - 1;
    assert(domains.matches(max_threads, check_rows) && "construction must store a matching partition");
    const int thread_count = domains.thread_count;
    std::vector<int> thread_nodes(static_cast<size_t>(thread_count), -1);
    std::vector<int> matched(static_cast<size_t>(thread_count), 0);
    std::vector<int> sampled(static_cast<size_t>(thread_count), 0);

    #pragma omp parallel num_threads(thread_count)
    {
        const int tid = omp_get_thread_num();
        thread_nodes[static_cast<size_t>(tid)] = current_node();

        // Sample the middle half of the domain: boundary OS pages (and huge
        // pages up to ~2 MiB around each boundary) may legitimately belong to
        // a neighbor domain.
        const int row_begin = domains.domain_begin(tid);
        const int row_end = domains.domain_end(tid);
        const int span = row_end - row_begin;
        const int inner_begin = row_begin + span / 4;
        const int inner_end = row_end - span / 4;
        const int samples = 64;
        for (int s = 0; s < samples && inner_begin < inner_end; ++s) {
            const int row = inner_begin +
                static_cast<int>((static_cast<int64_t>(s) * (inner_end - inner_begin)) / samples);
            const int slot = adj_ptr[row];
            if (slot >= adj_ptr[row + 1]) {
                continue;  // structurally empty row
            }
            const int node = node_of_address(mat.block_data(slot));
            const int own = thread_nodes[static_cast<size_t>(tid)];
            if (node < 0 || own < 0) {
                continue;  // syscall unavailable; counted as unsampled
            }
            ++sampled[static_cast<size_t>(tid)];
            if (node == own) {
                ++matched[static_cast<size_t>(tid)];
            }
        }
    }

    const std::set<int> distinct_nodes(thread_nodes.begin(), thread_nodes.end());
    const bool spans_nodes =
        distinct_nodes.size() > 1 && distinct_nodes.count(-1) == 0;
    long total_sampled = 0;
    long total_matched = 0;
    for (int t = 0; t < thread_count; ++t) {
        total_sampled += sampled[static_cast<size_t>(t)];
        total_matched += matched[static_cast<size_t>(t)];
    }

    if (!spans_nodes || total_sampled == 0) {
        std::printf("%s: SKIP (threads span %zu node(s), %ld samples)\n",
                    label, distinct_nodes.size(), total_sampled);
        return false;
    }

    const double match_fraction =
        static_cast<double>(total_matched) / static_cast<double>(total_sampled);
    std::printf("%s: %ld/%ld interior samples node-local (%.1f%%)\n",
                label, total_matched, total_sampled, 100.0 * match_fraction);
    if (match_fraction < 0.9) {
        for (int t = 0; t < thread_count; ++t) {
            const int row = domains.domain_begin(t);
            const int probe_row = (domains.domain_begin(t) + domains.domain_end(t)) / 2;
            std::printf("  thread %d: node=%d rows=[%d,%d) matched %d/%d, mid-row page node=%d\n",
                        t, thread_nodes[static_cast<size_t>(t)],
                        row, domains.domain_end(t),
                        matched[static_cast<size_t>(t)], sampled[static_cast<size_t>(t)],
                        node_of_address(mat.block_data(adj_ptr[probe_row])));
        }
    }
    std::fflush(stdout);
    // 90%: tolerate stray pages (THP coalescing, OS balancing), catch the
    // defect mode (everything on node 0 => remote domains match ~0%).
    assert(match_fraction >= 0.9 && "value pages are not domain-local");
    return true;
}

#endif // __linux__ && _OPENMP

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int exercised = 0;

#if !defined(__linux__)
    std::printf("SKIP: locality invariant requires Linux (get_mempolicy)\n");
#elif !defined(_OPENMP)
    std::printf("SKIP: locality invariant requires OpenMP\n");
#else
    if (omp_get_max_threads() < 2) {
        std::printf("SKIP: needs >= 2 threads (run with OMP_NUM_THREADS>=2)\n");
    } else {
        {
            // CSR: block size 1. ~13 nnz/row * 8 B: 400k rows ~= 40 MiB.
            const int n_rows = 400000;
            std::vector<int> block_sizes(n_rows, 1);
            DistGraph graph(MPI_COMM_SELF);
            graph.construct_serial(n_rows, block_sizes, make_adjacency(n_rows));
            BlockSpMat<double> mat(&graph);
            assert(mat.matrix_kind() == MatrixKind::CSR);
            exercised += check_locality("csr", mat);
        }
        {
            // BSR: 8x8 blocks. ~13 blocks/row * 512 B: 8k rows ~= 53 MiB.
            const int n_rows = 8000;
            std::vector<int> block_sizes(n_rows, 8);
            DistGraph graph(MPI_COMM_SELF);
            graph.construct_serial(n_rows, block_sizes, make_adjacency(n_rows));
            BlockSpMat<double> mat(&graph);
            assert(mat.matrix_kind() == MatrixKind::BSR);
            exercised += check_locality("bsr", mat);
        }
        {
            // VBCSR: mixed 4/8/12 blocks, mean block ~72 values: 10k rows ~= 75 MiB.
            const int n_rows = 10000;
            std::vector<int> block_sizes(n_rows);
            for (int row = 0; row < n_rows; ++row) {
                block_sizes[static_cast<size_t>(row)] = 4 + 4 * (row % 3);
            }
            DistGraph graph(MPI_COMM_SELF);
            graph.construct_serial(n_rows, block_sizes, make_adjacency(n_rows));
            BlockSpMat<double> mat(&graph);
            assert(mat.matrix_kind() == MatrixKind::VBCSR);
            exercised += check_locality("vbcsr", mat);
        }
    }
#endif

    std::printf(exercised > 0 ? "PASSED (invariant exercised on %d kind(s))\n"
                              : "PASSED (skipped: environment cannot exercise the invariant; %d)\n",
                exercised);
    MPI_Finalize();
    return 0;
}
