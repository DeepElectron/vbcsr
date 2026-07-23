#ifndef VBCSR_DIST_MULTIVECTOR_HPP
#define VBCSR_DIST_MULTIVECTOR_HPP

#include "dist_graph.hpp"
#include "dist_vector.hpp"
#include "detail/distributed/mpi_utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <random>
#include <stdexcept>

#include <complex>
#include "scalar_traits.hpp"

namespace vbcsr {

// Layout contract (doc/row_major_migration_plan.md §2.1):
// ROW-major storage with a padded leading dimension. Element (row, vec) lives
// at data[row * ld + vec], rows are [owned | ghost], and
// ld = round_up(num_vectors * sizeof(T), 64) / sizeof(T) so every row starts
// a new cache line (double: multiple of 8; complex<double>: multiple of 4).
//
// PADDING INVARIANT: lanes [num_vectors, ld) of every row are always ZERO.
// The flat element-wise operations (scale, axpy, axpby, pointwise_mult,
// copy_from, bdot's accumulation) iterate the whole padded buffer and stay
// correct because the padding contributes exact zeros. Any code that writes
// the buffer must preserve the invariant (see zero_padding()).
template <typename T>
class DistMultiVector {
public:
    using value_type = T;
    DistGraph* graph;
    int num_vectors;
    // Padded leading dimension in elements (>= num_vectors).
    int ld;
    // Row-major: (local_rows + ghost_rows) x ld. NumaVector, not std::vector:
    // at rhs=16 this buffer rivals the matrix in bytes streamed per apply, so
    // its pages must spread across NUMA nodes too (numa_buffer.hpp).
    detail::NumaVector<T> data;
    int local_rows;
    int ghost_rows;

    static int compute_ld(int n_vecs) {
        const int bytes = static_cast<int>(sizeof(T));
        const int per_line = 64 / bytes;
        return ((n_vecs + per_line - 1) / per_line) * per_line;
    }

    DistMultiVector(DistGraph* g, int n_vecs)
        : graph(g), num_vectors(n_vecs), ld(compute_ld(n_vecs)) {
        graph->get_vector_structure(local_rows, ghost_rows);
        // Value-initialized: padding lanes start zero.
        data.resize(static_cast<size_t>(local_rows + ghost_rows) * ld);
    }

    // Accessors: (row, vec) -> data[row * ld + vec]
    T& operator()(int row, int col) {
        return data[static_cast<size_t>(row) * ld + col];
    }

    const T& operator()(int row, int col) const {
        return data[static_cast<size_t>(row) * ld + col];
    }

    // Contiguous row (all vectors of one scalar row, plus padding lanes).
    T* row_data(int row) {
        return data.data() + static_cast<size_t>(row) * ld;
    }

    const T* row_data(int row) const {
        return data.data() + static_cast<size_t>(row) * ld;
    }

    // Re-establish the padding invariant after a bulk write of the buffer.
    void zero_padding() {
        if (ld == num_vectors) return;
        const int total_rows = local_rows + ghost_rows;
        for (int row = 0; row < total_rows; ++row) {
            T* r = row_data(row);
            std::fill(r + num_vectors, r + ld, T(0));
        }
    }

    void conjugate() {
        for (int row = 0; row < local_rows; ++row) {
            T* r = row_data(row);
            for (int v = 0; v < num_vectors; ++v) {
                r[v] = ScalarTraits<T>::conjugate(r[v]);
            }
        }
    }

    // Bind to a new graph (must have same owned structure).
    // Same contract as DistVector::bind_to_graph: any owned-structure
    // mismatch (partition or block sizes) is a hard error, even when the
    // scalar sizes coincide.
    void bind_to_graph(DistGraph* new_graph) {
        if (graph == new_graph) return;

        if (!graph->same_owned_structure(*new_graph)) {
            throw std::runtime_error(
                "DistMultiVector::bind_to_graph: graph mismatch - target graph has "
                "a different owned block structure (partition or block sizes)");
        }

        int new_local_rows, new_ghost_rows;
        new_graph->get_vector_structure(new_local_rows, new_ghost_rows);

        // Row-major makes this trivial: owned rows are the buffer prefix, so
        // only the ghost tail is resized. Grown rows are value-initialized
        // (zeros, preserving the padding invariant); ghost values are
        // invalid until the next sync either way.
        data.resize(static_cast<size_t>(new_local_rows + new_ghost_rows) * ld);
        graph = new_graph;
        ghost_rows = new_ghost_rows;
    }

    // Operations
    void set_constant(T val) {
        const int total_rows = local_rows + ghost_rows;
        for (int row = 0; row < total_rows; ++row) {
            T* r = row_data(row);
            std::fill(r, r + num_vectors, val);
        }
        // Padding stays zero by construction (never written here).
    }

    void scale(T alpha) {
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) data[i] *= alpha;
    }

    void axpy(T alpha, const DistMultiVector<T>& x) {
        assert(x.num_vectors == num_vectors);
        assert(x.data.size() == data.size());
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) data[i] += alpha * x.data[i];
    }

    void axpby(T alpha, const DistMultiVector<T>& x, T beta) {
        assert(x.num_vectors == num_vectors);
        assert(x.data.size() == data.size());
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = alpha * x.data[i] + beta * data[i];
        }
    }

    void pointwise_mult(const DistMultiVector<T>& other) {
        assert(other.num_vectors == num_vectors);
        assert(other.data.size() == data.size());
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] *= other.data[i];
        }
    }

    void pointwise_mult(const DistVector<T>& other) {
        int total_rows = local_rows + ghost_rows;
        assert(other.full_size() == total_rows);

        const T* vec_data = other.local_data();
        #pragma omp parallel for
        for (int row = 0; row < total_rows; ++row) {
            T* r = row_data(row);
            const T value = vec_data[row];
            for (int v = 0; v < num_vectors; ++v) {
                r[v] *= value;
            }
        }
    }

    // Helper to get a column as a DistVector (copy; columns are strided now).
    DistVector<T> get_col(int col) {
        DistVector<T> vec(graph);
        T* dst = vec.local_data();
        for (int row = 0; row < local_rows; ++row) {
            dst[row] = (*this)(row, col);
        }
        // Ghosts are not synced
        return vec;
    }

    // Set a column from a DistVector
    void set_col(int col, const DistVector<T>& vec) {
        const T* src = vec.local_data();
        for (int row = 0; row < local_rows; ++row) {
            (*this)(row, col) = src[row];
        }
    }

    void copy_from(const DistMultiVector<T>& other) {
        if (local_rows != other.local_rows || num_vectors != other.num_vectors) {
            throw std::runtime_error("DistMultiVector::copy_from: size mismatch");
        }
        std::copy(other.data.begin(), other.data.end(), data.begin());
    }

    DistMultiVector<T> duplicate() const {
        DistMultiVector<T> copy(graph, num_vectors);
        copy.copy_from(*this);
        return copy;
    }

    void swap(DistMultiVector<T>& other) {
        std::swap(data, other.data);
        std::swap(local_rows, other.local_rows);
        std::swap(ghost_rows, other.ghost_rows);
        std::swap(num_vectors, other.num_vectors);
        std::swap(ld, other.ld);
        std::swap(graph, other.graph);
    }

    // Batched dot product (global)
    // Returns a vector of size num_vectors
    std::vector<T> bdot(const DistMultiVector<T>& x) const {
        if (num_vectors != x.num_vectors) throw std::runtime_error("Dimension mismatch in bdot");

        std::vector<T> local_dots(num_vectors, T(0));

        #pragma omp parallel
        {
            std::vector<T> thread_dots(num_vectors, T(0));
            #pragma omp for nowait
            for (int row = 0; row < local_rows; ++row) {
                const T* r_this = row_data(row);
                const T* r_x = x.row_data(row);
                for (int v = 0; v < num_vectors; ++v) {
                    thread_dots[v] += ScalarTraits<T>::conjugate(r_this[v]) * r_x[v];
                }
            }
            #pragma omp critical
            {
                for (int v = 0; v < num_vectors; ++v) {
                    local_dots[v] += thread_dots[v];
                }
            }
        }

        if (graph->size == 1) return local_dots;

        std::vector<T> global_dots(num_vectors);
        MPI_Datatype type = get_mpi_type();
        MPI_Allreduce(local_dots.data(), global_dots.data(), num_vectors, type, MPI_SUM, graph->comm);

        return global_dots;
    }

    // Column-wise dot product (alias to bdot, matching implementation plan)
    void dot(const DistMultiVector<T>& other, std::vector<T>& results) const {
        results = bdot(other);
    }

    void set_random_normal(bool normalize = true) {
        std::random_device rd;
        const unsigned base_seed = rd();
        const int total_rows = local_rows + ghost_rows;
        #pragma omp parallel
        {
            // Per-thread generator (a shared engine would race); each thread
            // fills its static row slice. Fill only the real lanes: padding
            // must stay zero.
            std::mt19937 gen(base_seed + 0x9e3779b9u *
#ifdef _OPENMP
                static_cast<unsigned>(omp_get_thread_num())
#else
                0u
#endif
            );
            std::normal_distribution<double> d(0.0, 1.0);
            #pragma omp for schedule(static)
            for (int row = 0; row < total_rows; ++row) {
                T* r = row_data(row);
                for (int v = 0; v < num_vectors; ++v) {
                    if constexpr (std::is_same<T, std::complex<double>>::value) {
                        r[v] = std::complex<double>(d(gen), d(gen));
                    } else {
                        r[v] = (T)d(gen);
                    }
                }
            }
        }

        if (normalize) {
            std::vector<T> dots = this->bdot(*this);
            std::vector<T> factors(num_vectors);
            for (int v = 0; v < num_vectors; ++v) {
                double norm = std::sqrt(std::abs(dots[v]));
                factors[v] = 1.0 / norm;
            }
            for (int row = 0; row < total_rows; ++row) {
                T* r = row_data(row);
                for (int v = 0; v < num_vectors; ++v) {
                    r[v] *= factors[v];
                }
            }
        }
    }

    // Persistent buffers
    std::vector<T> send_buf;
    std::vector<T> recv_buf;

    // Communication diagnostics: cumulative wall seconds and call count of
    // ghost exchanges (pack + MPI + unpack); read/reset by the benchmark
    // harness to report comm fractions. Serial calls are not counted.
    double comm_seconds = 0.0;
    long comm_calls = 0;

    void reset_comm_stats() {
        comm_seconds = 0.0;
        comm_calls = 0;
    }

    // Sync ghosts for all vectors.
    // Wire format: per block, rows in order, each row's num_vectors lanes
    // (tight — padding lanes never travel). When ld == num_vectors a whole
    // block is one contiguous span and packs with a single memcpy.
    void sync_ghosts() {
        if (graph->size == 1) return;
        const double comm_t0 = MPI_Wtime();

        const auto& block_offsets = graph->block_offsets;
        const auto& send_counts_scalar = graph->send_counts_scalar;
        const auto& recv_counts_scalar = graph->recv_counts_scalar;
        const auto& send_displs_scalar = graph->send_displs_scalar;
        const auto& recv_displs_scalar = graph->recv_displs_scalar;

        size_t total_send_elements = static_cast<size_t>(send_displs_scalar[graph->size]) * num_vectors;
        if (send_buf.size() < total_send_elements) send_buf.resize(total_send_elements);

        size_t total_recv_elements = static_cast<size_t>(recv_displs_scalar[graph->size]) * num_vectors;
        if (recv_buf.size() < total_recv_elements) recv_buf.resize(total_recv_elements);

        // Element counts scale with num_vectors and overflow int at large
        // halos (the buffer sizes above are already size_t for that reason),
        // so the exchange uses 64-bit counts and the chunked safe_alltoallv —
        // a plain MPI_Alltoallv caps every message at 2^31 elements.
        const int np = graph->size;
        std::vector<size_t> s_counts(np), r_counts(np), s_displs(np), r_displs(np);
        for (int r = 0; r < np; ++r) {
            s_counts[r] = static_cast<size_t>(send_counts_scalar[r]) * num_vectors;
            r_counts[r] = static_cast<size_t>(recv_counts_scalar[r]) * num_vectors;
            s_displs[r] = static_cast<size_t>(send_displs_scalar[r]) * num_vectors;
            r_displs[r] = static_cast<size_t>(recv_displs_scalar[r]) * num_vectors;
        }

        int current_idx = 0;
        size_t buf_ptr = 0;

        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->send_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->send_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];

                const T* src = row_data(blk_offset);
                T* dst = send_buf.data() + buf_ptr;
                if (ld == num_vectors) {
                    std::memcpy(dst, src, static_cast<size_t>(blk_size) * num_vectors * sizeof(T));
                } else {
                    for (int i = 0; i < blk_size; ++i) {
                        std::memcpy(dst + static_cast<size_t>(i) * num_vectors,
                                    src + static_cast<size_t>(i) * ld,
                                    num_vectors * sizeof(T));
                    }
                }
                buf_ptr += static_cast<size_t>(blk_size) * num_vectors;
            }
        }

        // Exchange
        MPI_Datatype type = get_mpi_type();
        safe_alltoallv(send_buf.data(), s_counts, s_displs, type,
                       recv_buf.data(), r_counts, r_displs, type, graph->comm);

        // Unpack
        current_idx = 0;
        buf_ptr = 0;
        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->recv_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->recv_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];

                const T* src = recv_buf.data() + buf_ptr;
                T* dst = row_data(blk_offset);
                if (ld == num_vectors) {
                    std::memcpy(dst, src, static_cast<size_t>(blk_size) * num_vectors * sizeof(T));
                } else {
                    for (int i = 0; i < blk_size; ++i) {
                        std::memcpy(dst + static_cast<size_t>(i) * ld,
                                    src + static_cast<size_t>(i) * num_vectors,
                                    num_vectors * sizeof(T));
                    }
                }
                buf_ptr += static_cast<size_t>(blk_size) * num_vectors;
            }
        }

        comm_seconds += MPI_Wtime() - comm_t0;
        ++comm_calls;
    }

    void reduce_ghosts() {
        if (graph->size == 1) return;
        const double comm_t0 = MPI_Wtime();
        const auto& block_offsets = graph->block_offsets;
        const auto& send_counts_scalar = graph->send_counts_scalar;
        const auto& recv_counts_scalar = graph->recv_counts_scalar;
        const auto& send_displs_scalar = graph->send_displs_scalar;
        const auto& recv_displs_scalar = graph->recv_displs_scalar;

        size_t total_send_elements = static_cast<size_t>(recv_displs_scalar[graph->size]) * num_vectors;
        std::vector<T> s_buf(total_send_elements);

        int current_idx = 0;
        size_t buf_ptr = 0;
        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->recv_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->recv_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];

                const T* src = row_data(blk_offset);
                T* dst = s_buf.data() + buf_ptr;
                if (ld == num_vectors) {
                    std::memcpy(dst, src, static_cast<size_t>(blk_size) * num_vectors * sizeof(T));
                } else {
                    for (int i = 0; i < blk_size; ++i) {
                        std::memcpy(dst + static_cast<size_t>(i) * num_vectors,
                                    src + static_cast<size_t>(i) * ld,
                                    num_vectors * sizeof(T));
                    }
                }
                buf_ptr += static_cast<size_t>(blk_size) * num_vectors;
            }
        }

        size_t total_recv_elements = static_cast<size_t>(send_displs_scalar[graph->size]) * num_vectors;
        std::vector<T> r_buf(total_recv_elements);

        // Same 64-bit + chunked treatment as sync_ghosts (roles swapped).
        const int np = graph->size;
        std::vector<size_t> s_counts(np), r_counts(np), s_displs(np), r_displs(np);
        for (int r = 0; r < np; ++r) {
            s_counts[r] = static_cast<size_t>(recv_counts_scalar[r]) * num_vectors;
            r_counts[r] = static_cast<size_t>(send_counts_scalar[r]) * num_vectors;
            s_displs[r] = static_cast<size_t>(recv_displs_scalar[r]) * num_vectors;
            r_displs[r] = static_cast<size_t>(send_displs_scalar[r]) * num_vectors;
        }

        MPI_Datatype type = get_mpi_type();
        safe_alltoallv(s_buf.data(), s_counts, s_displs, type,
                       r_buf.data(), r_counts, r_displs, type, graph->comm);

        current_idx = 0;
        buf_ptr = 0;
        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->send_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->send_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];

                const T* src = r_buf.data() + buf_ptr;
                for (int i = 0; i < blk_size; ++i) {
                    T* dst = row_data(blk_offset + i);
                    const T* s = src + static_cast<size_t>(i) * num_vectors;
                    for (int v = 0; v < num_vectors; ++v) dst[v] += s[v];
                }
                buf_ptr += static_cast<size_t>(blk_size) * num_vectors;
            }
        }

        comm_seconds += MPI_Wtime() - comm_t0;
        ++comm_calls;
    }

    MPI_Datatype get_mpi_type() const;
};

template <> inline MPI_Datatype DistMultiVector<double>::get_mpi_type() const { return MPI_DOUBLE; }
template <> inline MPI_Datatype DistMultiVector<float>::get_mpi_type() const { return MPI_FLOAT; }
template <> inline MPI_Datatype DistMultiVector<int>::get_mpi_type() const { return MPI_INT; }
template <> inline MPI_Datatype DistMultiVector<std::complex<double>>::get_mpi_type() const { return MPI_CXX_DOUBLE_COMPLEX; }

} // namespace vbcsr

#endif
