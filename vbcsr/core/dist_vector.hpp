#ifndef VBCSR_DIST_VECTOR_HPP
#define VBCSR_DIST_VECTOR_HPP

#include "dist_graph.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#include "detail/storage/numa_buffer.hpp"
#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <random>

#include <complex>
#include "scalar_traits.hpp"

namespace vbcsr {

template <typename T>
class DistVector {
public:
    DistGraph* graph;
    // Owned + Ghosts. NumaVector, not std::vector: pages are first-touched by
    // an even parallel split so streaming applies use both sockets' memory
    // controllers (see detail/storage/numa_buffer.hpp).
    detail::NumaVector<T> data;
    int local_size;
    int ghost_size;

public:
    DistVector(DistGraph* g) : graph(g) {
        graph->get_vector_structure(local_size, ghost_size);
        data.resize(local_size + ghost_size);
    }

    // Accessors
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
    
    T* local_data() { return data.data(); }
    const T* local_data() const { return data.data(); }
    
    int size() const { return local_size; } // Only owned size
    int full_size() const { return local_size + ghost_size; }

    // Bind to a new graph (must have same owned structure).
    // Throws on any owned-structure mismatch, including graphs whose scalar
    // sizes coincide but whose block partitions differ (e.g. two 1x1 blocks
    // vs one 2x2 block). Used by BlockSpMat::mult and friends, so a
    // vector/matrix graph mismatch is a hard error, never silently accepted.
    void bind_to_graph(DistGraph* new_graph) {
        if (graph == new_graph) return;

        if (!graph->same_owned_structure(*new_graph)) {
            throw std::runtime_error(
                "DistVector::bind_to_graph: graph mismatch - target graph has a "
                "different owned block structure (partition or block sizes)");
        }

        int new_local_size, new_ghost_size;
        new_graph->get_vector_structure(new_local_size, new_ghost_size);

        graph = new_graph;
        ghost_size = new_ghost_size;
        // Resize preserves first local_size elements (owned data)
        data.resize(local_size + ghost_size);
    }

    // Operations
    void set_constant(T val) {
        #pragma omp parallel for
        for (int i = 0; i < local_size; ++i) data[i] = val;
    }

    void scale(T alpha) {
        #pragma omp parallel for
        for (int i = 0; i < local_size; ++i) data[i] *= alpha;
    }

    void axpy(T alpha, const DistVector<T>& x) {
        assert(x.size() == size());
        #pragma omp parallel for
        for (int i = 0; i < local_size; ++i) data[i] += alpha * x[i];
    }

    void axpby(T alpha, const DistVector<T>& x, T beta) {
        assert(x.size() == size());
        #pragma omp parallel for
        for (int i = 0; i < local_size; ++i) {
            data[i] = alpha * x[i] + beta * data[i];
        }
    }

    void pointwise_mult(const DistVector<T>& other) {
        assert(other.size() == size());
        #pragma omp parallel for
        for (int i = 0; i < local_size; ++i) {
            data[i] *= other[i];
        }
    }
    
    void conjugate() {
        #pragma omp parallel for
        for (int i = 0; i < local_size; ++i) {
            data[i] = ScalarTraits<T>::conjugate(data[i]);
        }
    }

    // Dot product (global)
    T dot(const DistVector<T>& x) const {
        T local_dot = 0;
        #pragma omp parallel for reduction(+:local_dot)
        for (int i = 0; i < local_size; ++i) local_dot += ScalarTraits<T>::conjugate(data[i]) * x[i];
        
        if (graph->size == 1) return local_dot;

        T global_dot;
        allreduce_sum(&local_dot, &global_dot, 1);
        return global_dot;
    }

    void set_random_normal(bool normalize = true) {
        std::random_device rd;
        const unsigned base_seed = rd();
        #pragma omp parallel
        {
            // Per-thread generator (a shared engine would race); each thread
            // fills its static slice.
            std::mt19937 gen(base_seed + 0x9e3779b9u *
#ifdef _OPENMP
                static_cast<unsigned>(omp_get_thread_num())
#else
                0u
#endif
            );
            std::normal_distribution<double> d(0.0, 1.0);
            #pragma omp for schedule(static)
            for (int i = 0; i < local_size; ++i) {
                if constexpr (std::is_same<T, std::complex<double>>::value) {
                    data[i] = std::complex<double>(d(gen), d(gen));
                } else {
                    data[i] = (T)d(gen);
                }
            }
        }
        
        if (normalize) {
            T n = this->dot(*this);
            double norm = std::sqrt(std::abs(n));
            this->scale(1.0 / norm);
        }
    }

    // Create a duplicate with the same structure and data
    DistVector<T> duplicate() const {
        DistVector<T> new_vec(graph);
        // Copy data (including ghosts, though ghosts might be stale)
        new_vec.data = data; 
        return new_vec;
    }

    // Copy from another vector
    void copy_from(const DistVector<T>& other) {
        if (data.size() != other.data.size()) {
            throw std::runtime_error("Vector size mismatch in copy_from");
        }
        // Copy data (including ghosts)
        std::copy(other.data.begin(), other.data.end(), data.begin());
    }

    void swap(DistVector<T>& other) {
        std::swap(data, other.data);
        std::swap(local_size, other.local_size);
        std::swap(ghost_size, other.ghost_size);
        std::swap(graph, other.graph);
    }

    // Sync ghosts
    // Persistent buffers
    std::vector<T> send_buf;
    // No recv_buf needed for zero-copy

    // Communication diagnostics: cumulative wall seconds and call count of
    // ghost exchanges (pack + MPI + unpack). The benchmark harness reads and
    // resets these to report the comm fraction of distributed applies on
    // real interconnects. Serial (size == 1) calls are not counted.
    double comm_seconds = 0.0;
    long comm_calls = 0;

    void reset_comm_stats() {
        comm_seconds = 0.0;
        comm_calls = 0;
    }

    void sync_ghosts() {
        if (graph->size == 1) return;
        const double comm_t0 = MPI_Wtime();

        // 1. Pack send buffers
        const auto& block_offsets = graph->block_offsets;
        const auto& send_counts_elems = graph->send_counts_scalar;
        const auto& recv_counts_elems = graph->recv_counts_scalar;
        const auto& sdispls_elems = graph->send_displs_scalar;
        const auto& rdispls_elems = graph->recv_displs_scalar;
        
        int total_send = sdispls_elems[graph->size];
        if (send_buf.size() < total_send) send_buf.resize(total_send);
        
        // Pack Data
        int current_idx = 0;
        int buf_ptr = 0;
        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->send_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->send_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];
                std::memcpy(send_buf.data() + buf_ptr, data.data() + blk_offset, blk_size * sizeof(T));
                buf_ptr += blk_size;
            }
        }
        
        // Exchange
        // Receive directly into data (zero-copy)
        // Ghosts start after owned elements.
        // Since ghosts are sorted by owner rank in DistGraph, and MPI_Alltoallv receives in rank order,
        // the data will be placed correctly.
        
        int n_owned_elems = block_offsets[graph->owned_global_indices.size()];
        T* recv_ptr = data.data() + n_owned_elems;
        MPI_Datatype type = get_mpi_type();
        MPI_Alltoallv(send_buf.data(), send_counts_elems.data(), sdispls_elems.data(), type,
                      recv_ptr, recv_counts_elems.data(), rdispls_elems.data(), type, graph->comm);

        comm_seconds += MPI_Wtime() - comm_t0;
        ++comm_calls;
    }

    void reduce_ghosts() {
        if (graph->size == 1) return;

        const auto& block_offsets = graph->block_offsets;
        const auto& send_counts_elems = graph->send_counts_scalar;
        const auto& recv_counts_elems = graph->recv_counts_scalar;
        const auto& sdispls_elems = graph->send_displs_scalar;
        const auto& rdispls_elems = graph->recv_displs_scalar;
        
        // In reduce_ghosts, we send our ghosts (recv_counts_scalar) back to owners (send_counts_scalar)
        int total_send = rdispls_elems[graph->size];
        std::vector<T> s_buf(total_send);
        
        int n_owned_elems = block_offsets[graph->owned_global_indices.size()];
        std::memcpy(s_buf.data(), data.data() + n_owned_elems, total_send * sizeof(T));
        
        int total_recv = sdispls_elems[graph->size];
        std::vector<T> r_buf(total_recv);
        
        MPI_Datatype type = get_mpi_type();
        MPI_Alltoallv(s_buf.data(), recv_counts_elems.data(), rdispls_elems.data(), type,
                      r_buf.data(), send_counts_elems.data(), sdispls_elems.data(), type, graph->comm);
        
        // Unpack and accumulate
        int current_idx = 0;
        int buf_ptr = 0;
        for (int r = 0; r < graph->size; ++r) {
            int n_blocks = graph->send_counts[r];
            for (int k = 0; k < n_blocks; ++k) {
                int blk_idx = graph->send_indices[current_idx++];
                int blk_size = graph->block_sizes[blk_idx];
                int blk_offset = block_offsets[blk_idx];
                
                T* dst = data.data() + blk_offset;
                const T* src = r_buf.data() + buf_ptr;
                for (int i = 0; i < blk_size; ++i) dst[i] += src[i];
                buf_ptr += blk_size;
            }
        }
    }

private:
    void allreduce_sum(const T* send, T* recv, int count) const {
        if (graph->size == 1 || graph->comm == MPI_COMM_NULL) {
            std::copy(send, send + count, recv);
            return;
        }
        MPI_Datatype type = get_mpi_type();
        MPI_Allreduce(send, recv, count, type, MPI_SUM, graph->comm);
    }

    MPI_Datatype get_mpi_type() const;
};

// Template specializations for MPI types
template <> inline MPI_Datatype DistVector<double>::get_mpi_type() const { return MPI_DOUBLE; }
template <> inline MPI_Datatype DistVector<float>::get_mpi_type() const { return MPI_FLOAT; }
template <> inline MPI_Datatype DistVector<int>::get_mpi_type() const { return MPI_INT; }
template <> inline MPI_Datatype DistVector<std::complex<double>>::get_mpi_type() const { return MPI_CXX_DOUBLE_COMPLEX; }

} // namespace vbcsr

#endif
