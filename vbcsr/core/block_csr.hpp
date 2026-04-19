#ifndef VBCSR_BLOCK_CSR_HPP
#define VBCSR_BLOCK_CSR_HPP

namespace vbcsr {

enum class AssemblyMode {
    INSERT,
    ADD
};

enum class MatrixLayout {
    RowMajor,
    ColMajor
};

enum class MatrixKind {
    CSR,
    BSR,
    VBCSR
};

inline const char* matrix_kind_name(MatrixKind kind) {
    switch (kind) {
        case MatrixKind::CSR:
            return "csr";
        case MatrixKind::BSR:
            return "bsr";
        case MatrixKind::VBCSR:
            return "vbcsr";
    }
    return "unknown";
}

} // namespace vbcsr

#include "dist_graph.hpp"
#include "dist_vector.hpp"
#include "dist_multivector.hpp"
#include "kernels.hpp"
#include "detail/backend_handle.hpp"
#include "detail/bsr_kernels.hpp"
#include "detail/csr_kernels.hpp"
#include "detail/axpby_ops.hpp"
#include "detail/bsr_spmm.hpp"
#include "detail/csr_spmm.hpp"
#include "detail/transpose_ops.hpp"
#include "detail/vbcsr_spmm.hpp"
#include "detail/block_payload_exchange.hpp"
#include "detail/vbcsr_kernels.hpp"
#include "mpi_utils.hpp"
#include <xmmintrin.h>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <complex>
#include <limits>
#include <string>
#include <type_traits>
#include <set>
#include <map>
#include <cstring>
#include <mutex>
#include <utility>

namespace vbcsr {

// Helper for Matrix Market output
template<typename T>
struct MMWriter {
    static void write(std::ostream& os, const T& v) {
        os << v;
    }
    static bool is_complex() { return false; }
};

template<typename T>
struct MMWriter<std::complex<T>> {
    static void write(std::ostream& os, const std::complex<T>& v) {
        os << v.real() << " " << v.imag();
    }
    static bool is_complex() { return true; }
};

template <typename T, typename Kernel = DefaultKernel<T>>
class BlockSpMat {
private:
    // Friend access for backend-specialized executors and builders.
    template <typename, typename>
    friend class BlockSpMat;
    template <typename>
    friend struct detail::CSRSpMMExecutor;
    template <typename>
    friend struct detail::BSRSpMMExecutor;
    template <typename>
    friend struct detail::CSRTransposeExecutor;
    template <typename>
    friend struct detail::BSRTransposeExecutor;
    template <typename>
    friend struct detail::VBCSRTransposeExecutor;
    template <typename>
    friend struct detail::CSRAxpbyExecutor;
    template <typename>
    friend struct detail::BSRAxpbyExecutor;
    template <typename>
    friend struct detail::VBCSRAxpbyExecutor;
    template <typename>
    friend struct detail::VBCSRSpMMExecutor;
    template <typename>
    friend struct detail::VBCSRShapeBatchExecutor;

    MatrixKind kind = MatrixKind::CSR;
    using VBCSRBackendStorage = detail::VBCSRMatrixBackend<T, Kernel>;
    using CSRBackendStorage = detail::CSRMatrixBackend<T>;
    using BSRBackendStorage = detail::BSRMatrixBackend<T>;
    using BackendHandle = detail::MatrixBackendHandle<T, Kernel>;
    using CommittedBackendStorage = BackendHandle;

    struct ConstructionToken {};

    BackendHandle backend_handle_;
    uint32_t configured_page_size_ = 0;

public:
    // Public facade state and view types.
    DistGraph* graph;
    using value_type = T;
    using KernelType = Kernel;

    struct ConstLocalBlockView {
        // Flat local nonzero-block index. Slots run across the local sparsity pattern:
        // for row r, the row's slots are graph->adj_ptr[r] .. graph->adj_ptr[r + 1) - 1.
        int slot = -1;
        int row = -1;
        int col = -1;
        int row_dim = 0;
        int col_dim = 0;
        size_t size = 0;
        const T* values = nullptr;
    };

    struct LocalBlockView {
        // Flat local nonzero-block index. In CSR this names one scalar entry; in
        // BSR/VBCSR it names one stored block entry.
        int slot = -1;
        int row = -1;
        int col = -1;
        int row_dim = 0;
        int col_dim = 0;
        size_t size = 0;
        T* values = nullptr;
    };
    
    // Cached block norms
    mutable std::vector<double> block_norms;
    mutable bool norms_valid = false;

    bool owns_graph = false;
    // Facade inspection helpers.
    const std::vector<double>& get_block_norms() const {
        if (!norms_valid) {
            block_norms = compute_block_norms();
            norms_valid = true;
        }
        return block_norms;
    }

    MatrixKind matrix_kind() const {
        return kind;
    }

    std::string matrix_kind_string() const {
        return std::string(vbcsr::matrix_kind_name(kind));
    }

    const std::vector<int>& row_ptr() const {
        return require_live_graph(graph, "row_ptr")->adj_ptr;
    }

    const std::vector<int>& col_ind() const {
        return require_live_graph(graph, "col_ind")->adj_ind;
    }

    size_t local_block_nnz() const {
        return graph->adj_ind.size();
    }

    uint32_t configured_page_size() const {
        return configured_page_size_;
    }

    void set_page_size(uint32_t page_size) {
        DistGraph* live_graph = require_live_graph(graph, "set_page_size");
        const uint32_t normalized = normalize_page_size(kind, live_graph, page_size);
        if (configured_page_size_ == normalized) {
            return;
        }
        configured_page_size_ = normalized;
        if (std::holds_alternative<std::monostate>(backend_handle_)) {
            return;
        }
        rebuild_backend_for_page_size("set_page_size");
    }

    uint32_t page_size() const {
        switch (kind) {
        case MatrixKind::CSR:
            return active_csr_backend().active_page_size();
        case MatrixKind::BSR:
            return active_bsr_backend().active_blocks_per_page();
        case MatrixKind::VBCSR:
            return active_vbcsr_backend().configured_blocks_per_page();
        }
        return 0;
    }

    bool has_contiguous_layout() const {
        return true;
    }

    void pack_contiguous() {
        require_assembled_for_state_copy("pack_contiguous");
    }

    size_t local_scalar_nnz() const {
        if (kind == MatrixKind::CSR) {
            return static_cast<size_t>(active_csr_backend().nnz_count());
        }
        if (kind == MatrixKind::BSR) {
            return static_cast<size_t>(active_bsr_backend().scalar_value_count());
        }
        return active_vbcsr_backend().local_scalar_nnz();
    }

    int shape_class_count() const {
        if (kind != MatrixKind::VBCSR) {
            return 0;
        }
        return active_vbcsr_backend().shape_class_count();
    }

    int block_row_count() const {
        const auto& structure_row_ptr = graph->adj_ptr;
        return structure_row_ptr.empty() ? 0 : static_cast<int>(structure_row_ptr.size()) - 1;
    }

    // Map a flat local block index back to its owning local block row.
    int block_row_from_slot(int slot) const {
        const auto& structure_row_ptr = graph->adj_ptr;
        auto it = std::upper_bound(structure_row_ptr.begin(), structure_row_ptr.end(), slot);
        return std::max(0, static_cast<int>(std::distance(structure_row_ptr.begin(), it) - 1));
    }

    // Return the local block column of a flat local block index.
    int block_col_from_slot(int slot) const {
        return graph->adj_ind[slot];
    }

    int block_row_dim(int local_row) const {
        return graph->block_sizes[local_row];
    }

    int block_col_dim(int local_col) const {
        return graph->block_sizes[local_col];
    }

    // Return the payload pointer for one flat local block index.
    const T* block_data(int idx) const {
        if (kind == MatrixKind::CSR) {
            return active_csr_backend().value_ptr(idx);
        }
        if (kind == MatrixKind::BSR) {
            return active_bsr_backend().block_ptr(idx);
        }
        if (kind == MatrixKind::VBCSR) {
            return active_vbcsr_backend().block_ptr_for_graph_block(idx);
        }
        throw std::logic_error("Unknown matrix backend in block_data");
    }

    // Mutable payload pointer for one flat local block index.
    T* mutable_block_data(int idx) {
        norms_valid = false;
        if (kind == MatrixKind::CSR) {
            return active_csr_backend().value_ptr(idx);
        }
        if (kind == MatrixKind::BSR) {
            return active_bsr_backend().block_ptr(idx);
        }
        if (kind == MatrixKind::VBCSR) {
            return active_vbcsr_backend().block_ptr_for_graph_block(idx);
        }
        throw std::logic_error("Unknown matrix backend in mutable_block_data");
    }

    // Number of scalar values stored in one flat local block index.
    size_t block_size_elements(int idx) const {
        if (kind == MatrixKind::CSR) {
            return 1;
        }
        if (kind == MatrixKind::BSR) {
            return active_bsr_backend().values_per_block();
        }
        if (kind == MatrixKind::VBCSR) {
            return active_vbcsr_backend().block_size_elements_for_graph_block(idx);
        }
        throw std::logic_error("Unknown matrix backend in block_size_elements");
    }

    // Lifetime and copy/move semantics.
    BlockSpMat(DistGraph* g)
        : BlockSpMat(
              require_live_graph(g, "BlockSpMat"),
              detect_matrix_kind(require_live_graph(g, "BlockSpMat")),
              false,
              ConstructionToken{}) {
        allocate_from_graph();
    }

    ~BlockSpMat() {
        clear_remote_assembly_state(this);
        release_graph_reference();
    }

    // Move constructor
    BlockSpMat(BlockSpMat&& other) noexcept : 
        kind(other.kind),
        backend_handle_(std::move(other.backend_handle_)),
        configured_page_size_(other.configured_page_size_),
        graph(other.graph),
        block_norms(std::move(other.block_norms)),
        norms_valid(other.norms_valid),
        owns_graph(other.owns_graph)
    {
        transfer_remote_assembly_state(&other, this);
        other.graph = nullptr;
        other.owns_graph = false;
        other.backend_handle_ = std::monostate{};
    }

    // Move assignment
    BlockSpMat& operator=(BlockSpMat&& other) noexcept {
        if (this != &other) {
            clear_remote_assembly_state(this);
            release_graph_reference();
            graph = other.graph;
            kind = other.kind;
            backend_handle_ = std::move(other.backend_handle_);
            configured_page_size_ = other.configured_page_size_;
            block_norms = std::move(other.block_norms);
            norms_valid = other.norms_valid;
            owns_graph = other.owns_graph;
            transfer_remote_assembly_state(&other, this);
            
            other.graph = nullptr;
            other.owns_graph = false;
            other.backend_handle_ = std::monostate{};
        }
        return *this;
    }

    // Disable copy (use duplicate() instead)
    BlockSpMat(const BlockSpMat&) = delete;
    BlockSpMat& operator=(const BlockSpMat&) = delete;

    // Create a deep copy of the matrix
    BlockSpMat<T, Kernel> duplicate(bool independent_graph = true) const {
        require_assembled_for_state_copy("duplicate");
        DistGraph* new_graph = graph;
        bool new_owns_graph = false;
        if (independent_graph && graph) {
            new_graph = graph->duplicate();
            new_owns_graph = true;
        } else {
            // Matrix-owned graphs become delete-on-last-release before sharing.
            prepare_graph_for_shared_use();
        }
        BlockSpMat<T, Kernel> new_mat(new_graph);
        new_mat.owns_graph = new_owns_graph;
        if (new_mat.owns_graph) {
            new_mat.graph->enable_matrix_lifetime_management();
        }
        new_mat.set_page_size(configured_page_size_);
        new_mat.copy_from(*this);
        if (norms_valid) {
            new_mat.block_norms = block_norms;
            new_mat.norms_valid = true;
        }
        return new_mat;
    }

    // Matrix operations and conversions.
    void allocate_from_graph() {
        attach_backend(build_backend_for_structure(kind, graph, configured_page_size_));
    }

    // Add a block (local or remote)
    // Input data layout is specified by `layout`
    void add_block(int global_row, int global_col, const T* data, int rows, int cols, AssemblyMode mode = AssemblyMode::ADD, MatrixLayout layout = MatrixLayout::ColMajor) {
        int owner = graph->find_owner(global_row);
        
        if (owner == graph->rank) {
            // Local: Try to update immediately
            if (graph->global_to_local.find(global_row) != graph->global_to_local.end()) {
                int local_row = graph->global_to_local.at(global_row);
                
                int local_col = -1;
                if (graph->global_to_local.count(global_col)) {
                    local_col = graph->global_to_local.at(global_col);
                }
                
                if (local_col != -1) {
                    // update_local_block handles layout
                    if (update_local_block(local_row, local_col, data, rows, cols, mode, layout)) {
                        return; // Success
                    }
                }
            }
            std::cerr << "Warning: Block (row=" << global_row << ", col=" << global_col << ") not found in local graph. Ignoring." << std::endl;
            return;
        } 
        
        // Remote
        if (owner < 0 || owner >= graph->size) {
             std::cerr << "Warning: Block (row=" << global_row << ", col=" << global_col << ") belongs to invalid rank " << owner << ". Ignoring." << std::endl;
             return;
        }

        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& thread_remote_blocks = remote_assembly_buffers();
        if (tid >= static_cast<int>(thread_remote_blocks.size())) {
            thread_remote_blocks.resize(static_cast<size_t>(tid) + 1);
        }

        auto& blocks_map = thread_remote_blocks[tid][owner];
        std::pair<int, int> key = {global_row, global_col};
        auto it = blocks_map.find(key);
        
        if (it != blocks_map.end()) {
            // Block exists in buffer
            PendingBlock& pb = it->second;
            
            // Check dims
            if (pb.rows != rows || pb.cols != cols) {
                throw std::runtime_error("Dimension mismatch in add_block accumulation");
            }
            
            // We store pending blocks in ColMajor (canonical format for transport)
            
            if (mode == AssemblyMode::INSERT) {
                // Overwrite
                pb.mode_code = static_cast<int>(AssemblyMode::INSERT);
                if (layout == MatrixLayout::ColMajor) {
                    std::memcpy(pb.data.data(), data, rows * cols * sizeof(T));
                } else {
                    // Transpose RowMajor -> ColMajor
                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            pb.data[j * rows + i] = data[i * cols + j];
                        }
                    }
                }
            } else {
                // ADD
                // Accumulate
                if (layout == MatrixLayout::ColMajor) {
                    for (size_t i = 0; i < pb.data.size(); ++i) {
                        pb.data[i] += data[i];
                    }
                } else {
                    // Transpose add
                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            pb.data[j * rows + i] += data[i * cols + j];
                        }
                    }
                }
            }
        } else {
            // New block
            PendingBlock pb;
            pb.rows = rows;
            pb.cols = cols;
            pb.mode_code = static_cast<int>(mode);
            pb.data.resize(rows * cols);
            
            if (layout == MatrixLayout::ColMajor) {
                std::memcpy(pb.data.data(), data, rows * cols * sizeof(T));
            } else {
                // Transpose RowMajor -> ColMajor
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        pb.data[j * rows + i] = data[i * cols + j];
                    }
                }
            }
            blocks_map[key] = std::move(pb);
        }
    }

    // Finalize assembly by exchanging remote blocks
    void assemble() {
        if (graph->size == 1) { // mpisize, serial fallback
            clear_remote_assembly_state(this);
            norms_valid = false;
            return;
        }
        int size = graph->size;
        auto& thread_remote_blocks = remote_assembly_buffers();
        
        // 1. Counting pass
        std::vector<size_t> send_counts(size, 0);
        
        for (const auto& remote_blocks : thread_remote_blocks) {
            for (const auto& kv : remote_blocks) {
                int target = kv.first;
                size_t bytes = 0;
                for (const auto& inner_kv : kv.second) {
                    bytes += 5 * sizeof(int) + inner_kv.second.data.size() * sizeof(T);
                }
                send_counts[target] += bytes;
            }
        }
        
        // 2. Exchange counts and setup displacements
        std::vector<size_t> recv_counts(size);
        if (graph->size > 1) {
            MPI_Alltoall(send_counts.data(), sizeof(size_t), MPI_BYTE, recv_counts.data(), sizeof(size_t), MPI_BYTE, graph->comm);
        } else {
            recv_counts = send_counts;
        }
        
        std::vector<size_t> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
        }
        
        // 3. Pack flat buffer
        std::vector<char> send_blob(sdispls[size]);
        std::vector<size_t> current_offsets = sdispls; // Track current write position per rank

        for (auto& remote_blocks : thread_remote_blocks) {
            for (auto& kv : remote_blocks) {
                int target = kv.first;
                char* ptr = send_blob.data() + current_offsets[target];
                for (auto& inner_kv : kv.second) {
                    const int g_row = inner_kv.first.first;
                    const int g_col = inner_kv.first.second;
                    auto& blk = inner_kv.second;
                    size_t data_bytes = blk.data.size() * sizeof(T);
                    
                    std::memcpy(ptr, &g_row, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(ptr, &g_col, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(ptr, &blk.rows, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(ptr, &blk.cols, sizeof(int)); ptr += sizeof(int);
                    int mode_int = blk.mode_code;
                    std::memcpy(ptr, &mode_int, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(ptr, blk.data.data(), data_bytes); ptr += data_bytes;
                }
                current_offsets[target] = ptr - send_blob.data();
            }
        }
        
        // 4. Exchange data
        std::vector<char> recv_blob(rdispls[size]);
        if (graph->size > 1) {
            safe_alltoallv(send_blob.data(), send_counts, sdispls, MPI_BYTE,
                           recv_blob.data(), recv_counts, rdispls, MPI_BYTE, graph->comm);
        } else {
            recv_blob = send_blob;
        }
                  
        // 5. Process received
        for(int i=0; i<size; ++i) {
            char* ptr = recv_blob.data() + rdispls[i];
            char* end = recv_blob.data() + rdispls[i+1];
            
            while(ptr < end) {
                int g_row, g_col, rows, cols, mode_int;
                std::memcpy(&g_row, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&g_col, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&rows, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&cols, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&mode_int, ptr, sizeof(int)); ptr += sizeof(int);
                AssemblyMode mode = static_cast<AssemblyMode>(mode_int);
                
                if (graph->global_to_local.find(g_row) == graph->global_to_local.end()) {
                     std::cerr << "Warning: Received block for non-owned row " << g_row << ". Ignoring." << std::endl;
                     ptr += rows * cols * sizeof(T);
                     continue;
                }
                int l_row = graph->global_to_local.at(g_row);

                int l_col = -1;
                if (graph->global_to_local.count(g_col)) {
                    l_col = graph->global_to_local.at(g_col);
                }

                size_t data_bytes = rows * cols * sizeof(T);

                if (l_col == -1 || !update_local_block(l_row, l_col, (const T*)ptr, rows, cols, mode, MatrixLayout::ColMajor)) {
                    std::cerr << "Warning: Received block (row=" << g_row << ", col=" << g_col << ") not in graph. Ignoring." << std::endl;
                    // Fall through to ptr += data_bytes
                }

                ptr += data_bytes;
            }
        }

        clear_remote_assembly_state(this);
        norms_valid = false;
    }

    // Internal extension hook used by image accumulation and remote assembly.
    bool update_local_block(
        int local_row,
        int local_col,
        const T* data,
        int rows,
        int cols,
        AssemblyMode mode,
        MatrixLayout layout = MatrixLayout::ColMajor);

    // Matrix-Vector Multiplication
    void mult(DistVector<T>& x, DistVector<T>& y) {
        if (kind == MatrixKind::CSR) {
            detail::csr_mult(graph, detail::require_csr_backend<T, Kernel>(backend_handle_), x, y);
            return;
        }
        if (kind == MatrixKind::BSR) {
            detail::bsr_mult(graph, detail::require_bsr_backend<T, Kernel>(backend_handle_), x, y);
            return;
        }
        detail::VBCSRShapeBatchExecutor<BlockSpMat<T, Kernel>>::mult(*this, x, y);
    }
    
    // Refined mult with offsets
    void mult_optimized(DistVector<T>& x, DistVector<T>& y) {
        mult(x, y);
    }

    // Matrix-Matrix Multiplication (Dense RHS)
    void mult_dense(DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        if (kind == MatrixKind::CSR) {
            detail::csr_mult_dense(graph, detail::require_csr_backend<T, Kernel>(backend_handle_), X, Y);
            return;
        }
        if (kind == MatrixKind::BSR) {
            detail::bsr_mult_dense(graph, detail::require_bsr_backend<T, Kernel>(backend_handle_), X, Y);
            return;
        }
        detail::VBCSRShapeBatchExecutor<BlockSpMat<T, Kernel>>::mult_dense(*this, X, Y);
    }

    // Adjoint Matrix-Vector Multiplication: y = A^dagger * x
    void mult_adjoint(DistVector<T>& x, DistVector<T>& y) {
        if (kind == MatrixKind::CSR) {
            detail::csr_mult_adjoint(graph, detail::require_csr_backend<T, Kernel>(backend_handle_), x, y);
            return;
        }
        if (kind == MatrixKind::BSR) {
            detail::bsr_mult_adjoint(graph, detail::require_bsr_backend<T, Kernel>(backend_handle_), x, y);
            return;
        }
        detail::VBCSRShapeBatchExecutor<BlockSpMat<T, Kernel>>::mult_adjoint(*this, x, y);
    }

    // Adjoint Matrix-Matrix Multiplication: Y = A^dagger * X
    void mult_dense_adjoint(DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        if (kind == MatrixKind::CSR) {
            detail::csr_mult_dense_adjoint(graph, detail::require_csr_backend<T, Kernel>(backend_handle_), X, Y);
            return;
        }
        if (kind == MatrixKind::BSR) {
            detail::bsr_mult_dense_adjoint(graph, detail::require_bsr_backend<T, Kernel>(backend_handle_), X, Y);
            return;
        }
        detail::VBCSRShapeBatchExecutor<BlockSpMat<T, Kernel>>::mult_dense_adjoint(*this, X, Y);
    }

    // Utilities
    void scale(T alpha) {
        #pragma omp parallel for
        for (size_t i = 0; i < local_block_nnz(); ++i) {
            T* block = mutable_block_data(static_cast<int>(i));
            const size_t size = block_size_elements(static_cast<int>(i));
            for (size_t j = 0; j < size; ++j) {
                block[j] *= alpha;
            }
        }
        // Norms are scaled by abs(alpha)
        if (norms_valid) {
            double abs_alpha = std::abs(alpha);
            #pragma omp parallel for
            for(size_t i=0; i<block_norms.size(); ++i) block_norms[i] *= abs_alpha;
        }
    }

    void conjugate() {
        if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
            #pragma omp parallel for
            for (size_t i = 0; i < local_block_nnz(); ++i) {
                T* block = mutable_block_data(static_cast<int>(i));
                const size_t size = block_size_elements(static_cast<int>(i));
                for (size_t j = 0; j < size; ++j) {
                    block[j] = std::conj(block[j]);
                }
            }
        }
    }

    void copy_from(const BlockSpMat<T, Kernel>& other) {
        require_assembled_for_state_copy("copy_from");
        other.require_assembled_for_state_copy("copy_from");
        if (!has_same_logical_structure(other)) {
            throw std::runtime_error("Incompatible graph structure in copy_from");
        }

        int n_rows = graph->adj_ptr.size() - 1;
        for (int i = 0; i < n_rows; ++i){
            int start = graph->adj_ptr[i];
            int end = graph->adj_ptr[i+1];
            for (int k = start; k < end; ++k){
                T* block_val = mutable_block_data(k);
                const T* block_val_other = other.block_data(k);
                std::memcpy(block_val, block_val_other, block_size_elements(k) * sizeof(T));
            }
        }
        norms_valid = false;
    }
    

    // Return real part as a new matrix (Double)
    // Only valid if T is complex, otherwise returns copy?
    // Actually we need to return BlockSpMat<RealType>
    auto get_real() const {
        using RealT = typename ScalarTraits<T>::real_type;
        prepare_graph_for_shared_use();
        BlockSpMat<RealT, DefaultKernel<RealT>> res(graph);
        res.set_page_size(configured_page_size_);
        
        // Copy and cast data
        #pragma omp parallel for
        for (size_t i = 0; i < local_block_nnz(); ++i) {
             RealT* dest = res.mutable_block_data(static_cast<int>(i));
             const T* src = block_data(static_cast<int>(i));
             const size_t size = block_size_elements(static_cast<int>(i));
              for(size_t j=0; j<size; ++j) {
                  if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
                      dest[j] = src[j].real();
                  } else {
                     dest[j] = src[j];
                 }
             }
        }
        return res;
    }

    auto get_imag() const {
        using RealT = typename ScalarTraits<T>::real_type;
        prepare_graph_for_shared_use();
        BlockSpMat<RealT, DefaultKernel<RealT>> res(graph);
        res.set_page_size(configured_page_size_);
        
        #pragma omp parallel for
        for (size_t i = 0; i < local_block_nnz(); ++i) {
             RealT* dest = res.mutable_block_data(static_cast<int>(i));
             const T* src = block_data(static_cast<int>(i));
             const size_t size = block_size_elements(static_cast<int>(i));
              for(size_t j=0; j<size; ++j) {
                  if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
                      dest[j] = src[j].imag();
                  } else {
                     dest[j] = 0;
                 }
             }
        }
        return res;
    }

    // Get a specific block (copy)
    std::vector<T> get_block(int local_row, int local_col, MatrixLayout layout = MatrixLayout::RowMajor) const {
        int start = graph->adj_ptr[local_row];
        int end = graph->adj_ptr[local_row+1];
        
        for (int k = start; k < end; ++k) {
            if (graph->adj_ind[k] == local_col) {
                int r_dim = block_row_dim(local_row);
                int c_dim = block_col_dim(local_col);
                size_t sz = block_size_elements(k);
                
                std::vector<T> result(sz);
                const T* block_ptr = block_data(k);
                
                if (layout == MatrixLayout::ColMajor) {
                    std::memcpy(result.data(), block_ptr, sz * sizeof(T));
                } else {
                    // Transpose ColMajor -> RowMajor
                    for (int r = 0; r < r_dim; ++r) {
                        for (int c = 0; c < c_dim; ++c) {
                            result[r * c_dim + c] = block_ptr[c * r_dim + r];
                        }
                    }
                }
                return result;
            }
        }
        return std::vector<T>(); // Empty if not found (or throw?)
    }

    // Export packed data for Python/Scipy
    // Returns a single vector containing all blocks concatenated.
    // Blocks are ordered by (row, col) as in col_ind.
    // If layout is RowMajor, blocks are transposed (if internal is ColMajor).
    std::vector<T> get_values(MatrixLayout layout = MatrixLayout::RowMajor) const {
        // Calculate total size
        size_t total_size = 0;
        for (int slot = 0; slot < static_cast<int>(graph->adj_ind.size()); ++slot) {
            total_size += block_size_elements(slot);
        }
        
        std::vector<T> result(total_size);
        size_t offset = 0;
        
        int n_rows = graph->adj_ptr.size() - 1;
        for (int i = 0; i < n_rows; ++i) {
            int r_dim = block_row_dim(i);
            int start = graph->adj_ptr[i];
            int end = graph->adj_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = graph->adj_ind[k];
                int c_dim = block_col_dim(col);
                const T* block_ptr = block_data(k);
                size_t sz = block_size_elements(k); // should be r_dim * c_dim
                
                T* dest = result.data() + offset;
                
                // Internal storage is ColMajor
                if (layout == MatrixLayout::ColMajor) {
                    std::memcpy(dest, block_ptr, sz * sizeof(T));
                } else {
                    // Transpose ColMajor -> RowMajor
                    // Internal: A[j*r_dim + i]
                    // Output:   A[i*c_dim + j]
                    for (int r = 0; r < r_dim; ++r) {
                        for (int c = 0; c < c_dim; ++c) {
                            dest[r * c_dim + c] = block_ptr[c * r_dim + r];
                        }
                    }
                }
                offset += sz;
            }
        }
        return result;
    }

    void axpby(T alpha, const BlockSpMat<T, Kernel>& X, T beta) {
        if (this == &X) {
            this->scale(alpha + beta);
            return;
        }

        ensure_same_backend_family(*this, X, "axpby");
        if (kind == MatrixKind::VBCSR) {
            ensure_vbcsr_binary_compatibility(*this, X);
        }
        if (kind == MatrixKind::CSR) {
            ensure_csr_binary_compatibility(*this, X);
        }
        if (kind == MatrixKind::BSR) {
            ensure_bsr_binary_compatibility(*this, X);
        }

        // Optimization Checks
        if (alpha == T(0)) {
            this->scale(beta);
            return;
        }

        if (kind == MatrixKind::CSR && X.kind == MatrixKind::CSR) {
            detail::CSRAxpbyExecutor<BlockSpMat<T, Kernel>>::run(*this, X, alpha, beta);
            return;
        }
        if (kind == MatrixKind::BSR && X.kind == MatrixKind::BSR) {
            detail::BSRAxpbyExecutor<BlockSpMat<T, Kernel>>::run(*this, X, alpha, beta);
            return;
        }
        detail::VBCSRAxpbyExecutor<BlockSpMat<T, Kernel>>::run(*this, X, alpha, beta);
    }

    void axpy(T alpha, const BlockSpMat<T, Kernel>& other) {
        axpby(alpha, other, T(1));
    }

    // Add alpha to diagonal elements
    void shift(T alpha) {
        int n_rows = graph->adj_ptr.size() - 1;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            // Find diagonal block (col == local_col corresponding to local_row i)
            // Local row i corresponds to global row G = graph->get_global_index(i)
            
            int start = graph->adj_ptr[i];
            int end = graph->adj_ptr[i+1];
            
            int global_row = graph->get_global_index(i);
            
            for (int k = start; k < end; ++k) {
                int local_col = graph->adj_ind[k];
                // Check if this local_col maps to global_row
                
                if (graph->get_global_index(local_col) == global_row) {
                    // Found diagonal block
                    T* target = mutable_block_data(k);
                    int r_dim = graph->block_sizes[i];
                    int c_dim = graph->block_sizes[local_col];
                    
                    // Add alpha to diagonal of the block
                    // Block is ColMajor or RowMajor? Internal is ColMajor.
                    // Diagonal elements are at [j*r_dim + j] for j=0..min(r,c)
                    
                    int min_dim = std::min(r_dim, c_dim);
                    for (int j = 0; j < min_dim; ++j) {
                        target[j * r_dim + j] += alpha;
                    }
                    break; 
                }
            }
        }
        norms_valid = false;
    }

    // Add scalar diagonal entries: H_ii += v_i
    void add_diagonal(const DistVector<T>& diag) {
        int n_rows = graph->adj_ptr.size() - 1;
        const int required_size = graph->block_offsets[n_rows];
        if (diag.size() < required_size) {
            throw std::runtime_error("Vector size too small for add_diagonal");
        }

        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = graph->adj_ptr[i];
            int end = graph->adj_ptr[i+1];
            
            int global_row = graph->get_global_index(i);
            const int row_offset = graph->block_offsets[i];
            
            for (int k = start; k < end; ++k) {
                int local_col = graph->adj_ind[k];
                if (graph->get_global_index(local_col) == global_row) {
                    // Found diagonal block
                    T* target = mutable_block_data(k);
                    int r_dim = graph->block_sizes[i];
                    int c_dim = graph->block_sizes[local_col];
                    
                    int min_dim = std::min(r_dim, c_dim);
                    for (int j = 0; j < min_dim; ++j) {
                        target[j * r_dim + j] += diag[row_offset + j];
                    }
                    break; 
                }
            }
        }
        norms_valid = false;
    }

    // Compute C = [H, R] where R is a scalar diagonal operator stored in a DistVector.
    // Each block entry follows C(rc) = H(rc) * (R_col(c) - R_row(r)).
    void commutator_diagonal(const DistVector<T>& diag, BlockSpMat<T, Kernel>& result) {
        if (!result.has_same_logical_structure(*this)) {
            result = this->duplicate(false);
            result.fill(T(0));
        }

        const std::vector<T>& R = diag.data; // Includes ghosts
        
        int n_rows = graph->adj_ptr.size() - 1;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = graph->adj_ptr[i];
            int end = graph->adj_ptr[i+1];
            const int row_offset = graph->block_offsets[i];
            const int row_dim = graph->block_sizes[i];
            
            for (int k = start; k < end; ++k) {
                int col = graph->adj_ind[k];
                const int col_offset = graph->block_offsets[col];
                const int col_dim = graph->block_sizes[col];
                const T* H_ptr = block_data(k);
                T* C_ptr = result.mutable_block_data(k);
                
                for (int c = 0; c < col_dim; ++c) {
                    for (int r = 0; r < row_dim; ++r) {
                        const int idx = c * row_dim + r;
                        C_ptr[idx] = H_ptr[idx] * (R[col_offset + c] - R[row_offset + r]);
                    }
                }
            }
        }
    }

    void filter_blocks(double threshold) {
        if (threshold <= 0.0) return;

        require_assembled_for_state_copy("filter_blocks");
        get_block_norms();
        const int n_rows = graph->adj_ptr.size() - 1;
        std::vector<std::vector<int>> new_adj_global(n_rows);
        int local_removed_any = 0;
        #pragma omp parallel for reduction(|:local_removed_any)
        for (int row = 0; row < n_rows; ++row) {
            new_adj_global[row].reserve(static_cast<size_t>(graph->adj_ptr[row + 1] - graph->adj_ptr[row]));
            for (int slot = graph->adj_ptr[row]; slot < graph->adj_ptr[row + 1]; ++slot) {
                if (block_norms[slot] >= threshold) {
                    new_adj_global[row].push_back(graph->get_global_index(graph->adj_ind[slot]));
                } else {
                    local_removed_any = 1;
                }
            }
        }
        int global_removed_any = local_removed_any;
        if (graph->size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &global_removed_any, 1, MPI_INT, MPI_MAX, graph->comm);
        }
        if (global_removed_any == 0) {
            return;
        }

        DistGraph* new_graph = new DistGraph(graph->comm);
        const int n_owned = graph->owned_global_indices.size();
        std::vector<int> owned_block_sizes(
            graph->block_sizes.begin(),
            graph->block_sizes.begin() + n_owned);
        new_graph->construct_distributed(graph->owned_global_indices, owned_block_sizes, new_adj_global);

        std::vector<double> new_norms(new_graph->adj_ind.size(), 0.0);
        const auto copy_kept_blocks = [&](auto& result) {
            #pragma omp parallel for
            for (int row = 0; row < n_rows; ++row) {
                size_t kept_index = 0;
                const auto& kept_globals = new_adj_global[row];
                for (int slot = graph->adj_ptr[row]; slot < graph->adj_ptr[row + 1]; ++slot) {
                    if (block_norms[slot] < threshold) {
                        continue;
                    }
                    const int global_col = kept_globals[kept_index++];
                    const int dest_col = new_graph->global_to_local.at(global_col);
                    const int dest_start = new_graph->adj_ptr[row];
                    const int dest_end = new_graph->adj_ptr[row + 1];
                    auto begin = new_graph->adj_ind.begin() + dest_start;
                    auto end = new_graph->adj_ind.begin() + dest_end;
                    auto it = std::lower_bound(begin, end, dest_col);
                    if (it == end || *it != dest_col) {
                        throw std::runtime_error("filter_blocks could not locate destination block");
                    }
                    const int dest_graph_block =
                        static_cast<int>(std::distance(new_graph->adj_ind.begin(), it));
                    const size_t size = block_size_elements(slot);
                    std::memcpy(
                        result.mutable_block_data(dest_graph_block),
                        block_data(slot),
                        size * sizeof(T));
                    new_norms[dest_graph_block] = block_norms[slot];
                }
            }
        };

        BlockSpMat result(new_graph);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(configured_page_size_);
        copy_kept_blocks(result);
        *this = std::move(result);

        block_norms = std::move(new_norms);
        norms_valid = true;
    }

    BlockSpMat spmm(const BlockSpMat& B, double threshold, bool transA = false, bool transB = false) const {
        ensure_same_backend_family(*this, B, "spmm");

        if (transA) {
            BlockSpMat A_T = this->transpose();
            return A_T.spmm(B, threshold, false, transB);
        }

        if (transB) {
            BlockSpMat B_T = B.transpose();
            return this->spmm(B_T, threshold, transA, false);
        }

        if (kind == MatrixKind::CSR && B.kind == MatrixKind::CSR) {
            return detail::CSRSpMMExecutor<BlockSpMat<T, Kernel>>::run(*this, B, threshold);
        }
        if (kind == MatrixKind::BSR && B.kind == MatrixKind::BSR) {
            return detail::BSRSpMMExecutor<BlockSpMat<T, Kernel>>::run(*this, B, threshold);
        }
        
        return detail::VBCSRSpMMExecutor<BlockSpMat<T, Kernel>>::run(*this, B, threshold);
    }

    BlockSpMat spmm_self(double threshold, bool transA = false) {
        return spmm(*this, threshold, transA, false);
    }

    BlockSpMat add(const BlockSpMat& B, double alpha = 1.0, double beta = 1.0) {
        ensure_same_backend_family(*this, B, "add");
        BlockSpMat C = this->duplicate();
        C.scale(alpha);
        C.axpy(beta, B);
        return C;
    }

    void fill(T val) {
        #pragma omp parallel for
        for (int i = 0; i < graph->adj_ind.size(); ++i) {
            T* data = mutable_block_data(i);
            const size_t size = block_size_elements(i);
            std::fill(data, data + size, val);
        }
        norms_valid = false;
    }

    BlockSpMat transpose() const {
        if (kind == MatrixKind::CSR) {
            return detail::CSRTransposeExecutor<BlockSpMat<T, Kernel>>::run(*this);
        }
        if (kind == MatrixKind::BSR) {
            return detail::BSRTransposeExecutor<BlockSpMat<T, Kernel>>::run(*this);
        }

        return detail::VBCSRTransposeExecutor<BlockSpMat<T, Kernel>>::run(*this);
    }

    void save_matrix_market(const std::string& filename) {
        if (graph->size > 1) {
            throw std::runtime_error("save_matrix_market only supported for serial execution (MPI size = 1)");
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        file << "%%MatrixMarket matrix coordinate " << (MMWriter<T>::is_complex() ? "complex" : "real") << " general\n";

        // Calculate total NNZ and dimensions
        size_t element_nnz = 0;
        int total_rows = 0;
        int total_cols = 0;
        
        int n_owned = graph->adj_ptr.size() - 1;
        
        // Calculate total rows
        for (int i = 0; i < n_owned; ++i) {
            total_rows += graph->block_sizes[i];
        }
        
        // Calculate total cols (assuming serial, so owned == all local cols)
        for (size_t i = 0; i < graph->block_sizes.size(); ++i) {
             total_cols += graph->block_sizes[i];
        }
        
        // Count NNZ
        for (int i = 0; i < n_owned; ++i) {
            int r_dim = graph->block_sizes[i];
            int start = graph->adj_ptr[i];
            int end = graph->adj_ptr[i+1];
            for (int k = start; k < end; ++k) {
                int col = graph->adj_ind[k];
                int c_dim = graph->block_sizes[col];
                element_nnz += r_dim * c_dim;
            }
        }
        
        file << total_rows << " " << total_cols << " " << element_nnz << "\n";
        
        // Write data
        file << std::scientific << std::setprecision(16);
        
        for (int i = 0; i < n_owned; ++i) {
            int r_dim = graph->block_sizes[i];
            int row_start_idx = graph->block_offsets[i] + 1; // 1-based
            
            int start = graph->adj_ptr[i];
            int end = graph->adj_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = graph->adj_ind[k];
                int c_dim = graph->block_sizes[col];
                int col_start_idx = graph->block_offsets[col] + 1; // 1-based
                
                // long long offset = blk_ptr[k];
                const T* block_data = this->block_data(k);
                
                // Block is stored in ColMajor
                for (int c = 0; c < c_dim; ++c) {
                    for (int r = 0; r < r_dim; ++r) {
                        T value = block_data[c * r_dim + r];
                        
                        // Write (row, col, val)
                        file << (row_start_idx + r) << " " << (col_start_idx + c) << " ";
                        MMWriter<T>::write(file, value);
                        file << "\n";
                    }
                }
            }
        }
    }

    using GhostBlockRef = detail::GhostBlockRef<T>;

    // void numeric_multiply(
    //     const BlockSpMat& B,
    //     const std::map<int, std::vector<GhostBlockRef>>& ghost_rows,
    //     BlockSpMat& C,
    //     double threshold,
    //     const std::vector<double>& A_norms,
    //     const std::vector<double>& B_local_norms) const {
    //     detail::VBCSRSpMMExecutor<BlockSpMat<T, Kernel>>::run_numeric(
    //         *this,
    //         B,
    //         ghost_rows,
    //         C,
    //         threshold,
    //         A_norms,
    //         B_local_norms);
    // }

    using BlockData = detail::FetchedBlock<T>;
    using FetchContext = detail::FetchedBlockContext<T>;

    // Construct a submatrix from fetched data
    BlockSpMat<T, Kernel> construct_submatrix(const std::vector<int>& global_indices, const FetchContext& ctx) {
        // 1. Map global index to local index in the submatrix (0 to M-1)
        std::map<int, int> global_to_sub;
        for(size_t i=0; i<global_indices.size(); ++i) {
            global_to_sub[global_indices[i]] = i;
        }

        int M = global_indices.size();
        std::vector<int> sub_block_sizes(M, 0);
        std::vector<std::vector<int>> sub_adj(M);
        
        // 2. Fill sizes
        for(int gid : global_indices) {
            if(ctx.row_sizes.count(gid)) {
                sub_block_sizes[global_to_sub[gid]] = ctx.row_sizes.at(gid);
            }
        }
        
        // 3. Filter blocks and build adjacency
        std::vector<const BlockData*> relevant_blocks;
        for(const auto& bd : ctx.blocks) {
            if(global_to_sub.count(bd.global_row) && global_to_sub.count(bd.global_col)) {
                relevant_blocks.push_back(&bd);
                int sub_row = global_to_sub[bd.global_row];
                int sub_col = global_to_sub[bd.global_col];
                sub_adj[sub_row].push_back(sub_col);
            }
        }
        
        // 4. Construct Matrix
        DistGraph* sub_graph = new DistGraph(this->graph->comm == MPI_COMM_NULL ? MPI_COMM_NULL : MPI_COMM_SELF); // this make this method thread safe
        sub_graph->construct_serial(M, sub_block_sizes, sub_adj);
        
        BlockSpMat<T, Kernel> sub_mat(sub_graph);
        sub_mat.owns_graph = true;
        sub_mat.graph->enable_matrix_lifetime_management();
        
        for(const auto* bd : relevant_blocks) {
            int sub_row = global_to_sub[bd->global_row];
            int sub_col = global_to_sub[bd->global_col];
            
            // printf("Rank %d: Adding block (%d, %d)\n", rank, sub_row, sub_col);
            // fflush(stdout);
            
            sub_mat.add_block(sub_row, sub_col, bd->data.data(), bd->r_dim, bd->c_dim, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
        
        sub_mat.assemble();
        return sub_mat;
    }

    // Extract submatrix defined by global_indices
    BlockSpMat<T, Kernel> extract_submatrix(const std::vector<int>& global_indices) {
        auto ctx = detail::fetch_batched_block_payloads(*this, {global_indices});
        return construct_submatrix(global_indices, ctx);
    }

    // Extract multiple submatrices efficiently
    std::vector<BlockSpMat<T, Kernel>> extract_submatrix_batched(const std::vector<std::vector<int>>& batch_indices) {
        auto ctx = detail::fetch_batched_block_payloads(*this, batch_indices);
        std::vector<BlockSpMat<T, Kernel>> results;
        results.reserve(batch_indices.size());
        for(const auto& indices : batch_indices) {
            results.push_back(construct_submatrix(indices, ctx));
        }
        return results;
    }

    // Insert submatrix back (In-Place)
    void insert_submatrix(const BlockSpMat<T, Kernel>& submat, const std::vector<int>& global_indices) {
        // global_indices maps submat indices 0..M-1 to global indices
        if(submat.graph->owned_global_indices.size() != global_indices.size()) {
            throw std::runtime_error("insert_submatrix: global_indices size mismatch");
        }
        
        // Iterate over submat blocks
        int n_rows = submat.graph->adj_ptr.size() - 1;
        for(int i=0; i<n_rows; ++i) {
            int r_dim = submat.graph->block_sizes[i];
            int start = submat.graph->adj_ptr[i];
            int end = submat.graph->adj_ptr[i+1];
            
            int global_row = global_indices[i];
            
            for(int k=start; k<end; ++k) {
                int col = submat.graph->adj_ind[k];
                int c_dim = submat.graph->block_sizes[col];
                int global_col = global_indices[col];
                
                const T* data = submat.block_data(k);
                
                // Use add_block with INSERT mode. 
                // It handles local update and remote buffering.
                // Data is in ColMajor (internal storage of submat).
                this->add_block(global_row, global_col, data, r_dim, c_dim, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            }
        }
        
        // Flush remote updates
        this->assemble();
    }

    // Returns the local dense view with owned rows and local columns, including ghosts.
    std::vector<T> to_dense() const {
        int n_owned = graph->owned_global_indices.size();
        
        // Rows: Sum of sizes of owned blocks
        int my_rows = graph->block_offsets[n_owned];
        
        // Cols: Sum of sizes of ALL local blocks (owned + ghost)
        int my_cols = graph->block_offsets.back();
        
        std::vector<T> dense(my_rows * my_cols, T(0));
        
        // Fill
        for(int i=0; i<n_owned; ++i) {
            int r_dim = graph->block_sizes[i];
            int row_offset = graph->block_offsets[i]; // Local offset
            
            int start = graph->adj_ptr[i];
            int end = graph->adj_ptr[i+1];
            
            for(int k=start; k<end; ++k) {
                int col = graph->adj_ind[k];
                int c_dim = graph->block_sizes[col];
                int col_offset = graph->block_offsets[col];
                
                // const T* data = val.data() + blk_ptr[k];
                const T* data = block_data(k);
                
                // Copy block to dense (ColMajor block to RowMajor dense)
                for(int c=0; c<c_dim; ++c) {
                    for(int r=0; r<r_dim; ++r) {
                        // Dense index
                        int dr = row_offset + r;
                        int dc = col_offset + c;
                        if(dr < my_rows && dc < my_cols) {
                            dense[dr * my_cols + dc] = data[c * r_dim + r];
                        }
                    }
                }
            }
        }
        return dense;
    }

    // Update from dense (Row-Major)
    // Expects dense matrix of size (owned_rows) x (all_local_cols)
    void from_dense(const std::vector<T>& dense) {
        int n_owned = graph->owned_global_indices.size();
        int my_rows = graph->block_offsets[n_owned];
        int my_cols = graph->block_offsets.back();
        
        if(dense.size() != my_rows * my_cols) {
            throw std::runtime_error("from_dense: size mismatch");
        }
        
        for(int i=0; i<n_owned; ++i) {
            int r_dim = graph->block_sizes[i];
            int row_offset = graph->block_offsets[i];
            
            int start = graph->adj_ptr[i];
            int end = graph->adj_ptr[i+1];
            
            for(int k=start; k<end; ++k) {
                int col = graph->adj_ind[k];
                int c_dim = graph->block_sizes[col];
                int col_offset = graph->block_offsets[col];
                
                // T* data = val.data() + blk_ptr[k];
                T* data = mutable_block_data(k);
                
                // Copy dense to block (RowMajor dense to ColMajor block)
                for(int c=0; c<c_dim; ++c) {
                    for(int r=0; r<r_dim; ++r) {
                        int dr = row_offset + r;
                        int dc = col_offset + c;
                        if(dr < my_rows && dc < my_cols) {
                            data[c * r_dim + r] = dense[dr * my_cols + dc];
                        }
                    }
                }
            }
        }
    }
    // Calculate block density (global nnz blocks / total global blocks^2)
    double get_block_density() const {
        long long local_nnz = graph->adj_ind.size();
        long long global_nnz = 0;
        
        if (graph->size > 1) {
            MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_LONG_LONG, MPI_SUM, graph->comm);
        } else {
            global_nnz = local_nnz;
        }
        
        if (graph->block_displs.empty()) {
            return 0.0;
        }
        long long N = graph->block_displs.back();
        
        if (N == 0) return 0.0;
        
        double density = (double)global_nnz / (double)(N * N);
        return density;
    }

    // Diagnostics and test hooks.
    template <typename Fn>
    void for_each_shape_class(Fn&& fn) const {
        if (kind != MatrixKind::VBCSR) {
            return;
        }
        active_vbcsr_backend().for_each_shape_class(std::forward<Fn>(fn));
    }

    template <typename Fn>
    void for_each_shape_batch(Fn&& fn) const {
        if (kind != MatrixKind::VBCSR) {
            return;
        }
        active_vbcsr_backend().for_each_shape_batch(
            graph->adj_ptr,
            graph->adj_ind,
            std::forward<Fn>(fn));
    }

    template <typename Fn>
    void for_each_local_block(Fn&& fn) const {
        const int n_rows = block_row_count();
        const auto& structure_row_ptr = graph->adj_ptr;
        const auto& structure_col_ind = graph->adj_ind;
        for (int row = 0; row < n_rows; ++row) {
            const int row_dim = graph->block_sizes[row];
            for (int slot = structure_row_ptr[row]; slot < structure_row_ptr[row + 1]; ++slot) {
                const int col = structure_col_ind[slot];
                fn(ConstLocalBlockView{
                    slot,
                    row,
                    col,
                    row_dim,
                    graph->block_sizes[col],
                    block_size_elements(slot),
                    block_data(slot)});
            }
        }
    }

    template <typename Fn>
    void for_each_local_block(Fn&& fn) {
        const int n_rows = block_row_count();
        const auto& structure_row_ptr = graph->adj_ptr;
        const auto& structure_col_ind = graph->adj_ind;
        for (int row = 0; row < n_rows; ++row) {
            const int row_dim = graph->block_sizes[row];
            for (int slot = structure_row_ptr[row]; slot < structure_row_ptr[row + 1]; ++slot) {
                const int col = structure_col_ind[slot];
                fn(LocalBlockView{
                    slot,
                    row,
                    col,
                    row_dim,
                    graph->block_sizes[col],
                    block_size_elements(slot),
                    mutable_block_data(slot)});
            }
        }
    }

private:
    // Backend state, structure binding, and executor support helpers.
    struct PendingBlock {
        int rows = 0;
        int cols = 0;
        int mode_code = 0;
        std::vector<T> data;
    };

    using RemoteOwnerBlocks = std::map<int, std::map<std::pair<int, int>, PendingBlock>>;
    using RemoteThreadBuffers = std::vector<RemoteOwnerBlocks>;

    struct RemoteAssemblyRegistry {
        std::mutex mutex;
        std::map<const BlockSpMat*, RemoteThreadBuffers> buffers;
    };

    static double get_sq_norm(const T& v);
    std::vector<double> compute_block_norms() const;
    static int max_omp_threads();
    static RemoteAssemblyRegistry& remote_assembly_registry();
    void release_graph_reference() const;
    void prepare_graph_for_shared_use() const;
    void require_assembled_for_state_copy(const char* op_name) const;

    BlockSpMat(DistGraph* g, MatrixKind matrix_kind, bool owns_graph_flag, ConstructionToken);

    VBCSRBackendStorage& active_vbcsr_backend();
    const VBCSRBackendStorage& active_vbcsr_backend() const;
    CSRBackendStorage& active_csr_backend();
    const CSRBackendStorage& active_csr_backend() const;
    BSRBackendStorage& active_bsr_backend();
    const BSRBackendStorage& active_bsr_backend() const;
    RemoteThreadBuffers& remote_assembly_buffers() const;
    static bool has_pending_remote_assembly(const BlockSpMat* matrix);
    static void transfer_remote_assembly_state(const BlockSpMat* from, const BlockSpMat* to);
    static void clear_remote_assembly_state(const BlockSpMat* matrix);
    static DistGraph* require_live_graph(DistGraph* graph, const char* op_name);
    static const DistGraph* require_live_graph(const DistGraph* graph, const char* op_name);
    static uint32_t default_page_size_for(MatrixKind matrix_kind, const DistGraph* graph);
    static uint32_t normalize_page_size(
        MatrixKind matrix_kind,
        const DistGraph* graph,
        uint32_t page_size);
    void rebuild_backend_for_page_size(const char* op_name);

    static MatrixKind detect_matrix_kind(const DistGraph* g);
    static CommittedBackendStorage build_backend_for_structure(
        MatrixKind matrix_kind,
        DistGraph* graph,
        uint32_t page_size);

    void attach_backend(VBCSRBackendStorage backend);
    void attach_backend(CSRBackendStorage backend);
    void attach_backend(BSRBackendStorage backend);
    void attach_backend(CommittedBackendStorage backend);
    static void ensure_same_backend_family(const BlockSpMat& lhs, const BlockSpMat& rhs, const char* op_name);
    static void ensure_csr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs);
    static void ensure_bsr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs);
    static void ensure_vbcsr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs);
    bool has_same_logical_structure(const BlockSpMat& other) const;

    static void write_transposed_conjugate_values(T* dest, const T* src, int src_rows, int src_cols);
};

template <typename T, typename Kernel>
double BlockSpMat<T, Kernel>::get_sq_norm(const T& v) {
    if constexpr (std::is_same<T, std::complex<double>>::value ||
                  std::is_same<T, std::complex<float>>::value) {
        return std::norm(v);
    } else {
        return v * v;
    }
}

template <typename T, typename Kernel>
std::vector<double> BlockSpMat<T, Kernel>::compute_block_norms() const {
    int nnz = static_cast<int>(graph->adj_ind.size());
    std::vector<double> norms(nnz);

    #pragma omp parallel for
    for (int i = 0; i < nnz; ++i) {
        double sum = 0.0;
        const T* data = block_data(i);
        const size_t size = block_size_elements(i);
        for (size_t k = 0; k < size; ++k) {
            sum += get_sq_norm(data[k]);
        }
        norms[i] = std::sqrt(sum);
    }
    return norms;
}

template <typename T, typename Kernel>
int BlockSpMat<T, Kernel>::max_omp_threads() {
    int max_threads = 1;
    #ifdef _OPENMP
    max_threads = omp_get_max_threads();
    #endif
    return max_threads;
}

template <typename T, typename Kernel>
typename BlockSpMat<T, Kernel>::RemoteAssemblyRegistry& BlockSpMat<T, Kernel>::remote_assembly_registry() {
    static RemoteAssemblyRegistry instance;
    return instance;
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::release_graph_reference() const {
    if (graph == nullptr) {
        return;
    }
    if (owns_graph) {
        graph->enable_matrix_lifetime_management();
    }
    if (graph->release_matrix_reference()) {
        delete graph;
    }
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::prepare_graph_for_shared_use() const {
    DistGraph* live_graph = require_live_graph(graph, "prepare_graph_for_shared_use");
    if (owns_graph && !live_graph->has_managed_matrix_lifetime()) {
        // Promote a singly owned graph before another matrix starts sharing it.
        live_graph->enable_matrix_lifetime_management();
    }
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::require_assembled_for_state_copy(const char* op_name) const {
    if (has_pending_remote_assembly(this)) {
        throw std::runtime_error(
            std::string(op_name) +
            " requires an assembled matrix; pending remote add_block updates still exist");
    }
}

template <typename T, typename Kernel>
BlockSpMat<T, Kernel>::BlockSpMat(
    DistGraph* g,
    MatrixKind matrix_kind,
    bool owns_graph,
    ConstructionToken)
    : kind(matrix_kind),
      backend_handle_(std::monostate{}),
      configured_page_size_(default_page_size_for(matrix_kind, require_live_graph(g, "BlockSpMat"))),
      graph(require_live_graph(g, "BlockSpMat")),
      owns_graph(owns_graph) {
    graph->acquire_matrix_reference();
    if (owns_graph) {
        graph->enable_matrix_lifetime_management();
    }
}

template <typename T, typename Kernel>
typename BlockSpMat<T, Kernel>::VBCSRBackendStorage& BlockSpMat<T, Kernel>::active_vbcsr_backend() {
    return detail::require_vbcsr_backend<T, Kernel>(backend_handle_);
}

template <typename T, typename Kernel>
const typename BlockSpMat<T, Kernel>::VBCSRBackendStorage& BlockSpMat<T, Kernel>::active_vbcsr_backend() const {
    return detail::require_vbcsr_backend<T, Kernel>(backend_handle_);
}

template <typename T, typename Kernel>
typename BlockSpMat<T, Kernel>::CSRBackendStorage& BlockSpMat<T, Kernel>::active_csr_backend() {
    return detail::require_csr_backend<T, Kernel>(backend_handle_);
}

template <typename T, typename Kernel>
const typename BlockSpMat<T, Kernel>::CSRBackendStorage& BlockSpMat<T, Kernel>::active_csr_backend() const {
    return detail::require_csr_backend<T, Kernel>(backend_handle_);
}

template <typename T, typename Kernel>
typename BlockSpMat<T, Kernel>::BSRBackendStorage& BlockSpMat<T, Kernel>::active_bsr_backend() {
    return detail::require_bsr_backend<T, Kernel>(backend_handle_);
}

template <typename T, typename Kernel>
const typename BlockSpMat<T, Kernel>::BSRBackendStorage& BlockSpMat<T, Kernel>::active_bsr_backend() const {
    return detail::require_bsr_backend<T, Kernel>(backend_handle_);
}

template <typename T, typename Kernel>
typename BlockSpMat<T, Kernel>::RemoteThreadBuffers& BlockSpMat<T, Kernel>::remote_assembly_buffers() const {
    auto& reg = remote_assembly_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto& buffers = reg.buffers[this];
    if (buffers.empty()) {
        buffers.resize(static_cast<size_t>(max_omp_threads()));
    }
    return buffers;
}

template <typename T, typename Kernel>
bool BlockSpMat<T, Kernel>::has_pending_remote_assembly(const BlockSpMat* matrix) {
    auto& reg = remote_assembly_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.buffers.find(matrix);
    if (it == reg.buffers.end()) {
        return false;
    }
    for (const auto& thread_buffers : it->second) {
        for (const auto& owner_entry : thread_buffers) {
            if (!owner_entry.second.empty()) {
                return true;
            }
        }
    }
    return false;
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::transfer_remote_assembly_state(const BlockSpMat* from, const BlockSpMat* to) {
    auto& reg = remote_assembly_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.buffers.find(from);
    if (it == reg.buffers.end()) {
        return;
    }
    reg.buffers[to] = std::move(it->second);
    reg.buffers.erase(it);
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::clear_remote_assembly_state(const BlockSpMat* matrix) {
    auto& reg = remote_assembly_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.buffers.erase(matrix);
}

template <typename T, typename Kernel>
DistGraph* BlockSpMat<T, Kernel>::require_live_graph(DistGraph* graph, const char* op_name) {
    if (graph == nullptr) {
        throw std::invalid_argument(std::string(op_name) + " requires a live DistGraph");
    }
    return graph;
}

template <typename T, typename Kernel>
const DistGraph* BlockSpMat<T, Kernel>::require_live_graph(const DistGraph* graph, const char* op_name) {
    if (graph == nullptr) {
        throw std::invalid_argument(std::string(op_name) + " requires a live DistGraph");
    }
    return graph;
}

template <typename T, typename Kernel>
uint32_t BlockSpMat<T, Kernel>::default_page_size_for(
    MatrixKind matrix_kind,
    const DistGraph* graph) {
    graph = require_live_graph(graph, "default_page_size_for");
    switch (matrix_kind) {
    case MatrixKind::CSR:
        return CSRBackendStorage::max_page_size();
    case MatrixKind::BSR: {
        int block_size = 0;
        for (int dim : graph->block_sizes) {
            if (dim <= 1) {
                throw std::runtime_error("BSR backend requires uniform block sizes greater than 1");
            }
            if (block_size == 0) {
                block_size = dim;
            } else if (dim != block_size) {
                throw std::runtime_error("BSR backend requires a uniform block size");
            }
        }
        return BSRBackendStorage::max_blocks_per_page(block_size);
    }
    case MatrixKind::VBCSR:
        return VBCSRBackendStorage::hard_safe_blocks_per_page();
    }
    throw std::runtime_error("Unsupported matrix kind in default_page_size_for");
}

template <typename T, typename Kernel>
uint32_t BlockSpMat<T, Kernel>::normalize_page_size(
    MatrixKind matrix_kind,
    const DistGraph* graph,
    uint32_t page_size) {
    graph = require_live_graph(graph, "normalize_page_size");
    switch (matrix_kind) {
    case MatrixKind::CSR:
        return CSRBackendStorage::normalize_page_size(page_size);
    case MatrixKind::BSR:
        if (page_size == 0) {
            throw std::runtime_error("bsr page size must be positive");
        }
        {
            int block_size = 0;
            for (int dim : graph->block_sizes) {
                if (dim <= 1) {
                    throw std::runtime_error("BSR backend requires uniform block sizes greater than 1");
                }
                if (block_size == 0) {
                    block_size = dim;
                } else if (dim != block_size) {
                    throw std::runtime_error("BSR backend requires a uniform block size");
                }
            }
            return BSRBackendStorage::normalize_blocks_per_page(page_size, block_size);
        }
    case MatrixKind::VBCSR:
        return VBCSRBackendStorage::normalize_blocks_per_page(page_size);
    }
    throw std::runtime_error("Unsupported matrix kind in normalize_page_size");
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::rebuild_backend_for_page_size(const char* op_name) {
    require_assembled_for_state_copy(op_name);
    const bool had_norms = norms_valid;
    std::vector<double> saved_norms = had_norms ? block_norms : std::vector<double>{};

    BlockSpMat rebuilt(graph, kind, false, ConstructionToken{});
    rebuilt.configured_page_size_ = configured_page_size_;
    rebuilt.attach_backend(build_backend_for_structure(kind, graph, configured_page_size_));
    rebuilt.copy_from(*this);

    attach_backend(std::move(rebuilt.backend_handle_));
    if (had_norms) {
        block_norms = std::move(saved_norms);
        norms_valid = true;
    }
}

template <typename T, typename Kernel>
MatrixKind BlockSpMat<T, Kernel>::detect_matrix_kind(const DistGraph* g) {
    g = require_live_graph(g, "detect_matrix_kind");

    const int n_owned = static_cast<int>(g->owned_global_indices.size());
    int local_min = std::numeric_limits<int>::max();
    int local_max = 0;
    for (int i = 0; i < n_owned; ++i) {
        local_min = std::min(local_min, g->block_sizes[i]);
        local_max = std::max(local_max, g->block_sizes[i]);
    }

    int global_min = local_min;
    int global_max = local_max;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized && g->comm != MPI_COMM_NULL && g->size > 1) {
        MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, g->comm);
        MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, g->comm);
    }

    if (global_max <= 1) {
        return MatrixKind::CSR;
    }
    if (global_min == global_max) {
        return MatrixKind::BSR;
    }
    return MatrixKind::VBCSR;
}

template <typename T, typename Kernel>
typename BlockSpMat<T, Kernel>::CommittedBackendStorage BlockSpMat<T, Kernel>::build_backend_for_structure(
    MatrixKind matrix_kind,
    DistGraph* graph,
    uint32_t page_size) {
    graph = require_live_graph(graph, "build_backend_for_structure");
    const uint32_t normalized = normalize_page_size(matrix_kind, graph, page_size);
    switch (matrix_kind) {
    case MatrixKind::CSR: {
        CSRBackendStorage backend;
        backend.initialize_structure(graph->adj_ind.size(), normalized);
        return detail::make_csr_backend_handle<T, Kernel>(std::move(backend));
    }
    case MatrixKind::BSR: {
        int block_size = 0;
        for (int dim : graph->block_sizes) {
            if (dim <= 1) {
                throw std::runtime_error("BSR backend requires uniform block sizes greater than 1");
            }
            if (block_size == 0) {
                block_size = dim;
                continue;
            }
            if (dim != block_size) {
                throw std::runtime_error("BSR backend requires a uniform block size");
            }
        }

        BSRBackendStorage backend;
        backend.initialize_structure(graph->adj_ind.size(), block_size, normalized);
        return detail::make_bsr_backend_handle<T, Kernel>(std::move(backend));
    }
    case MatrixKind::VBCSR: {
        VBCSRBackendStorage backend(normalized);
        const size_t nnz = graph->adj_ind.size();
        backend.initialize_graph_block_handles(nnz);

        std::map<std::pair<int, int>, size_t> shape_counts;
        const int n_rows = static_cast<int>(graph->owned_global_indices.size());
        for (int row = 0; row < n_rows; ++row) {
            const int row_dim = graph->block_sizes[row];
            for (int graph_block_index = graph->adj_ptr[row];
                 graph_block_index < graph->adj_ptr[row + 1];
                 ++graph_block_index) {
                const int col = graph->adj_ind[graph_block_index];
                const int col_dim = graph->block_sizes[col];
                ++shape_counts[std::make_pair(row_dim, col_dim)];
            }
        }

        for (const auto& [shape, count] : shape_counts) {
            backend.ensure_shape(shape.first, shape.second, count);
        }

        for (int row = 0; row < n_rows; ++row) {
            const int row_dim = graph->block_sizes[row];
            for (int graph_block_index = graph->adj_ptr[row];
                 graph_block_index < graph->adj_ptr[row + 1];
                 ++graph_block_index) {
                const int col = graph->adj_ind[graph_block_index];
                const int col_dim = graph->block_sizes[col];
                const int shape_id =
                    backend.ensure_shape(row_dim, col_dim, shape_counts[std::make_pair(row_dim, col_dim)]);
                backend.set_graph_block_handle(
                    graph_block_index,
                    backend.append_block_for_shape(shape_id, graph_block_index));
            }
        }

        return detail::make_vbcsr_backend_handle<T, Kernel>(std::move(backend));
    }
    }
    throw std::runtime_error("Unsupported matrix kind in build_backend_for_structure");
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::attach_backend(VBCSRBackendStorage backend) {
    configured_page_size_ = backend.configured_blocks_per_page();
    backend_handle_ = detail::make_vbcsr_backend_handle<T, Kernel>(std::move(backend));
    block_norms.clear();
    norms_valid = false;
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::attach_backend(CSRBackendStorage backend) {
    configured_page_size_ = backend.configured_page_size();
    backend_handle_ = detail::make_csr_backend_handle<T, Kernel>(std::move(backend));
    block_norms.clear();
    norms_valid = false;
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::attach_backend(BSRBackendStorage backend) {
    configured_page_size_ = backend.configured_blocks_per_page();
    backend_handle_ = detail::make_bsr_backend_handle<T, Kernel>(std::move(backend));
    block_norms.clear();
    norms_valid = false;
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::attach_backend(CommittedBackendStorage backend) {
    std::visit(
        [this](auto&& committed_backend) {
            attach_backend(std::move(committed_backend));
        },
        std::move(backend));
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::ensure_same_backend_family(
    const BlockSpMat& lhs,
    const BlockSpMat& rhs,
    const char* op_name) {
    if (lhs.kind != rhs.kind) {
        throw std::runtime_error(
            std::string(op_name) +
            " requires matrices from the same backend family, got " +
            lhs.matrix_kind_string() + " and " + rhs.matrix_kind_string());
    }
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::ensure_csr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs) {
    if (lhs.graph->owned_global_indices != rhs.graph->owned_global_indices) {
        throw std::runtime_error("CSR binary operation requires matching owned row distribution");
    }
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::ensure_bsr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs) {
    if (lhs.graph->owned_global_indices != rhs.graph->owned_global_indices) {
        throw std::runtime_error("BSR binary operation requires matching owned row distribution");
    }
    const int lhs_block_size = detail::require_bsr_backend<T, Kernel>(lhs.backend_handle_).block_size;
    const int rhs_block_size = detail::require_bsr_backend<T, Kernel>(rhs.backend_handle_).block_size;
    if (lhs_block_size != rhs_block_size) {
        throw std::runtime_error("BSR binary operation requires matching uniform block sizes");
    }
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::ensure_vbcsr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs) {
    if (lhs.graph->owned_global_indices != rhs.graph->owned_global_indices) {
        throw std::runtime_error("VBCSR binary operation requires matching owned row distribution");
    }
    const size_t owned_rows = lhs.graph->owned_global_indices.size();
    if (owned_rows != rhs.graph->owned_global_indices.size()) {
        throw std::runtime_error("VBCSR binary operation requires matching owned row count");
    }
    for (size_t row = 0; row < owned_rows; ++row) {
        if (lhs.graph->block_sizes[row] != rhs.graph->block_sizes[row]) {
            throw std::runtime_error("VBCSR binary operation requires matching owned row block sizes");
        }
    }
}

template <typename T, typename Kernel>
bool BlockSpMat<T, Kernel>::has_same_logical_structure(const BlockSpMat& other) const {
    if (kind != other.kind) {
        return false;
    }

    if (graph == nullptr || other.graph == nullptr) {
        return false;
    }

    return graph->owned_global_indices == other.graph->owned_global_indices &&
           graph->block_sizes == other.graph->block_sizes &&
           graph->adj_ptr == other.graph->adj_ptr &&
           graph->adj_ind == other.graph->adj_ind;
}

template <typename T, typename Kernel>
void BlockSpMat<T, Kernel>::write_transposed_conjugate_values(
    T* dest,
    const T* src,
    int src_rows,
    int src_cols) {
    for (int dest_col = 0; dest_col < src_rows; ++dest_col) {
        for (int dest_row = 0; dest_row < src_cols; ++dest_row) {
            dest[dest_col * src_cols + dest_row] =
                ConjHelper<T>::apply(src[dest_row * src_rows + dest_col]);
        }
    }
}

template <typename T, typename Kernel>
bool BlockSpMat<T, Kernel>::update_local_block(
    int local_row,
    int local_col,
    const T* data,
    int rows,
    int cols,
    AssemblyMode mode,
    MatrixLayout layout) {
    int start = graph->adj_ptr[local_row];
    int end = graph->adj_ptr[local_row + 1];

    for (int k = start; k < end; ++k) {
        if (graph->adj_ind[k] == local_col) {
            T* target = mutable_block_data(k);

            int r_dim = graph->block_sizes[local_row];
            int c_dim = graph->block_sizes[local_col];

            if (r_dim != rows || c_dim != cols) {
                std::stringstream ss;
                ss << "\n Dimension mismatch in update_local_block (DEBUG): "
                   << "Row: " << local_row << " (Expected: " << r_dim << ", Got: " << rows << ") \n"
                   << "Col: " << local_col << " (Expected: " << c_dim << ", Got: " << cols << ") \n";
                throw std::runtime_error(ss.str());
            }

            if (mode == AssemblyMode::INSERT) {
                if (layout == MatrixLayout::ColMajor) {
                    std::memcpy(target, data, r_dim * c_dim * sizeof(T));
                } else {
                    for (int i = 0; i < r_dim; ++i) {
                        for (int j = 0; j < c_dim; ++j) {
                            target[j * r_dim + i] = data[i * c_dim + j];
                        }
                    }
                }
            } else {
                if (layout == MatrixLayout::ColMajor) {
                    for (int i = 0; i < r_dim * c_dim; ++i) {
                        target[i] += data[i];
                    }
                } else {
                    for (int i = 0; i < r_dim; ++i) {
                        for (int j = 0; j < c_dim; ++j) {
                            target[j * r_dim + i] += data[i * c_dim + j];
                        }
                    }
                }
            }
            return true;
        }
    }
    return false;
}

} // namespace vbcsr

#endif
