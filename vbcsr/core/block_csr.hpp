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

// Canonical within-block element order for block STORAGE, pending assembly
// buffers, and every MPI payload that carries raw block values
// (doc/row_major_migration_plan.md §2.1). Element (i, j) of an r×c block sits
// at block[i * c + j]. Every staging/transport call site names this constant
// instead of a literal enum so the invariant stays single-sourced and
// greppable.
inline constexpr MatrixLayout kCanonicalBlockLayout = MatrixLayout::RowMajor;

// Block-redistribution reduction (doc/design/35). Copy = each source block is
// written to every target owner of its row (one->one, or one->many = broadcast /
// send-down). Sum = contributions from several source ranks that hold the same
// (i,j) partial are accumulated at the target owner (reduce-up).
enum class RedistOp {
    Copy,
    Sum
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
#include "detail/kernels/dense_kernels.hpp"
#include "detail/backend/bsr_backend.hpp"
#include "detail/backend/csr_backend.hpp"
#include "detail/backend/vbcsr_backend.hpp"
#include "detail/kernels/bsr_apply.hpp"
#include "detail/kernels/csr_apply.hpp"
#include "detail/kernels/vbcsr_apply.hpp"
#include "detail/ops/axpby.hpp"
#include "detail/ops/spmm/bsr.hpp"
#include "detail/ops/spmm/csr.hpp"
#include "detail/ops/spmm/vbcsr.hpp"
#include "detail/ops/transpose.hpp"
#include "detail/distributed/block_payload_exchange.hpp"
#include "detail/distributed/mpi_utils.hpp"
#include <algorithm>
#include <vector>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <complex>
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <map>
#include <unordered_map>
#include <cstring>
#include <functional>
#include <mutex>
#include <utility>
#include <random>

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

template <typename T>
class BlockSpMat {
private:
    // Friend access for backend-specialized executors and builders.
    template <typename>
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
    template <typename U>
    friend void graph_matrix_function(
        BlockSpMat<U>&,
        BlockSpMat<U>*,
        std::function<U(double)>,
        bool);

    MatrixKind kind = MatrixKind::CSR;
    using VBCSRBackendStorage = detail::VBCSRMatrixBackend<T>;
    using CSRBackendStorage = detail::CSRMatrixBackend<T>;
    using BSRBackendStorage = detail::BSRMatrixBackend<T>;
    using BackendHandle = std::variant<
        std::monostate,
        CSRBackendStorage,
        BSRBackendStorage,
        VBCSRBackendStorage>;

    struct ConstructionToken {};

    BackendHandle backend_handle_;
    uint32_t configured_page_size_ = 0;

public:
    // Public facade state and view types.
    DistGraph* graph;
    using value_type = T;

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

    std::string vendor_backend_name() const {
        switch (kind) {
        case MatrixKind::CSR:
            return active_csr_backend().vendor_backend_name();
        case MatrixKind::BSR:
            return active_bsr_backend().vendor_backend_name();
        case MatrixKind::VBCSR:
            return "none";
        }
        return "unknown";
    }

    uint64_t vendor_launch_count() const {
        switch (kind) {
        case MatrixKind::CSR:
            return active_csr_backend().get_vendor_launch_count();
        case MatrixKind::BSR:
            return active_bsr_backend().get_vendor_launch_count();
        case MatrixKind::VBCSR:
            return 0;
        }
        return 0;
    }

    void reset_vendor_launch_count() const {
        switch (kind) {
        case MatrixKind::CSR:
            active_csr_backend().reset_vendor_launch_count();
            return;
        case MatrixKind::BSR:
            active_bsr_backend().reset_vendor_launch_count();
            return;
        case MatrixKind::VBCSR:
            return;
        }
    }

    bool has_contiguous_layout() const {
        return true;
    }

    void pack_contiguous() {
        require_assembled_for_state_copy("pack_contiguous");
    }

    double maxabs() const {
        require_live_graph(graph, "maxabs");

        const int nnz = static_cast<int>(local_block_nnz());
        double local_max = 0.0;

        #pragma omp parallel for reduction(max:local_max)
        for (int i = 0; i < nnz; ++i) {
            const T* data = block_data(i);
            const size_t size = block_size_elements(i);
            for (size_t k = 0; k < size; ++k) {
                using std::abs;
                local_max = std::max(local_max, static_cast<double>(abs(data[k])));
            }
        }

        double global_max = local_max;
        if (graph->size > 1) {
            MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
        }
        return global_max;
    }

    double frobenius_norm() const {
        require_live_graph(graph, "frobenius_norm");

        const int nnz = static_cast<int>(local_block_nnz());
        double local_sum = 0.0;

        #pragma omp parallel for reduction(+:local_sum)
        for (int i = 0; i < nnz; ++i) {
            const T* data = block_data(i);
            const size_t size = block_size_elements(i);
            for (size_t k = 0; k < size; ++k) {
                local_sum += get_sq_norm(data[k]);
            }
        }

        double global_sum = local_sum;
        if (graph->size > 1) {
            MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, graph->comm);
        }
        return std::sqrt(global_sum);
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
    // Every BlockSpMat registers with its graph, even when the graph remains
    // user-owned. Matrix-owned result graphs are promoted to delete-on-last
    // release; externally owned graphs are only reference-counted.
    BlockSpMat(DistGraph* g)
        : BlockSpMat(
              require_live_graph(g, "BlockSpMat"),
              detect_matrix_kind(require_live_graph(g, "BlockSpMat")),
              false,
              ConstructionToken{}) {
        allocate_from_graph();
    }

    ~BlockSpMat() {
        // Remote assembly buffers are keyed by this matrix address, so they
        // must be cleared before the address can be reused by another object.
        clear_remote_assembly_state(this);
        release_graph_reference();
    }

    // Move constructor. Pending remote assembly state follows the matrix
    // address that now owns the backend and graph reference.
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

    // Move assignment first releases this object's current graph/reference and
    // address-keyed assembly buffers, then adopts the moved-from state.
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
    BlockSpMat<T> duplicate(bool independent_graph = true) const {
        require_assembled_for_state_copy("duplicate");
        DistGraph* new_graph = graph;
        bool new_owns_graph = false;
        if (independent_graph && graph) {
            new_graph = graph->duplicate();
            new_owns_graph = true;
        } else {
            // duplicate(false) shares this graph with another matrix. If this
            // matrix had owned the graph outright, promote it to managed
            // delete-on-last-release before exposing the shared pointer.
            prepare_graph_for_shared_use();
        }
        BlockSpMat<T> new_mat(new_graph);
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
    void add_block(int global_row, int global_col, const T* data, int rows, int cols, AssemblyMode mode = AssemblyMode::ADD, MatrixLayout layout = kCanonicalBlockLayout) {
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
            
            // Pending blocks are stored in kCanonicalBlockLayout (row-major,
            // the canonical format for transport).

            if (mode == AssemblyMode::INSERT) {
                // Overwrite
                pb.mode_code = static_cast<int>(AssemblyMode::INSERT);
                if (layout == kCanonicalBlockLayout) {
                    std::memcpy(pb.data.data(), data, rows * cols * sizeof(T));
                } else {
                    // Transpose ColMajor -> canonical RowMajor
                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            pb.data[i * cols + j] = data[j * rows + i];
                        }
                    }
                }
            } else {
                // ADD
                // Accumulate
                if (layout == kCanonicalBlockLayout) {
                    for (size_t i = 0; i < pb.data.size(); ++i) {
                        pb.data[i] += data[i];
                    }
                } else {
                    // Transpose add
                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            pb.data[i * cols + j] += data[j * rows + i];
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

            if (layout == kCanonicalBlockLayout) {
                std::memcpy(pb.data.data(), data, rows * cols * sizeof(T));
            } else {
                // Transpose ColMajor -> canonical RowMajor
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        pb.data[i * cols + j] = data[j * rows + i];
                    }
                }
            }
            blocks_map[key] = std::move(pb);
        }
    }

    // Finalize assembly by exchanging remote blocks
    // Redistribute to a different partition of the SAME global block structure
    // (doc/design/35). Increment 1: same communicator (graph->comm == target->comm).
    // Enumerate this rank's owned blocks and re-add them into a matrix built on
    // ``target``; ``assemble`` routes each block to its target owner. ``mode`` =
    // INSERT (a one-to-one repartition) or ADD (accumulate when several source ranks
    // contribute the same (i,j) block — a same-comm reduce). Cross-comm / broadcast /
    // multi-owner variants follow in later increments.
    BlockSpMat redistribute(DistGraph* target, AssemblyMode mode = AssemblyMode::INSERT) const {
        BlockSpMat result(target);
        const DistGraph* g = graph;
        const int n_owned = static_cast<int>(g->owned_global_indices.size());
        auto l2g = [&](int lb) -> int {
            return lb < n_owned ? g->owned_global_indices[lb]
                                : g->ghost_global_indices[lb - n_owned];
        };
        // ``target`` must be a fully-constructed graph (ghost block sizes backfilled,
        // as any assembled operator graph is) — add_block needs the column dim.
        for (int i = 0; i < n_owned; ++i) {
            const int gi = g->owned_global_indices[i];
            const int r_dim = g->block_sizes[i];
            for (int k = g->adj_ptr[i]; k < g->adj_ptr[i + 1]; ++k) {
                const int lcol = g->adj_ind[k];
                const int gj = l2g(lcol);
                const int c_dim = g->block_sizes[lcol];
                result.add_block(gi, gj, block_data(k), r_dim, c_dim, mode, kCanonicalBlockLayout);
            }
        }
        result.assemble();
        return result;
    }

    // Cross-comm redistribute (doc/design/35, increment 2). Move blocks from this
    // matrix's partition (``graph->comm``, e.g. comm_world / L1) to ``target``'s
    // partition (``target->comm``, e.g. pool_comm / L3), transporting on
    // ``common_comm`` (a comm spanning both rank sets, e.g. comm_world). The result
    // is built on ``target``, so it genuinely lives on ``target->comm``.
    //
    // The owner map is *derived*: every common_comm rank announces the global rows it
    // owns in the TARGET layout (one Allgatherv), giving global-row -> {owner cc-ranks}.
    // A row owned by several ranks (pool replication) makes Copy fan out (= send-down);
    // partials held by several source ranks make Sum accumulate at the unique target
    // owner (= reduce-up). ``target`` must be assembled (ghost block sizes backfilled).
    BlockSpMat redistribute(DistGraph* target, RedistOp op, MPI_Comm common_comm) const {
        int cc_size;
        MPI_Comm_size(common_comm, &cc_size);

        // 1. Derive global-row -> owner cc-ranks in the target layout.
        const int my_n = static_cast<int>(target->owned_global_indices.size());
        std::vector<int> all_n(cc_size);
        MPI_Allgather(&my_n, 1, MPI_INT, all_n.data(), 1, MPI_INT, common_comm);
        std::vector<int> displ(cc_size + 1, 0);
        for (int i = 0; i < cc_size; ++i) displ[i + 1] = displ[i] + all_n[i];
        std::vector<int> all_owned(displ[cc_size]);
        MPI_Allgatherv(target->owned_global_indices.data(), my_n, MPI_INT,
                       all_owned.data(), all_n.data(), displ.data(), MPI_INT, common_comm);
        std::unordered_map<int, std::vector<int>> target_owners;
        for (int r = 0; r < cc_size; ++r) {
            for (int k = displ[r]; k < displ[r + 1]; ++k) {
                target_owners[all_owned[k]].push_back(r);
            }
        }

        // 2. Pack this rank's owned blocks, one copy per target owner of the row.
        const DistGraph* g = graph;
        const int n_owned = static_cast<int>(g->owned_global_indices.size());
        auto l2g = [&](int lb) -> int {
            return lb < n_owned ? g->owned_global_indices[lb]
                                : g->ghost_global_indices[lb - n_owned];
        };
        std::vector<size_t> send_counts(cc_size, 0);
        for (int i = 0; i < n_owned; ++i) {
            const int gi = g->owned_global_indices[i];
            auto it = target_owners.find(gi);
            if (it == target_owners.end()) continue;
            const int r_dim = g->block_sizes[i];
            for (int k = g->adj_ptr[i]; k < g->adj_ptr[i + 1]; ++k) {
                const int c_dim = g->block_sizes[g->adj_ind[k]];
                const size_t blk_bytes = 4 * sizeof(int)
                                       + static_cast<size_t>(r_dim) * c_dim * sizeof(T);
                for (int dest : it->second) send_counts[dest] += blk_bytes;
            }
        }
        std::vector<size_t> sdispls(cc_size + 1, 0);
        for (int i = 0; i < cc_size; ++i) sdispls[i + 1] = sdispls[i] + send_counts[i];
        std::vector<char> send_blob(sdispls[cc_size]);
        std::vector<size_t> off = sdispls;
        for (int i = 0; i < n_owned; ++i) {
            const int gi = g->owned_global_indices[i];
            auto it = target_owners.find(gi);
            if (it == target_owners.end()) continue;
            const int r_dim = g->block_sizes[i];
            for (int k = g->adj_ptr[i]; k < g->adj_ptr[i + 1]; ++k) {
                const int lcol = g->adj_ind[k];
                const int gj = l2g(lcol);
                const int c_dim = g->block_sizes[lcol];
                const T* bd = block_data(k);
                const size_t data_bytes = static_cast<size_t>(r_dim) * c_dim * sizeof(T);
                for (int dest : it->second) {
                    char* ptr = send_blob.data() + off[dest];
                    std::memcpy(ptr, &gi, sizeof(int));    ptr += sizeof(int);
                    std::memcpy(ptr, &gj, sizeof(int));    ptr += sizeof(int);
                    std::memcpy(ptr, &r_dim, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(ptr, &c_dim, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(ptr, bd, data_bytes);      ptr += data_bytes;
                    off[dest] = ptr - send_blob.data();
                }
            }
        }

        // 3. Exchange counts + blocks on the common comm.
        std::vector<size_t> recv_counts(cc_size);
        MPI_Alltoall(send_counts.data(), sizeof(size_t), MPI_BYTE,
                     recv_counts.data(), sizeof(size_t), MPI_BYTE, common_comm);
        std::vector<size_t> rdispls(cc_size + 1, 0);
        for (int i = 0; i < cc_size; ++i) rdispls[i + 1] = rdispls[i] + recv_counts[i];
        std::vector<char> recv_blob(rdispls[cc_size]);
        safe_alltoallv(send_blob.data(), send_counts, sdispls, MPI_BYTE,
                       recv_blob.data(), recv_counts, rdispls, MPI_BYTE, common_comm);

        // 4. Place received blocks on the target. INSERT for Copy; for Sum, zero
        //    first and ADD so several senders' partials accumulate at the owner.
        BlockSpMat result(target);
        const AssemblyMode amode = (op == RedistOp::Sum) ? AssemblyMode::ADD
                                                         : AssemblyMode::INSERT;
        if (op == RedistOp::Sum) result.fill(T(0));
        const char* ptr = recv_blob.data();
        const char* rend = recv_blob.data() + recv_blob.size();
        while (ptr < rend) {
            int gi, gj, r_dim, c_dim;
            std::memcpy(&gi, ptr, sizeof(int));    ptr += sizeof(int);
            std::memcpy(&gj, ptr, sizeof(int));    ptr += sizeof(int);
            std::memcpy(&r_dim, ptr, sizeof(int)); ptr += sizeof(int);
            std::memcpy(&c_dim, ptr, sizeof(int)); ptr += sizeof(int);
            result.add_block(gi, gj, reinterpret_cast<const T*>(ptr), r_dim, c_dim,
                             amode, kCanonicalBlockLayout);
            ptr += static_cast<size_t>(r_dim) * c_dim * sizeof(T);
        }
        result.assemble();
        return result;
    }

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

                if (l_col == -1 || !update_local_block(l_row, l_col, (const T*)ptr, rows, cols, mode, kCanonicalBlockLayout)) {
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
        MatrixLayout layout = kCanonicalBlockLayout);

    // Matrix-Vector Multiplication
    void mult(DistVector<T>& x, DistVector<T>& y) {
        if (kind == MatrixKind::CSR) {
            detail::csr_mult(graph, active_csr_backend(), x, y);
            return;
        }
        if (kind == MatrixKind::BSR) {
            detail::bsr_mult(graph, active_bsr_backend(), x, y);
            return;
        }
        detail::vbcsr_mult(graph, active_vbcsr_backend(), x, y);
    }
    
    // Refined mult with offsets
    void mult_optimized(DistVector<T>& x, DistVector<T>& y) {
        mult(x, y);
    }

    // Matrix-Matrix Multiplication (Dense RHS)
    void mult_dense(DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        if (kind == MatrixKind::CSR) {
            detail::csr_mult_dense(graph, active_csr_backend(), X, Y);
            return;
        }
        if (kind == MatrixKind::BSR) {
            detail::bsr_mult_dense(graph, active_bsr_backend(), X, Y);
            return;
        }
        detail::vbcsr_mult_dense(graph, active_vbcsr_backend(), X, Y);
    }

    // Adjoint Matrix-Vector Multiplication: y = A^dagger * x
    void mult_adjoint(DistVector<T>& x, DistVector<T>& y) {
        if (kind == MatrixKind::CSR) {
            detail::csr_mult_adjoint(graph, active_csr_backend(), x, y);
            return;
        }
        if (kind == MatrixKind::BSR) {
            detail::bsr_mult_adjoint(graph, active_bsr_backend(), x, y);
            return;
        }
        detail::vbcsr_mult_adjoint(graph, active_vbcsr_backend(), x, y);
    }

    // Adjoint Matrix-Matrix Multiplication: Y = A^dagger * X
    void mult_dense_adjoint(DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        if (kind == MatrixKind::CSR) {
            detail::csr_mult_dense_adjoint(graph, active_csr_backend(), X, Y);
            return;
        }
        if (kind == MatrixKind::BSR) {
            detail::bsr_mult_dense_adjoint(graph, active_bsr_backend(), X, Y);
            return;
        }
        detail::vbcsr_mult_dense_adjoint(graph, active_vbcsr_backend(), X, Y);
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

    void copy_from(const BlockSpMat<T>& other) {
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
    

    // Return the real part as a new BlockSpMat<real_type>. For real T this is
    // a plain copy.
    auto get_real() const {
        using RealT = typename ScalarTraits<T>::real_type;
        // The result reuses the same graph with a different scalar type, so a
        // matrix-owned graph must be promoted before the pointer is shared.
        prepare_graph_for_shared_use();
        BlockSpMat<RealT> res(graph);
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
        // As in get_real(), the graph is shared between two matrix objects with
        // independent value storage.
        prepare_graph_for_shared_use();
        BlockSpMat<RealT> res(graph);
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
        // Only owned rows have CSR storage: adj_ptr has n_owned+1 entries, so a ghost local
        // index (>= n_owned, see DistGraph's ghost convention) would read out of range here.
        // Absent -> empty, consistent with the not-found return below.
        if (local_row < 0 || local_row + 1 >= static_cast<int>(graph->adj_ptr.size())) {
            return std::vector<T>();
        }
        int start = graph->adj_ptr[local_row];
        int end = graph->adj_ptr[local_row+1];
        
        for (int k = start; k < end; ++k) {
            if (graph->adj_ind[k] == local_col) {
                int r_dim = block_row_dim(local_row);
                int c_dim = block_col_dim(local_col);
                size_t sz = block_size_elements(k);
                
                std::vector<T> result(sz);
                const T* block_ptr = block_data(k);

                if (layout == kCanonicalBlockLayout) {
                    std::memcpy(result.data(), block_ptr, sz * sizeof(T));
                } else {
                    // Transpose canonical RowMajor -> ColMajor
                    for (int r = 0; r < r_dim; ++r) {
                        for (int c = 0; c < c_dim; ++c) {
                            result[c * r_dim + r] = block_ptr[r * c_dim + c];
                        }
                    }
                }
                return result;
            }
        }
        return std::vector<T>(); // Empty if not found
    }

    // Export packed data for Python/Scipy
    // Returns a single vector containing all blocks concatenated.
    // Blocks are ordered by (row, col) as in col_ind.
    // Internal storage is canonical RowMajor, so the default export is a
    // straight copy; ColMajor requests transpose per block.
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

                // Internal storage is canonical RowMajor
                if (layout == kCanonicalBlockLayout) {
                    std::memcpy(dest, block_ptr, sz * sizeof(T));
                } else {
                    // Transpose canonical RowMajor -> ColMajor
                    // Internal: A[i*c_dim + j]
                    // Output:   A[j*r_dim + i]
                    for (int r = 0; r < r_dim; ++r) {
                        for (int c = 0; c < c_dim; ++c) {
                            dest[c * r_dim + r] = block_ptr[r * c_dim + c];
                        }
                    }
                }
                offset += sz;
            }
        }
        return result;
    }

    void axpby(T alpha, const BlockSpMat<T>& X, T beta) {
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
            detail::CSRAxpbyExecutor<BlockSpMat<T>>::run(*this, X, alpha, beta);
            return;
        }
        if (kind == MatrixKind::BSR && X.kind == MatrixKind::BSR) {
            detail::BSRAxpbyExecutor<BlockSpMat<T>>::run(*this, X, alpha, beta);
            return;
        }
        detail::VBCSRAxpbyExecutor<BlockSpMat<T>>::run(*this, X, alpha, beta);
    }

    void axpy(T alpha, const BlockSpMat<T>& other) {
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
                    
                    // Add alpha to diagonal of the block.
                    // Canonical RowMajor storage: element (j, j) sits at
                    // [j*c_dim + j] for j=0..min(r,c).

                    int min_dim = std::min(r_dim, c_dim);
                    for (int j = 0; j < min_dim; ++j) {
                        target[j * c_dim + j] += alpha;
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
                        target[j * c_dim + j] += diag[row_offset + j];
                    }
                    break;
                }
            }
        }
        norms_valid = false;
    }

    // Compute C = [H, R] where R is a scalar diagonal operator stored in a DistVector.
    // Each block entry follows C(rc) = H(rc) * (R_col(c) - R_row(r)).
    void commutator_diagonal(const DistVector<T>& diag, BlockSpMat<T>& result) {
        if (!result.has_same_logical_structure(*this)) {
            result = this->duplicate(false);
            result.fill(T(0));
        }
        result.norms_valid = false;

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
                
                for (int r = 0; r < row_dim; ++r) {
                    for (int c = 0; c < col_dim; ++c) {
                        const int idx = r * col_dim + c;
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
                int dest_graph_block = new_graph->adj_ptr[row];
                for (int slot = graph->adj_ptr[row]; slot < graph->adj_ptr[row + 1]; ++slot) {
                    if (block_norms[slot] < threshold) {
                        continue;
                    }

                    const size_t size = block_size_elements(slot);
                    std::memcpy(
                        result.mutable_block_data(dest_graph_block),
                        block_data(slot),
                        size * sizeof(T));
                    new_norms[dest_graph_block] = block_norms[slot];
                    ++dest_graph_block;
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

        // TODO: optimizable for direct branching on the trans condition without explicitly form the transpose matrix
        if (transA) {
            BlockSpMat A_T = this->transpose();
            return A_T.spmm(B, threshold, false, transB);
        }

        if (transB) {
            BlockSpMat B_T = B.transpose();
            return this->spmm(B_T, threshold, transA, false);
        }

        if (kind == MatrixKind::CSR && B.kind == MatrixKind::CSR) {
            return detail::CSRSpMMExecutor<BlockSpMat<T>>::run(*this, B, threshold);
        }
        if (kind == MatrixKind::BSR && B.kind == MatrixKind::BSR) {
            return detail::BSRSpMMExecutor<BlockSpMat<T>>::run(*this, B, threshold);
        }
        
        return detail::VBCSRSpMMExecutor<BlockSpMat<T>>::run(*this, B, threshold);
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

    void fill_random() {
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        #pragma omp parallel for
        for (int i = 0; i < graph->adj_ind.size(); ++i) {
            T* data = mutable_block_data(i);
            const size_t size = block_size_elements(i);
            for (size_t j = 0; j < size; ++j) {
                if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
                    data[j] = T(dist(rng), dist(rng));
                } else {
                    data[j] = dist(rng);
                }
            }
        }
        norms_valid = false;
    }

    BlockSpMat transpose() const {
        if (kind == MatrixKind::CSR) {
            return detail::CSRTransposeExecutor<BlockSpMat<T>>::run(*this);
        }
        if (kind == MatrixKind::BSR) {
            return detail::BSRTransposeExecutor<BlockSpMat<T>>::run(*this);
        }

        return detail::VBCSRTransposeExecutor<BlockSpMat<T>>::run(*this);
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
                
                const T* block_data = this->block_data(k);

                // Block is stored in canonical RowMajor
                for (int r = 0; r < r_dim; ++r) {
                    for (int c = 0; c < c_dim; ++c) {
                        T value = block_data[r * c_dim + c];

                        // Write (row, col, val)
                        file << (row_start_idx + r) << " " << (col_start_idx + c) << " ";
                        MMWriter<T>::write(file, value);
                        file << "\n";
                    }
                }
            }
        }
    }

    // Extract submatrix defined by global_indices
    BlockSpMat<T> extract_submatrix(const std::vector<int>& global_indices) {
        auto ctx = detail::fetch_batched_block_payloads(*this, {global_indices});
        return construct_submatrix(global_indices, ctx);
    }

    // Extract multiple submatrices efficiently
    std::vector<BlockSpMat<T>> extract_submatrix_batched(const std::vector<std::vector<int>>& batch_indices) {
        auto ctx = detail::fetch_batched_block_payloads(*this, batch_indices);
        std::vector<BlockSpMat<T>> results;
        results.reserve(batch_indices.size());
        // TODO: threading?
        for(const auto& indices : batch_indices) {
            results.push_back(construct_submatrix(indices, ctx));
        }
        return results;
    }

    // Insert submatrix back (In-Place)
    void insert_submatrix(const BlockSpMat<T>& submat, const std::vector<int>& global_indices) {
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
                // Data is in the canonical layout (internal storage of submat).
                this->add_block(global_row, global_col, data, r_dim, c_dim, AssemblyMode::INSERT, kCanonicalBlockLayout);
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
                
                const T* data = block_data(k);

                // Copy block to dense: both are row-major now, so each block
                // row is one contiguous memcpy into the dense view.
                for(int r=0; r<r_dim; ++r) {
                    int dr = row_offset + r;
                    if (dr >= my_rows || col_offset + c_dim > my_cols) {
                        continue;
                    }
                    std::memcpy(
                        dense.data() + static_cast<size_t>(dr) * my_cols + col_offset,
                        data + static_cast<size_t>(r) * c_dim,
                        static_cast<size_t>(c_dim) * sizeof(T));
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
                
                T* data = mutable_block_data(k);

                // Copy dense to block: both row-major, contiguous per block row.
                for(int r=0; r<r_dim; ++r) {
                    int dr = row_offset + r;
                    if (dr >= my_rows || col_offset + c_dim > my_cols) {
                        continue;
                    }
                    std::memcpy(
                        data + static_cast<size_t>(r) * c_dim,
                        dense.data() + static_cast<size_t>(dr) * my_cols + col_offset,
                        static_cast<size_t>(c_dim) * sizeof(T));
                }
            }
        }
    }
    // Calculate block density (global nnz blocks / total global blocks^2)
    double get_block_density() const {
        // TODO: can switch to a more precise method to targeting per element fidelity?
        // need to use a weighted row based percentage average method.
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

    // Remote add_block calls can arrive before assemble_remote_blocks() folds
    // them into local storage. The buffers live outside the object and are keyed
    // by BlockSpMat address, so move construction/assignment must transfer them
    // and destruction must clear them.
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
    template <typename Backend>
    Backend& active_backend_storage();
    template <typename Backend>
    const Backend& active_backend_storage() const;
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
    static BackendHandle build_backend_for_structure(
        MatrixKind matrix_kind,
        DistGraph* graph,
        uint32_t page_size);

    void attach_backend(VBCSRBackendStorage backend);
    void attach_backend(CSRBackendStorage backend);
    void attach_backend(BSRBackendStorage backend);
    void attach_backend(BackendHandle backend);
    static void ensure_same_backend_family(const BlockSpMat& lhs, const BlockSpMat& rhs, const char* op_name);
    static void ensure_csr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs);
    static void ensure_bsr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs);
    static void ensure_vbcsr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs);
    bool has_same_logical_structure(const BlockSpMat& other) const;

    using GhostBlockRef = detail::GhostBlockRef<T>;
    using BlockData = detail::FetchedBlock<T>;
    using FetchContext = detail::FetchedBlockContext<T>;

    // Construct a submatrix from fetched data
    BlockSpMat<T> construct_submatrix(const std::vector<int>& global_indices, const FetchContext& ctx) {
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

        BlockSpMat<T> sub_mat(sub_graph);
        sub_mat.owns_graph = true;
        sub_mat.graph->enable_matrix_lifetime_management();

        for(const auto* bd : relevant_blocks) {
            int sub_row = global_to_sub[bd->global_row];
            int sub_col = global_to_sub[bd->global_col];

            sub_mat.add_block(sub_row, sub_col, bd->data.data(), bd->r_dim, bd->c_dim, AssemblyMode::INSERT, kCanonicalBlockLayout);
        }

        sub_mat.assemble();
        return sub_mat;
    }
};

template <typename T>
double BlockSpMat<T>::get_sq_norm(const T& v) {
    if constexpr (std::is_same<T, std::complex<double>>::value ||
                  std::is_same<T, std::complex<float>>::value) {
        return std::norm(v);
    } else {
        return v * v;
    }
}

template <typename T>
std::vector<double> BlockSpMat<T>::compute_block_norms() const {
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

template <typename T>
int BlockSpMat<T>::max_omp_threads() {
    int max_threads = 1;
    #ifdef _OPENMP
    max_threads = omp_get_max_threads();
    #endif
    return max_threads;
}

template <typename T>
typename BlockSpMat<T>::RemoteAssemblyRegistry& BlockSpMat<T>::remote_assembly_registry() {
    static RemoteAssemblyRegistry instance;
    return instance;
}

template <typename T>
void BlockSpMat<T>::release_graph_reference() const {
    if (graph == nullptr) {
        return;
    }
    if (owns_graph) {
        // Whether a graph has lifetime management enabled is determined by whether it have
        // at least one owner. During its release, it will promote the graph to the lifetime management
        // so the release of the matrix who referred to the graph will trigger the graph deletion.
        graph->enable_matrix_lifetime_management();
    }
    if (graph->release_matrix_reference()) {
        delete graph;
    }
}

template <typename T>
void BlockSpMat<T>::prepare_graph_for_shared_use() const {
    DistGraph* live_graph = require_live_graph(graph, "prepare_graph_for_shared_use");
    if (owns_graph && !live_graph->has_managed_matrix_lifetime()) {
        // Promote a singly owned graph before another matrix starts sharing it.
        live_graph->enable_matrix_lifetime_management();
    }
}

template <typename T>
void BlockSpMat<T>::require_assembled_for_state_copy(const char* op_name) const {
    if (has_pending_remote_assembly(this)) {
        throw std::runtime_error(
            std::string(op_name) +
            " requires an assembled matrix; pending remote add_block updates still exist");
    }
}

template <typename T>
BlockSpMat<T>::BlockSpMat(
    DistGraph* g,
    MatrixKind matrix_kind,
    bool owns_graph,
    ConstructionToken)
    : kind(matrix_kind),
      backend_handle_(std::monostate{}),
      configured_page_size_(default_page_size_for(matrix_kind, require_live_graph(g, "BlockSpMat"))),
      graph(require_live_graph(g, "BlockSpMat")),
      owns_graph(owns_graph) {
    // Reference accounting is unconditional. Ownership only controls whether
    // the graph is eligible for deletion when the final matrix releases it.
    graph->acquire_matrix_reference();
    if (owns_graph) {
        graph->enable_matrix_lifetime_management();
    }
}

template <typename T>
template <typename Backend>
Backend& BlockSpMat<T>::active_backend_storage() {
    auto* storage = std::get_if<Backend>(&backend_handle_);
    if (storage == nullptr) {
        throw std::logic_error("Active backend does not match requested storage path");
    }
    return *storage;
}

template <typename T>
template <typename Backend>
const Backend& BlockSpMat<T>::active_backend_storage() const {
    const auto* storage = std::get_if<Backend>(&backend_handle_);
    if (storage == nullptr) {
        throw std::logic_error("Active backend does not match requested storage path");
    }
    return *storage;
}

template <typename T>
typename BlockSpMat<T>::VBCSRBackendStorage& BlockSpMat<T>::active_vbcsr_backend() {
    return active_backend_storage<VBCSRBackendStorage>();
}

template <typename T>
const typename BlockSpMat<T>::VBCSRBackendStorage& BlockSpMat<T>::active_vbcsr_backend() const {
    return active_backend_storage<VBCSRBackendStorage>();
}

template <typename T>
typename BlockSpMat<T>::CSRBackendStorage& BlockSpMat<T>::active_csr_backend() {
    return active_backend_storage<CSRBackendStorage>();
}

template <typename T>
const typename BlockSpMat<T>::CSRBackendStorage& BlockSpMat<T>::active_csr_backend() const {
    return active_backend_storage<CSRBackendStorage>();
}

template <typename T>
typename BlockSpMat<T>::BSRBackendStorage& BlockSpMat<T>::active_bsr_backend() {
    return active_backend_storage<BSRBackendStorage>();
}

template <typename T>
const typename BlockSpMat<T>::BSRBackendStorage& BlockSpMat<T>::active_bsr_backend() const {
    return active_backend_storage<BSRBackendStorage>();
}

template <typename T>
typename BlockSpMat<T>::RemoteThreadBuffers& BlockSpMat<T>::remote_assembly_buffers() const {
    auto& reg = remote_assembly_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto& buffers = reg.buffers[this];
    if (buffers.empty()) {
        buffers.resize(static_cast<size_t>(max_omp_threads()));
    }
    return buffers;
}

template <typename T>
bool BlockSpMat<T>::has_pending_remote_assembly(const BlockSpMat* matrix) {
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

template <typename T>
void BlockSpMat<T>::transfer_remote_assembly_state(const BlockSpMat* from, const BlockSpMat* to) {
    auto& reg = remote_assembly_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.buffers.find(from);
    if (it == reg.buffers.end()) {
        return;
    }
    reg.buffers[to] = std::move(it->second);
    reg.buffers.erase(it);
}

template <typename T>
void BlockSpMat<T>::clear_remote_assembly_state(const BlockSpMat* matrix) {
    auto& reg = remote_assembly_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.buffers.erase(matrix);
}

template <typename T>
DistGraph* BlockSpMat<T>::require_live_graph(DistGraph* graph, const char* op_name) {
    if (graph == nullptr) {
        throw std::invalid_argument(std::string(op_name) + " requires a live DistGraph");
    }
    return graph;
}

template <typename T>
const DistGraph* BlockSpMat<T>::require_live_graph(const DistGraph* graph, const char* op_name) {
    if (graph == nullptr) {
        throw std::invalid_argument(std::string(op_name) + " requires a live DistGraph");
    }
    return graph;
}

template <typename T>
uint32_t BlockSpMat<T>::default_page_size_for(
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

template <typename T>
uint32_t BlockSpMat<T>::normalize_page_size(
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

template <typename T>
void BlockSpMat<T>::rebuild_backend_for_page_size(const char* op_name) {
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

template <typename T>
MatrixKind BlockSpMat<T>::detect_matrix_kind(const DistGraph* g) {
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

template <typename T>
typename BlockSpMat<T>::BackendHandle BlockSpMat<T>::build_backend_for_structure(
    MatrixKind matrix_kind,
    DistGraph* graph,
    uint32_t page_size) {
    graph = require_live_graph(graph, "build_backend_for_structure");
    const uint32_t normalized = normalize_page_size(matrix_kind, graph, page_size);
    switch (matrix_kind) {
    case MatrixKind::CSR: {
        CSRBackendStorage backend;
        backend.initialize_structure(graph->adj_ind.size(), normalized);
        return BackendHandle(
            std::in_place_type<CSRBackendStorage>,
            std::move(backend));
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
        return BackendHandle(
            std::in_place_type<BSRBackendStorage>,
            std::move(backend));
    }
    case MatrixKind::VBCSR: {
        VBCSRBackendStorage backend(normalized);
        const size_t nnz = graph->adj_ind.size();
        backend.initialize_graph_block_handles(nnz);

        // count the #matrices per shape
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
        return BackendHandle(
            std::in_place_type<VBCSRBackendStorage>,
            std::move(backend));
    }
    }
    throw std::runtime_error("Unsupported matrix kind in build_backend_for_structure");
}

template <typename T>
void BlockSpMat<T>::attach_backend(VBCSRBackendStorage backend) {
    configured_page_size_ = backend.configured_blocks_per_page();
    backend_handle_.template emplace<VBCSRBackendStorage>(std::move(backend));
    block_norms.clear();
    norms_valid = false;
}

template <typename T>
void BlockSpMat<T>::attach_backend(CSRBackendStorage backend) {
    configured_page_size_ = backend.configured_page_size();
    backend_handle_.template emplace<CSRBackendStorage>(std::move(backend));
    block_norms.clear();
    norms_valid = false;
}

template <typename T>
void BlockSpMat<T>::attach_backend(BSRBackendStorage backend) {
    configured_page_size_ = backend.configured_blocks_per_page();
    backend_handle_.template emplace<BSRBackendStorage>(std::move(backend));
    block_norms.clear();
    norms_valid = false;
}

template <typename T>
void BlockSpMat<T>::attach_backend(BackendHandle backend) {
    std::visit(
        [this](auto&& committed_backend) {
            using Backend = std::decay_t<decltype(committed_backend)>;
            if constexpr (std::is_same_v<Backend, std::monostate>) {
                throw std::logic_error("Cannot attach an empty backend handle");
            } else {
                attach_backend(std::move(committed_backend));
            }
        },
        std::move(backend));
}

template <typename T>
void BlockSpMat<T>::ensure_same_backend_family(
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

template <typename T>
void BlockSpMat<T>::ensure_csr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs) {
    if (lhs.graph->owned_global_indices != rhs.graph->owned_global_indices) {
        throw std::runtime_error("CSR binary operation requires matching owned row distribution");
    }
}

template <typename T>
void BlockSpMat<T>::ensure_bsr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs) {
    if (lhs.graph->owned_global_indices != rhs.graph->owned_global_indices) {
        throw std::runtime_error("BSR binary operation requires matching owned row distribution");
    }
    const int lhs_block_size = lhs.active_bsr_backend().block_size;
    const int rhs_block_size = rhs.active_bsr_backend().block_size;
    if (lhs_block_size != rhs_block_size) {
        throw std::runtime_error("BSR binary operation requires matching uniform block sizes");
    }
}

template <typename T>
void BlockSpMat<T>::ensure_vbcsr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs) {
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

template <typename T>
bool BlockSpMat<T>::has_same_logical_structure(const BlockSpMat& other) const {
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

template <typename T>
bool BlockSpMat<T>::update_local_block(
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
                if (layout == kCanonicalBlockLayout) {
                    std::memcpy(target, data, r_dim * c_dim * sizeof(T));
                } else {
                    // Transpose ColMajor input into canonical RowMajor storage.
                    for (int i = 0; i < r_dim; ++i) {
                        for (int j = 0; j < c_dim; ++j) {
                            target[i * c_dim + j] = data[j * r_dim + i];
                        }
                    }
                }
            } else {
                if (layout == kCanonicalBlockLayout) {
                    for (int i = 0; i < r_dim * c_dim; ++i) {
                        target[i] += data[i];
                    }
                } else {
                    for (int i = 0; i < r_dim; ++i) {
                        for (int j = 0; j < c_dim; ++j) {
                            target[i * c_dim + j] += data[j * r_dim + i];
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
