#ifndef VBCSR_BLOCK_CSR_HPP
#define VBCSR_BLOCK_CSR_HPP

#include "dist_graph.hpp"
#include "dist_vector.hpp"
#include "dist_multivector.hpp"
#include "kernels.hpp"
#include "block_memory_pool.hpp" // Added
#include "detail/backend_handle.hpp"
#include "detail/bsr_kernels.hpp"
#include "detail/bsr_result_builder.hpp"
#include "detail/csr_kernels.hpp"
#include "detail/csr_result_builder.hpp"
#include "detail/legacy_matrix_backend.hpp"
#include "detail/distributed_result_graph.hpp"
#include "detail/spmm_exchange.hpp"
#include "detail/transpose_exchange.hpp"
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

template <typename T, typename Kernel>
class BlockSpMat;

namespace detail {
template <typename Matrix>
struct CSRSpMMExecutor;
template <typename Matrix>
struct BSRSpMMExecutor;
template <typename Matrix>
struct CSRTransposeExecutor;
template <typename Matrix>
struct BSRTransposeExecutor;
template <typename Matrix>
struct LegacyTransposeExecutor;
template <typename Matrix>
struct CSRAxpbyExecutor;
template <typename Matrix>
struct BSRAxpbyExecutor;
template <typename Matrix>
struct LegacyAxpbyExecutor;
template <typename Matrix>
struct LegacySpMMExecutor;
}

template <typename T, typename Kernel>
class LegacyMatrixBuilder {
public:
    LegacyMatrixBuilder(DistGraph* graph, bool owns_graph)
        : graph_(graph), owns_graph_(owns_graph) {}

    BlockSpMat<T, Kernel> materialize() const;
    void write_transposed_conjugate_slot(
        BlockSpMat<T, Kernel>& matrix,
        int slot,
        const T* src,
        int src_rows,
        int src_cols) const;

private:
    DistGraph* graph_;
    bool owns_graph_;
};

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
public:
    DistGraph* graph;
    using value_type = T;
    using KernelType = Kernel;

private:
    template <typename, typename>
    friend class BlockSpMat;
    template <typename, typename>
    friend class LegacyMatrixBuilder;
    template <typename>
    friend struct detail::CSRSpMMExecutor;
    template <typename>
    friend struct detail::BSRSpMMExecutor;
    template <typename>
    friend struct detail::CSRTransposeExecutor;
    template <typename>
    friend struct detail::BSRTransposeExecutor;
    template <typename>
    friend struct detail::LegacyTransposeExecutor;
    template <typename>
    friend struct detail::CSRAxpbyExecutor;
    template <typename>
    friend struct detail::BSRAxpbyExecutor;
    template <typename>
    friend struct detail::LegacyAxpbyExecutor;
    template <typename>
    friend struct detail::LegacySpMMExecutor;

    MatrixKind kind = MatrixKind::CSR;
    using LegacyBackendStorage = detail::LegacyMatrixBackend<T>;
    using CSRBackendStorage = detail::CSRMatrixBackend<T>;
    using BSRBackendStorage = detail::BSRMatrixBackend<T>;
    using CommittedBackendStorage = std::variant<LegacyBackendStorage, CSRBackendStorage, BSRBackendStorage>;
    using BackendHandle = detail::MatrixBackendHandle<T, Kernel>;

    struct ConstructionToken {};

public:
    struct ConstLocalBlockView {
        int slot = -1;
        int row = -1;
        int col = -1;
        int row_dim = 0;
        int col_dim = 0;
        size_t size = 0;
        const T* values = nullptr;
    };

    struct LocalBlockView {
        int slot = -1;
        int row = -1;
        int col = -1;
        int row_dim = 0;
        int col_dim = 0;
        size_t size = 0;
        T* values = nullptr;
    };
    
    // Logical block structure views (local indices), graph-backed in steady state
    detail::BoundVector<int> row_ptr;
    detail::BoundVector<int> col_ind;

private:
    std::vector<int> detached_row_ptr_storage_;
    std::vector<int> detached_col_ind_storage_;
    LegacyBackendStorage legacy_backend_storage_;
    CSRBackendStorage csr_backend_storage_;
    BSRBackendStorage bsr_backend_storage_;
    BackendHandle backend_handle_;
    detail::BoundVector<uint64_t> blk_handles;
    detail::BoundVector<size_t> blk_sizes;
    detail::BoundArena<T> arena;

public:

    // Cached block norms
    mutable std::vector<double> block_norms;
    mutable bool norms_valid = false;

    bool owns_graph = false;

    // Buffer for remote assembly
    struct PendingBlock {
        int g_row;
        int g_col;
        int rows;
        int cols;
        AssemblyMode mode;
        std::vector<T> data;
    };
    // Map: Owner -> (GlobalRow, GlobalCol) -> PendingBlock
    // Thread-local storage for remote blocks to avoid locking
    std::vector<std::map<int, std::map<std::pair<int, int>, PendingBlock>>> thread_remote_blocks;

    // Helper for squared norm
    static double get_sq_norm(const T& v) {
        if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
            return std::norm(v);
        } else {
            return v * v;
        }
    }

    // Helper to compute Frobenius norms of local blocks
    std::vector<double> compute_block_norms() const {
        int nnz = static_cast<int>(active_col_ind().size());
        std::vector<double> norms(nnz);
        
        #pragma omp parallel for
        for (int i = 0; i < nnz; ++i) {
            double sum = 0.0;
            uint64_t handle = blk_handles[i];
            T* data = arena.get_ptr(handle);
            size_t size = blk_sizes[i];
            for (size_t k = 0; k < size; ++k) {
                sum += get_sq_norm(data[k]);
            }
            norms[i] = std::sqrt(sum);
        }
        return norms;
    }

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

    const std::vector<int>& logical_row_ptr() const {
        return active_row_ptr();
    }

    const std::vector<int>& logical_col_ind() const {
        return active_col_ind();
    }

    size_t local_block_nnz() const {
        return blk_handles.size();
    }

    size_t local_scalar_nnz() const {
        if (kind == MatrixKind::CSR) {
            return csr_backend_storage_.local_scalar_nnz();
        }
        if (kind == MatrixKind::BSR) {
            return bsr_backend_storage_.local_scalar_nnz();
        }
        return legacy_backend_storage_.local_scalar_nnz();
    }

    int block_row_count() const {
        const auto& structure_row_ptr = active_row_ptr();
        return structure_row_ptr.empty() ? 0 : static_cast<int>(structure_row_ptr.size()) - 1;
    }

    int block_row_from_slot(int slot) const {
        const auto& structure_row_ptr = active_row_ptr();
        auto it = std::upper_bound(structure_row_ptr.begin(), structure_row_ptr.end(), slot);
        return std::max(0, static_cast<int>(std::distance(structure_row_ptr.begin(), it) - 1));
    }

    int block_col_from_slot(int slot) const {
        return active_col_ind()[slot];
    }

    int block_row_dim(int local_row) const {
        return graph->block_sizes[local_row];
    }

    int block_col_dim(int local_col) const {
        return graph->block_sizes[local_col];
    }

    const T* block_data(int slot) const {
        return arena.get_ptr(blk_handles[slot]);
    }

    T* mutable_block_data(int slot) {
        norms_valid = false;
        return arena.get_ptr(blk_handles[slot]);
    }

    size_t block_size_elements(int slot) const {
        return blk_sizes[slot];
    }

    ConstLocalBlockView local_block_view(int slot) const {
        const int row = block_row_from_slot(slot);
        const int col = active_col_ind()[slot];
        return ConstLocalBlockView{
            slot,
            row,
            col,
            graph->block_sizes[row],
            graph->block_sizes[col],
            blk_sizes[slot],
            block_data(slot)};
    }

    LocalBlockView local_block_view(int slot) {
        const int row = block_row_from_slot(slot);
        const int col = active_col_ind()[slot];
        return LocalBlockView{
            slot,
            row,
            col,
            graph->block_sizes[row],
            graph->block_sizes[col],
            blk_sizes[slot],
            mutable_block_data(slot)};
    }

    template <typename Fn>
    void for_each_local_block(Fn&& fn) const {
        const int n_rows = block_row_count();
        const auto& structure_row_ptr = active_row_ptr();
        const auto& structure_col_ind = active_col_ind();
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
                    blk_sizes[slot],
                    block_data(slot)});
            }
        }
    }

    template <typename Fn>
    void for_each_local_block(Fn&& fn) {
        const int n_rows = block_row_count();
        const auto& structure_row_ptr = active_row_ptr();
        const auto& structure_col_ind = active_col_ind();
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
                    blk_sizes[slot],
                    mutable_block_data(slot)});
            }
        }
    }

public:
    BlockSpMat(DistGraph* g)
        : BlockSpMat(g, detect_matrix_kind(g), false, ConstructionToken{}) {
        allocate_from_graph();
    }

private:
    BlockSpMat(DistGraph* g, MatrixKind matrix_kind, bool owns_graph_flag, ConstructionToken)
        : graph(g),
          kind(matrix_kind),
          row_ptr(),
          col_ind(),
          detached_row_ptr_storage_(),
          detached_col_ind_storage_(),
          legacy_backend_storage_(),
          csr_backend_storage_(),
          bsr_backend_storage_(),
          backend_handle_(std::monostate{}),
          blk_handles(),
          blk_sizes(),
          arena(),
          owns_graph(owns_graph_flag) {
        bind_structure_views();
        bind_storage_views_for_kind(kind);
        int max_threads = 1;
        #ifdef _OPENMP
        max_threads = omp_get_max_threads();
        #endif
        thread_remote_blocks.resize(max_threads);
    }

public:
    ~BlockSpMat() {
        if (owns_graph && graph) {
            delete graph;
        }
    }

    // Move constructor
    BlockSpMat(BlockSpMat&& other) noexcept : 
        graph(other.graph), 
        kind(other.kind),
        row_ptr(),
        col_ind(),
        detached_row_ptr_storage_(std::move(other.detached_row_ptr_storage_)),
        detached_col_ind_storage_(std::move(other.detached_col_ind_storage_)),
        legacy_backend_storage_(std::move(other.legacy_backend_storage_)),
        csr_backend_storage_(std::move(other.csr_backend_storage_)),
        bsr_backend_storage_(std::move(other.bsr_backend_storage_)),
        backend_handle_(std::monostate{}),
        blk_handles(),
        blk_sizes(),
        arena(),
        block_norms(std::move(other.block_norms)),
        norms_valid(other.norms_valid),
        owns_graph(other.owns_graph),
        thread_remote_blocks(std::move(other.thread_remote_blocks))
    {
        bind_structure_views();
        bind_storage_views_for_kind(kind);
        other.graph = nullptr;
        other.owns_graph = false;
        other.bind_structure_views();
    }

    // Move assignment
    BlockSpMat& operator=(BlockSpMat&& other) noexcept {
        if (this != &other) {
            if (owns_graph && graph) delete graph;
            graph = other.graph;
            kind = other.kind;
            detached_row_ptr_storage_ = std::move(other.detached_row_ptr_storage_);
            detached_col_ind_storage_ = std::move(other.detached_col_ind_storage_);
            legacy_backend_storage_ = std::move(other.legacy_backend_storage_);
            csr_backend_storage_ = std::move(other.csr_backend_storage_);
            bsr_backend_storage_ = std::move(other.bsr_backend_storage_);
            bind_structure_views();
            bind_storage_views_for_kind(kind);
            
            // Fix: Move norms state
            block_norms = std::move(other.block_norms);
            norms_valid = other.norms_valid;
            owns_graph = other.owns_graph;
            thread_remote_blocks = std::move(other.thread_remote_blocks);
            
            other.graph = nullptr;
            other.owns_graph = false;
            other.bind_structure_views();
        }
        return *this;
    }

    // Disable copy (use duplicate() instead)
    BlockSpMat(const BlockSpMat&) = delete;
    BlockSpMat& operator=(const BlockSpMat&) = delete;

    // Create a deep copy of the matrix
    BlockSpMat<T, Kernel> duplicate(bool independent_graph = true) const {
        DistGraph* new_graph = graph;
        bool new_owns_graph = false;
        if (independent_graph && graph) {
            new_graph = graph->duplicate();
            new_owns_graph = true;
        }
        std::vector<int> row_ptr_copy;
        std::vector<int> col_ind_copy;
        if (new_graph != nullptr) {
            new_graph->get_matrix_structure(row_ptr_copy, col_ind_copy);
        }
        BlockSpMat<T, Kernel> new_mat(new_graph, kind, new_owns_graph, ConstructionToken{});
        new_mat.attach_backend(build_backend_for_structure(kind, new_graph, row_ptr_copy, col_ind_copy));
        new_mat.copy_from(*this);
        if (norms_valid) {
            new_mat.block_norms = block_norms;
            new_mat.norms_valid = true;
        }
        return new_mat;
    }

private:
    static MatrixKind detect_matrix_kind(const DistGraph* g) {
        if (g == nullptr) {
            return MatrixKind::CSR;
        }

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

    void bind_structure_views() {
        if (graph != nullptr) {
            row_ptr.bind(graph->adj_ptr);
            col_ind.bind(graph->adj_ind);
            return;
        }
        row_ptr.bind(detached_row_ptr_storage_);
        col_ind.bind(detached_col_ind_storage_);
    }

    const std::vector<int>& active_row_ptr() const {
        if (graph != nullptr) {
            return graph->adj_ptr;
        }
        return detached_row_ptr_storage_;
    }

    const std::vector<int>& active_col_ind() const {
        if (graph != nullptr) {
            return graph->adj_ind;
        }
        return detached_col_ind_storage_;
    }

    void bind_legacy_backend_handle() {
        blk_handles.bind(legacy_backend_storage_.blk_handles);
        blk_sizes.bind(legacy_backend_storage_.blk_sizes);
        arena.bind(legacy_backend_storage_.arena);
        backend_handle_ = detail::make_legacy_backend_handle<T, Kernel>(legacy_backend_storage_);
    }

    void bind_csr_backend_handle() {
        blk_handles.bind(csr_backend_storage_.blk_handles);
        blk_sizes.bind(csr_backend_storage_.blk_sizes);
        arena.bind(csr_backend_storage_.arena);
        backend_handle_ = detail::make_csr_backend_handle<T, Kernel>(csr_backend_storage_);
    }

    void bind_bsr_backend_handle() {
        blk_handles.bind(bsr_backend_storage_.blk_handles);
        blk_sizes.bind(bsr_backend_storage_.blk_sizes);
        arena.bind(bsr_backend_storage_.arena);
        backend_handle_ = detail::make_bsr_backend_handle<T, Kernel>(bsr_backend_storage_);
    }

    void bind_storage_views_for_kind(MatrixKind matrix_kind) {
        if (matrix_kind == MatrixKind::CSR) {
            bind_csr_backend_handle();
            return;
        }
        if (matrix_kind == MatrixKind::BSR) {
            bind_bsr_backend_handle();
            return;
        }
        bind_legacy_backend_handle();
    }

    static LegacyBackendStorage build_legacy_backend_for_structure(
        DistGraph* graph,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind) {
        LegacyBackendStorage backend;
        if (graph == nullptr) {
            return backend;
        }

        const int nnz = static_cast<int>(col_ind.size());
        backend.blk_handles.resize(nnz);
        backend.blk_sizes.resize(nnz);

        unsigned long long total_elements = 0;
        const int n_owned = static_cast<int>(graph->owned_global_indices.size());

        #pragma omp parallel for reduction(+ : total_elements)
        for (int i = 0; i < n_owned; ++i) {
            const int r_dim = graph->block_sizes[i];
            unsigned long long row_total_elements = 0;
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int col = col_ind[k];
                const int c_dim = graph->block_sizes[col];
                row_total_elements += static_cast<unsigned long long>(r_dim) * c_dim;
            }
            total_elements += row_total_elements;
        }

        backend.arena.reserve(total_elements);

        for (int i = 0; i < n_owned; ++i) {
            const int r_dim = graph->block_sizes[i];
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                const int col = col_ind[k];
                const int c_dim = graph->block_sizes[col];
                const size_t sz = static_cast<size_t>(r_dim) * c_dim;
                backend.blk_handles[k] = backend.arena.allocate(sz);
                backend.blk_sizes[k] = sz;
            }
        }

        return backend;
    }

    static CSRBackendStorage build_csr_backend_for_structure(
        DistGraph* graph,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind) {
        if (graph == nullptr) {
            return CSRBackendStorage{};
        }
        (void)row_ptr;
        (void)col_ind;
        return std::move(detail::CSRResultBuilder<T>(graph)).commit_backend();
    }

    static BSRBackendStorage build_bsr_backend_for_structure(
        DistGraph* graph,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind) {
        if (graph == nullptr) {
            return BSRBackendStorage{};
        }
        (void)row_ptr;
        (void)col_ind;
        return std::move(detail::BSRResultBuilder<T>(graph)).commit_backend();
    }

    static CommittedBackendStorage build_backend_for_structure(
        MatrixKind matrix_kind,
        DistGraph* graph,
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind) {
        if (matrix_kind == MatrixKind::CSR) {
            return build_csr_backend_for_structure(graph, row_ptr, col_ind);
        }
        if (matrix_kind == MatrixKind::BSR) {
            return build_bsr_backend_for_structure(graph, row_ptr, col_ind);
        }
        return build_legacy_backend_for_structure(graph, row_ptr, col_ind);
    }

    void attach_backend(LegacyBackendStorage backend) {
        legacy_backend_storage_ = std::move(backend);
        bind_legacy_backend_handle();
        block_norms.clear();
        norms_valid = false;
    }

    void attach_backend(CSRBackendStorage backend) {
        csr_backend_storage_ = std::move(backend);
        bind_csr_backend_handle();
        block_norms.clear();
        norms_valid = false;
    }

    void attach_backend(BSRBackendStorage backend) {
        bsr_backend_storage_ = std::move(backend);
        bind_bsr_backend_handle();
        block_norms.clear();
        norms_valid = false;
    }

    void attach_backend(CommittedBackendStorage backend) {
        std::visit(
            [this](auto&& committed_backend) {
                attach_backend(std::move(committed_backend));
            },
            std::move(backend));
    }

    void replace_with_parts(
        DistGraph* new_graph,
        bool new_owns_graph,
        MatrixKind new_kind,
        std::vector<int> new_row_ptr,
        std::vector<int> new_col_ind,
        CommittedBackendStorage backend) {
        if (owns_graph && graph && graph != new_graph) {
            delete graph;
        }
        graph = new_graph;
        owns_graph = new_owns_graph;
        kind = new_kind;
        bind_structure_views();
        attach_backend(std::move(backend));
    }

    static BlockSpMat from_parts(
        DistGraph* graph,
        bool owns_graph,
        MatrixKind kind,
        std::vector<int> row_ptr,
        std::vector<int> col_ind,
        CommittedBackendStorage backend) {
        BlockSpMat matrix(graph, kind, owns_graph, ConstructionToken{});
        matrix.attach_backend(std::move(backend));
        return matrix;
    }

    static void ensure_csr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs) {
        if (lhs.graph->owned_global_indices != rhs.graph->owned_global_indices) {
            throw std::runtime_error("CSR binary operation requires matching owned row distribution");
        }
    }

    static void ensure_bsr_binary_compatibility(const BlockSpMat& lhs, const BlockSpMat& rhs) {
        if (lhs.graph->owned_global_indices != rhs.graph->owned_global_indices) {
            throw std::runtime_error("BSR binary operation requires matching owned row distribution");
        }
        const int lhs_block_size = detail::require_bsr_backend<T, Kernel>(lhs.backend_handle_).block_size;
        const int rhs_block_size = detail::require_bsr_backend<T, Kernel>(rhs.backend_handle_).block_size;
        if (lhs_block_size != rhs_block_size) {
            throw std::runtime_error("BSR binary operation requires matching uniform block sizes");
        }
    }

    static BlockSpMat from_csr_builder(
        DistGraph* graph,
        bool owns_graph,
        detail::CSRResultBuilder<T>&& builder) {
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        graph->get_matrix_structure(row_ptr, col_ind);
        CommittedBackendStorage backend = std::move(builder).commit_backend();
        return from_parts(
            graph,
            owns_graph,
            MatrixKind::CSR,
            std::move(row_ptr),
            std::move(col_ind),
            std::move(backend));
    }

    void replace_with_csr_builder(DistGraph* graph, bool owns_graph, detail::CSRResultBuilder<T>&& builder) {
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        graph->get_matrix_structure(row_ptr, col_ind);
        CommittedBackendStorage backend = std::move(builder).commit_backend();
        replace_with_parts(
            graph,
            owns_graph,
            MatrixKind::CSR,
            std::move(row_ptr),
            std::move(col_ind),
            std::move(backend));
    }

    static BlockSpMat from_bsr_builder(
        DistGraph* graph,
        bool owns_graph,
        detail::BSRResultBuilder<T>&& builder) {
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        graph->get_matrix_structure(row_ptr, col_ind);
        CommittedBackendStorage backend = std::move(builder).commit_backend();
        return from_parts(
            graph,
            owns_graph,
            MatrixKind::BSR,
            std::move(row_ptr),
            std::move(col_ind),
            std::move(backend));
    }

    void replace_with_bsr_builder(DistGraph* graph, bool owns_graph, detail::BSRResultBuilder<T>&& builder) {
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        graph->get_matrix_structure(row_ptr, col_ind);
        CommittedBackendStorage backend = std::move(builder).commit_backend();
        replace_with_parts(
            graph,
            owns_graph,
            MatrixKind::BSR,
            std::move(row_ptr),
            std::move(col_ind),
            std::move(backend));
    }

    static void write_transposed_conjugate_values(T* dest, const T* src, int src_rows, int src_cols) {
        for (int dest_col = 0; dest_col < src_rows; ++dest_col) {
            for (int dest_row = 0; dest_row < src_cols; ++dest_row) {
                dest[dest_col * src_cols + dest_row] =
                    ConjHelper<T>::apply(src[dest_row * src_rows + dest_col]);
            }
        }
    }

    BlockSpMat transpose_csr_serial() const {
        return detail::CSRTransposeExecutor<BlockSpMat<T, Kernel>>::serial(*this);
    }

    BlockSpMat transpose_csr_distributed() const {
        return detail::CSRTransposeExecutor<BlockSpMat<T, Kernel>>::distributed(*this);
    }

    BlockSpMat transpose_csr() const {
        return detail::CSRTransposeExecutor<BlockSpMat<T, Kernel>>::run(*this);
    }

    BlockSpMat transpose_bsr_serial() const {
        return detail::BSRTransposeExecutor<BlockSpMat<T, Kernel>>::serial(*this);
    }

    BlockSpMat transpose_bsr_distributed() const {
        return detail::BSRTransposeExecutor<BlockSpMat<T, Kernel>>::distributed(*this);
    }

    BlockSpMat transpose_bsr() const {
        return detail::BSRTransposeExecutor<BlockSpMat<T, Kernel>>::run(*this);
    }

    void axpby_csr(T alpha, const BlockSpMat& X, T beta) {
        detail::CSRAxpbyExecutor<BlockSpMat<T, Kernel>>::run(*this, X, alpha, beta);
    }

    void axpby_bsr(T alpha, const BlockSpMat& X, T beta) {
        detail::BSRAxpbyExecutor<BlockSpMat<T, Kernel>>::run(*this, X, alpha, beta);
    }

    BlockSpMat spmm_csr(const BlockSpMat& B, double threshold) const;
    BlockSpMat spmm_bsr(const BlockSpMat& B, double threshold) const;

public:

    void allocate_from_graph() {
        attach_backend(build_backend_for_structure(kind, graph, graph->adj_ptr, graph->adj_ind));
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
        
        if (tid >= thread_remote_blocks.size()) {
            // This should rarely happen if max_threads is set correctly at construction
            // But if it does, we can't safely resize. 
            #pragma omp critical
            {
                if (tid >= thread_remote_blocks.size()) {
                    thread_remote_blocks.resize(tid + 1);
                }
            }
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
                pb.mode = AssemblyMode::INSERT;
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
            pb.g_row = global_row;
            pb.g_col = global_col;
            pb.rows = rows;
            pb.cols = cols;
            pb.mode = mode;
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
        if (graph->size == 1) {
            
            norms_valid = false;
            return;
        }
        int size = graph->size;
        
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
                    auto& blk = inner_kv.second;
                    size_t data_bytes = blk.data.size() * sizeof(T);
                    
                    std::memcpy(ptr, &blk.g_row, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(ptr, &blk.g_col, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(ptr, &blk.rows, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(ptr, &blk.cols, sizeof(int)); ptr += sizeof(int);
                    int mode_int = static_cast<int>(blk.mode);
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
        
        for(auto& map : thread_remote_blocks) map.clear();
        norms_valid = false;
    }

    
    // Helper to update local block
    // Input data layout is specified by `layout`
    // Internal storage is ColMajor
    bool update_local_block(int local_row, int local_col, const T* data, int rows, int cols, AssemblyMode mode, MatrixLayout layout = MatrixLayout::ColMajor) {
        int start = row_ptr[local_row];
        int end = row_ptr[local_row+1];
        
        // optimizable
        for (int k = start; k < end; ++k) {
            if (col_ind[k] == local_col) {
                T* target = arena.get_ptr(blk_handles[k]);
                
                // Check dims
                
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
                        // Direct copy: Input (ColMajor) -> Internal (ColMajor)
                        std::memcpy(target, data, r_dim * c_dim * sizeof(T));
                    } else {
                        // Transpose copy: Input (RowMajor) -> Internal (ColMajor)
                        // Internal[j*r_dim + i] = Input[i*c_dim + j]
                        for (int i = 0; i < r_dim; ++i) {
                            for (int j = 0; j < c_dim; ++j) {
                                target[j * r_dim + i] = data[i * c_dim + j];
                            }
                        }
                    }
                } else {
                    // ADD
                    if (layout == MatrixLayout::ColMajor) {
                        // Direct add
                         for (int i = 0; i < r_dim * c_dim; ++i) {
                            target[i] += data[i];
                        }
                    } else {
                        // Transpose copy and add: Internal[j*r_dim + i] += Input[i*c_dim + j]
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
        mult_optimized(x, y);
    }
    
    // Refined mult with offsets
    void mult_optimized(DistVector<T>& x, DistVector<T>& y) {
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        x.sync_ghosts();
        
        int n_rows = row_ptr.size() - 1;
        
        // Use precomputed offsets
        // y_offsets and x_offsets are members

        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int r_dim = graph->block_sizes[i];
            T* y_val = y.local_data() + graph->block_offsets[i];
            
            // Initialize y_val to 0? Or assume y is zeroed?
            // Usually mult implies y = A*x. So overwrite.
            std::memset(y_val, 0, r_dim * sizeof(T));
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {

                // remove to recover org Kernel switch
                if (k + 1 < end) {
                    _mm_prefetch((const char*)(arena.get_ptr(blk_handles[k+1])), _MM_HINT_T0);
                    int next_col = col_ind[k+1];
                    _mm_prefetch((const char*)(x.data.data() + graph->block_offsets[next_col]), _MM_HINT_T0);
                }
                // remove to recover org Kernel switch

                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                const T* block_val = arena.get_ptr(blk_handles[k]);
                const T* x_val = x.data.data() + graph->block_offsets[col]; // x.data includes ghosts
                
                // y_block += A_block * x_block
                SmartKernel<T>::gemv(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_val, 1);
                // Kernel::gemv(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_val, 1);
            }
        }
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
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        X.sync_ghosts();
        
        int n_rows = row_ptr.size() - 1;
        int num_vecs = X.num_vectors;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int r_dim = graph->block_sizes[i];
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            T* y_ptr = &Y(graph->block_offsets[i], 0);
            int ldc = Y.local_rows + Y.ghost_rows;
            
            bool first = true;
            for (int k = start; k < end; ++k) {
                if (k + 1 < end) {
                    _mm_prefetch((const char*)(arena.get_ptr(blk_handles[k+1])), _MM_HINT_T0);
                    int next_col = col_ind[k+1];
                    _mm_prefetch((const char*)(&X(graph->block_offsets[next_col], 0)), _MM_HINT_T0);
                }
                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                const T* block_val = arena.get_ptr(blk_handles[k]);
                const T* x_ptr = &X(graph->block_offsets[col], 0);
                int ldb = X.local_rows + X.ghost_rows;
                
                T beta = first ? T(0) : T(1);
                SmartKernel<T>::gemm(r_dim, num_vecs, c_dim, T(1), block_val, r_dim, x_ptr, ldb, beta, y_ptr, ldc);
                // Kernel::gemm(r_dim, num_vecs, c_dim, T(1), block_val, r_dim, x_ptr, ldb, beta, y_ptr, ldc);
                first = false;
            }
            
            if (first) {
                for (int v = 0; v < num_vecs; ++v) {
                    for (int r = 0; r < r_dim; ++r) {
                        y_ptr[v * ldc + r] = T(0);
                    }
                }
            }
        }
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
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        
        // Initialize y to 0 (including ghosts for accumulation)
        std::fill(y.data.begin(), y.data.end(), T(0));
        
        int n_rows = row_ptr.size() - 1;
        
        // We can't easily parallelize over rows because multiple rows contribute to the same y_j.
        // We need a thread-local y or atomic adds.
        // For simplicity and correctness, let's use a critical section or atomic if T supports it.
        // Or parallelize over rows but use a temporary per-thread y.
        
        #pragma omp parallel
        {
            std::vector<T> y_local(y.data.size(), T(0));
            
            #pragma omp for
            for (int i = 0; i < n_rows; ++i) {
                int r_dim = graph->block_sizes[i];
                const T* x_val = x.local_data() + graph->block_offsets[i];
                
                int start = row_ptr[i];
                int end = row_ptr[i+1];
                
                for (int k = start; k < end; ++k) {
                    int col = col_ind[k];
                    int c_dim = graph->block_sizes[col];
                    const T* block_val = arena.get_ptr(blk_handles[k]);
                    T* y_target = y_local.data() + graph->block_offsets[col];
                    
                    if (k + 1 < end) {
                        _mm_prefetch((const char*)(arena.get_ptr(blk_handles[k+1])), _MM_HINT_T0);
                        int next_col = col_ind[k+1];
                        _mm_prefetch((const char*)(y_local.data() + graph->block_offsets[next_col]), _MM_HINT_T0);
                    }
                    
                    // y_target += A_block^dagger * x_block
                    SmartKernel<T>::gemv_trans(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_target, 1);
                    // Kernel::gemv(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_target, 1, CblasConjTrans);
                }
            }
            
            #pragma omp critical
            {
                for (size_t i = 0; i < y.data.size(); ++i) y.data[i] += y_local[i];
            }
        }
        
        y.reduce_ghosts();
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
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        
        std::fill(Y.data.begin(), Y.data.end(), T(0));
        
        int n_rows = row_ptr.size() - 1;
        int num_vecs = X.num_vectors;
        
        #pragma omp parallel
        {
            std::vector<T> Y_local(Y.data.size(), T(0));
            int ldc_local = Y.local_rows + Y.ghost_rows;
            
            #pragma omp for
            for (int i = 0; i < n_rows; ++i) {
                int r_dim = graph->block_sizes[i];
                const T* x_ptr = &X(graph->block_offsets[i], 0);
                int ldb = X.local_rows + X.ghost_rows;
                
                int start = row_ptr[i];
                int end = row_ptr[i+1];
                
                for (int k = start; k < end; ++k) {
                    int col = col_ind[k];
                    int c_dim = graph->block_sizes[col];
                    const T* block_val = arena.get_ptr(blk_handles[k]);
                    T* y_ptr = &Y_local[col]; // This is wrong for column-major
                    // Correct pointer:
                    T* y_target = &Y_local[graph->block_offsets[col]]; 
                    
                    // Y_target += A_block^dagger * X_block
                    // A_block is r_dim x c_dim. A_block^dagger is c_dim x r_dim.
                    // X_block is r_dim x num_vecs.
                    // Y_target is c_dim x num_vecs.
                    
                    if (k + 1 < end) {
                        _mm_prefetch((const char*)(arena.get_ptr(blk_handles[k+1])), _MM_HINT_T0);
                        int next_col = col_ind[k+1];
                        _mm_prefetch((const char*)(Y_local.data() + graph->block_offsets[next_col]), _MM_HINT_T0);
                    }
                    
                    SmartKernel<T>::gemm_trans(c_dim, num_vecs, r_dim, T(1), block_val, r_dim, x_ptr, ldb, T(1), y_target, ldc_local);
                    // Kernel::gemm(c_dim, num_vecs, r_dim, T(1), block_val, r_dim, x_ptr, ldb, T(1), y_target, ldc_local, CblasConjTrans, CblasNoTrans);
                }
            }
            
            #pragma omp critical
            {
                for (size_t i = 0; i < Y.data.size(); ++i) Y.data[i] += Y_local[i];
            }
        }
        
        Y.reduce_ghosts();
    }

    // Utilities
    void scale(T alpha) {
        #pragma omp parallel for
        for (size_t i = 0; i < blk_handles.size(); ++i) {
            T* block = arena.get_ptr(blk_handles[i]);
            for (size_t j = 0; j < blk_sizes[i]; ++j) {
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
            for (size_t i = 0; i < blk_handles.size(); ++i) {
                T* block = arena.get_ptr(blk_handles[i]);
                for (size_t j = 0; j < blk_sizes[i]; ++j) {
                    block[j] = std::conj(block[j]);
                }
            }
        }
    }

    void copy_from(const BlockSpMat<T, Kernel>& other) {
        // this is used for copying the data from other blocks with the same graph
        // If graphs are different, we should at least check compatibility
        if (graph != other.graph) {
            // Check if owned indices and block sizes match
            if (graph->owned_global_indices != other.graph->owned_global_indices ||
                graph->block_sizes != other.graph->block_sizes) {
                 throw std::runtime_error("Incompatible graph structure in copy_from");
            }
        }

        int n_rows = row_ptr.size() - 1;
        for (int i = 0; i < n_rows; ++i){
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            for (int k = start; k < end; ++k){
                T* block_val = arena.get_ptr(blk_handles[k]);
                T* block_val_other = other.arena.get_ptr(other.blk_handles[k]);
                std::memcpy(block_val, block_val_other, blk_sizes[k] * sizeof(T));
            }
        }
        norms_valid = false;
    }
    

    // Return real part as a new matrix (Double)
    // Only valid if T is complex, otherwise returns copy?
    // Actually we need to return BlockSpMat<RealType>
    auto get_real() const {
        using RealT = typename ScalarTraits<T>::real_type;
        auto res = BlockSpMat<RealT, DefaultKernel<RealT>>::from_parts(
            graph,
            false,
            kind,
            row_ptr,
            col_ind,
            BlockSpMat<RealT, DefaultKernel<RealT>>::build_backend_for_structure(kind, graph, row_ptr, col_ind));
        
        // Copy and cast data
        #pragma omp parallel for
        for (size_t i = 0; i < blk_handles.size(); ++i) {
             RealT* dest = res.mutable_block_data(static_cast<int>(i));
             const T* src = block_data(static_cast<int>(i));
              for(size_t j=0; j<blk_sizes[i]; ++j) {
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
        auto res = BlockSpMat<RealT, DefaultKernel<RealT>>::from_parts(
            graph,
            false,
            kind,
            row_ptr,
            col_ind,
            BlockSpMat<RealT, DefaultKernel<RealT>>::build_backend_for_structure(kind, graph, row_ptr, col_ind));
        
        #pragma omp parallel for
        for (size_t i = 0; i < blk_handles.size(); ++i) {
             RealT* dest = res.mutable_block_data(static_cast<int>(i));
             const T* src = block_data(static_cast<int>(i));
              for(size_t j=0; j<blk_sizes[i]; ++j) {
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
        int start = row_ptr[local_row];
        int end = row_ptr[local_row+1];
        
        for (int k = start; k < end; ++k) {
            if (col_ind[k] == local_col) {
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
        for (size_t s : blk_sizes) total_size += s;
        
        std::vector<T> result(total_size);
        size_t offset = 0;
        
        int n_rows = row_ptr.size() - 1;
        for (int i = 0; i < n_rows; ++i) {
            int r_dim = block_row_dim(i);
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
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
        // Optimization Checks
        if (alpha == T(0)) {
            this->scale(beta);
            return;
        }

        if (this == &X) {
            this->scale(alpha + beta);
            return;
        }

        if (kind == MatrixKind::CSR && X.kind == MatrixKind::CSR) {
            axpby_csr(alpha, X, beta);
            return;
        }
        if (kind == MatrixKind::BSR && X.kind == MatrixKind::BSR) {
            axpby_bsr(alpha, X, beta);
            return;
        }
        detail::LegacyAxpbyExecutor<BlockSpMat<T, Kernel>>::run(*this, X, alpha, beta);
    }

    void axpy(T alpha, const BlockSpMat<T, Kernel>& other) {
        axpby(alpha, other, T(1));
    }

    // Add alpha to diagonal elements
    void shift(T alpha) {
        int n_rows = row_ptr.size() - 1;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            // Find diagonal block (col == local_col corresponding to local_row i)
            // Local row i corresponds to global row G = graph->get_global_index(i)
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            int global_row = graph->get_global_index(i);
            
            for (int k = start; k < end; ++k) {
                int local_col = col_ind[k];
                // Check if this local_col maps to global_row
                
                if (graph->get_global_index(local_col) == global_row) {
                    // Found diagonal block
                    T* target = arena.get_ptr(blk_handles[k]);
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

    // Add vector elements to diagonal: H_ii += v_i
    void add_diagonal(const DistVector<T>& diag) {
        int n_rows = row_ptr.size() - 1;
        // diag must have at least local_size elements
        if (diag.size() < n_rows) {
            throw std::runtime_error("Vector size too small for add_diagonal");
        }

        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            int global_row = graph->get_global_index(i);
            T v_val = diag[i]; // Local index i corresponds to owned part of diag
            
            for (int k = start; k < end; ++k) {
                int local_col = col_ind[k];
                if (graph->get_global_index(local_col) == global_row) {
                    // Found diagonal block
                    T* target = arena.get_ptr(blk_handles[k]);
                    int r_dim = graph->block_sizes[i];
                    int c_dim = graph->block_sizes[local_col];
                    
                    int min_dim = std::min(r_dim, c_dim);
                    for (int j = 0; j < min_dim; ++j) {
                        target[j * r_dim + j] += v_val;
                    }
                    break; 
                }
            }
        }
        norms_valid = false;
    }

    // Compute C = [H, R] where R is diagonal (stored as DistVector)
    // C_ij = H_ij * (R_j - R_i)
    // Result C has same structure as H (this).
    void commutator_diagonal(const DistVector<T>& diag, BlockSpMat<T, Kernel>& result) {
        // Ensure result has same structure
        // if (result.val.size() != val.size()) {
        if (result.blk_handles.size() != blk_handles.size()) {
            result.allocate_from_graph(); // Or throw
        }
        
        const std::vector<T>& R = diag.data; // Includes ghosts
        
        int n_rows = row_ptr.size() - 1;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            // R is a scalar vector. We assume the diagonal operator is constant per block
            // or we just take the first value? 
            // The test implies we treat R as having one value per block.
            // Using the value at the start of the block seems consistent with the test.
            int r_offset_vec = graph->block_offsets[i];
            T R_i = R[r_offset_vec]; 
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_offset_vec = graph->block_offsets[col];
                T R_j = R[c_offset_vec]; 
                
                T diff = R_j - R_i;
                
                int block_size = blk_sizes[k];
                const T* H_ptr = arena.get_ptr(blk_handles[k]);
                T* C_ptr = result.arena.get_ptr(result.blk_handles[k]);
                
                for (int b = 0; b < block_size; ++b) {
                    C_ptr[b] = H_ptr[b] * diff;
                }
            }
        }
    }

    void filter_blocks(double threshold) {
        if (threshold <= 0.0) return;
        
        // Ensure norms are valid (we need them for filtering)
        // need a major refactoring here, the logic will be totally different,
        // we can have much more efficient practice that avoid any reallocation.
        get_block_norms();
        
        // Detach graph if not owned
        if (!owns_graph) {
            graph = graph->duplicate();
            owns_graph = true;
        }
        
        int n_rows = row_ptr.size() - 1;
        std::vector<int> new_row_ptr(n_rows + 1);
        new_row_ptr[0] = 0;
        
        std::vector<int> new_col_ind;
        std::vector<uint64_t> new_blk_handles;
        std::vector<size_t> new_blk_sizes;
        
        // Estimate size to reserve
        new_col_ind.reserve(col_ind.size());
        new_blk_handles.reserve(blk_handles.size());
        new_blk_sizes.reserve(blk_sizes.size());
        
        std::vector<double> new_norms;
        new_norms.reserve(block_norms.size());
        
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                if (block_norms[k] >= threshold) {
                    int col = col_ind[k];
                    new_col_ind.push_back(col);
                    new_blk_handles.push_back(blk_handles[k]);
                    new_blk_sizes.push_back(blk_sizes[k]);
                    new_norms.push_back(block_norms[k]);
                } else {
                    // Drop -> Free
                    arena.free(blk_handles[k], blk_sizes[k]);
                }
            }
            new_row_ptr[i+1] = new_col_ind.size();
        }
        
        // Rebuild Graph to update communication pattern
        // 1. Collect global indices for the new adjacency list
        std::vector<std::vector<int>> new_adj_global(n_rows);
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = new_row_ptr[i];
            int end = new_row_ptr[i+1];
            new_adj_global[i].reserve(end - start);
            for (int k = start; k < end; ++k) {
                int local_col = new_col_ind[k];
                new_adj_global[i].push_back(graph->get_global_index(local_col));
            }
        }
        
        // 2. Construct new DistGraph
        DistGraph* new_graph = new DistGraph(graph->comm);
        // We need to pass the owned block sizes. 
        // graph->block_sizes contains owned + ghosts.
        int n_owned = graph->owned_global_indices.size();
        std::vector<int> owned_block_sizes(n_owned);
        for(int i=0; i<n_owned; ++i) owned_block_sizes[i] = graph->block_sizes[i];
        
        new_graph->construct_distributed(graph->owned_global_indices, owned_block_sizes, new_adj_global);
        
        // 3. Remap col_ind to new local indices
        // new_col_ind currently holds OLD local indices.
        // We need to map: Old Local -> Global -> New Local
        
        #pragma omp parallel for
        for (size_t k = 0; k < new_col_ind.size(); ++k) {
            int old_local = new_col_ind[k];
            int global_col = graph->get_global_index(old_local);
            
            // Use .at() to ensure it exists (throws if not)
            // However, .at() is not const-qualified in all C++ versions? No, it is.
            // But to be safe in OMP, ensure no writes happen to map.
            new_col_ind[k] = new_graph->global_to_local.at(global_col);
        }
        
        blk_handles = std::move(new_blk_handles);
        blk_sizes = std::move(new_blk_sizes);
        block_norms = std::move(new_norms);
        norms_valid = true;

        // 4. Replace Graph
        if (this->owns_graph && this->graph) delete this->graph;
        this->graph = new_graph;
        this->owns_graph = true;
        bind_structure_views();
    }

    // Map: Global Row of B -> List of (Global Col of B, Norm)

    BlockSpMat spmm(const BlockSpMat& B, double threshold, bool transA = false, bool transB = false) const {
        if (transA) {
            BlockSpMat A_T = this->transpose();
            return A_T.spmm(B, threshold, false, transB);
        }

        if (transB) {
            BlockSpMat B_T = B.transpose();
            return this->spmm(B_T, threshold, transA, false);
        }

        if (kind == MatrixKind::CSR && B.kind == MatrixKind::CSR) {
            return spmm_csr(B, threshold);
        }
        if (kind == MatrixKind::BSR && B.kind == MatrixKind::BSR) {
            return spmm_bsr(B, threshold);
        }
        
        return detail::LegacySpMMExecutor<BlockSpMat<T, Kernel>>::run(*this, B, threshold);
    }

    BlockSpMat spmm_self(double threshold, bool transA = false) {
        return spmm(*this, threshold, transA, false);
    }

    BlockSpMat add(const BlockSpMat& B, double alpha = 1.0, double beta = 1.0) {
        if (graph != B.graph) {
             throw std::runtime_error("General addition with different graphs not yet implemented");
        }
        BlockSpMat C = this->duplicate();
        C.scale(alpha);
        C.axpy(beta, B);
        return C;
    }

    void fill(T val) {
        #pragma omp parallel for
        for (int i = 0; i < col_ind.size(); ++i) {
            uint64_t handle = blk_handles[i];
            T* data = arena.get_ptr(handle);
            size_t size = blk_sizes[i];
            std::fill(data, data + size, val);
        }
        norms_valid = false;
    }

    BlockSpMat transpose() const {
        if (kind == MatrixKind::CSR) {
            return transpose_csr();
        }
        if (kind == MatrixKind::BSR) {
            return transpose_bsr();
        }

        return detail::LegacyTransposeExecutor<BlockSpMat<T, Kernel>>::run(*this);
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
        
        int n_owned = row_ptr.size() - 1;
        
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
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
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
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                int col_start_idx = graph->block_offsets[col] + 1; // 1-based
                
                // long long offset = blk_ptr[k];
                const T* block_data = arena.get_ptr(blk_handles[k]);
                
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

public:
    using GhostBlockRef = detail::GhostBlockRef<T>;
    using GhostBlockData = detail::GhostBlockData<T>;
    using GhostSizes = detail::GhostSizes;
    using GhostMetadata = detail::GhostMetadata;

    struct SymbolicResult {
        std::vector<int> c_row_ptr;
        std::vector<int> c_col_ind;
        std::vector<BlockID> required_blocks;
    };

    // SpMM Phase 1: Metadata Exchange
    GhostMetadata exchange_ghost_metadata(const BlockSpMat& B) const {
        return detail::exchange_ghost_metadata(*this, B);
    }

    // SpMM Phase 2: Symbolic Multiplication
    SymbolicResult symbolic_multiply_filtered(const BlockSpMat& B, const GhostMetadata& meta, double threshold) const {
        SymbolicResult res;
        int n_rows = row_ptr.size() - 1;
        res.c_row_ptr.resize(n_rows + 1);
        res.c_row_ptr[0] = 0;
        
        std::vector<double> A_norms = compute_block_norms();
        std::vector<double> B_local_norms = B.compute_block_norms();
        
        std::vector<std::vector<int>> thread_cols(n_rows);
        int max_threads = omp_get_max_threads();
        std::vector<std::set<BlockID>> thread_required(max_threads);

        struct SymbolicHashEntry {
            int key;
            double value;
            int tag;
        };
        const size_t HASH_SIZE = 8192;
        const size_t HASH_MASK = HASH_SIZE - 1;
        const size_t MAX_ROW_NNZ = static_cast<size_t>(HASH_SIZE * 0.7);

        std::vector<std::vector<SymbolicHashEntry>> thread_tables(max_threads, std::vector<SymbolicHashEntry>(HASH_SIZE, {-1, 0.0, 0}));
        std::vector<std::vector<int>> thread_touched(max_threads);
        std::vector<int> thread_tags(max_threads, 0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& table = thread_tables[tid];
            auto& touched = thread_touched[tid];
            int& tag = thread_tags[tid];
            
            #pragma omp for
            for (int i = 0; i < n_rows; ++i) {
                tag++;
                if (tag == 0) {
                    for(auto& e : table) e.tag = 0;
                    tag = 1;
                }
                touched.clear();
                
                int start = row_ptr[i];
                int end = row_ptr[i+1];
                
                for (int k = start; k < end; ++k) {
                    int global_col_A = graph->get_global_index(col_ind[k]);
                    double norm_A = A_norms[k];
                    
                    auto process_block = [&](int g_col_B, double norm_B) {
                        size_t h = (size_t)g_col_B & HASH_MASK;
                        size_t count = 0;
                        while (table[h].tag == tag) {
                            if (table[h].key == g_col_B) {
                                table[h].value += norm_A * norm_B;
                                return;
                            }
                            h = (h + 1) & HASH_MASK;
                            if (++count > HASH_SIZE) {
                                throw std::runtime_error("Hash table full in symbolic phase");
                            }
                        }
                        if (touched.size() > MAX_ROW_NNZ) {
                            throw std::runtime_error("Row density exceeds symbolic hash table capacity");
                        }
                        table[h] = {g_col_B, norm_A * norm_B, tag};
                        touched.push_back(h);
                    };

                    if (graph->find_owner(global_col_A) == graph->rank) {
                        int local_row_B = graph->global_to_local.at(global_col_A);
                        int start_B = B.row_ptr[local_row_B];
                        int end_B = B.row_ptr[local_row_B+1];
                        for (int j = start_B; j < end_B; ++j) {
                            process_block(B.graph->get_global_index(B.col_ind[j]), B_local_norms[j]);
                        }
                    } else {
                        auto it = meta.find(global_col_A);
                        if (it != meta.end()) {
                            for (const auto& m : it->second) {
                                process_block(m.col, m.norm);
                            }
                        }
                    }
                }
                
                for (int h_idx : touched) {
                    if (table[h_idx].value > threshold) {
                        thread_cols[i].push_back(table[h_idx].key);
                    }
                }
                std::sort(thread_cols[i].begin(), thread_cols[i].end());
            }
        }
        
        for(int i=0; i<n_rows; ++i) {
            res.c_col_ind.insert(res.c_col_ind.end(), thread_cols[i].begin(), thread_cols[i].end());
            res.c_row_ptr[i+1] = res.c_col_ind.size();
        }
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for
            for (int i = 0; i < n_rows; ++i) {
                int c_start = res.c_row_ptr[i];
                int c_end = res.c_row_ptr[i+1];
                if (c_start == c_end) continue;
                
                int start = row_ptr[i];
                int end = row_ptr[i+1];
                for (int k = start; k < end; ++k) {
                    int global_col_A = graph->get_global_index(col_ind[k]);
                    
                    if (graph->find_owner(global_col_A) != graph->rank) {
                        auto it = meta.find(global_col_A);
                        if (it != meta.end()) {
                            for (const auto& m : it->second) {
                                if (std::binary_search(res.c_col_ind.begin() + c_start, res.c_col_ind.begin() + c_end, m.col)) {
                                    thread_required[tid].insert({global_col_A, m.col});
                                }
                            }
                        }
                    }
                }
            }
        }
        
        std::set<BlockID> final_required;
        for(auto& s : thread_required) final_required.insert(s.begin(), s.end());
        res.required_blocks.assign(final_required.begin(), final_required.end());
        
        return res;
    }

    // SpMM Phase 3: Fetch Ghost Blocks
    std::pair<GhostBlockData, GhostSizes> fetch_ghost_blocks(const std::vector<BlockID>& required_blocks) const {
        return detail::fetch_ghost_blocks(*this, required_blocks);
    }

    void numeric_multiply(
        const BlockSpMat& B,
        const std::map<int, std::vector<GhostBlockRef>>& ghost_rows,
        BlockSpMat& C,
        double threshold,
        const std::vector<double>& A_norms,
        const std::vector<double>& B_local_norms) const {
        detail::LegacySpMMExecutor<BlockSpMat<T, Kernel>>::run_numeric(
            *this,
            B,
            ghost_rows,
            C,
            threshold,
            A_norms,
            B_local_norms);
    }

           // Data structure for holding a block with global indices
    struct BlockData {
        int global_row;
        int global_col;
        int r_dim;
        int c_dim;
        std::vector<T> data;
    };

    // Context holding fetched data (blocks and row sizes)
    struct FetchContext {
        std::vector<BlockData> blocks;
        std::map<int, int> row_sizes; // global_row -> block_size
    };

    // Helper to serve fetch requests from other processes
    // req_buffer: [NumRows, (RowGID, NumCols, ColGID...)...]
    // resp_buffer: Output buffer [TotalBlocks, (RowGID, Size)..., (RowGID, ColGID, RDim, CDim, Data)...]
    // Note: To simplify unpacking, we will structure response as:
    // [NumRows, (RowGID, Size)..., NumBlocks, (RowGID, ColGID, RDim, CDim, Data)...]
    void serve_fetch_requests(const char* req_buffer, std::vector<char>& resp_buffer) {
        const int* ptr = reinterpret_cast<const int*>(req_buffer);
        int num_rows = *ptr++;
        
        // We need to iterate requests twice: once for sizes, once for blocks.
        // Save start pointer.
        const int* req_start = ptr;
        
        // 1. Calculate Response Size and Pack Row Sizes
        // Header: NumRows + (RowGID, Size)*NumRows + NumBlocks
        size_t header_size = sizeof(int) + num_rows * 2 * sizeof(int) + sizeof(int);
        
        // Temporary storage for blocks to avoid double scanning or complex size prediction
        // But double scanning is fine for memory efficiency if we don't copy data yet.
        // Let's do two passes.
        
        resp_buffer.resize(header_size);
        char* buf_ptr = resp_buffer.data();
        
        // Write NumRows
        std::memcpy(buf_ptr, &num_rows, sizeof(int)); buf_ptr += sizeof(int);
        
        // Pass 1: Write Row Sizes
        ptr = req_start;
        for(int r=0; r<num_rows; ++r) {
            int gid = *ptr++;
            int num_cols = *ptr++;
            ptr += num_cols; // Skip cols
            
            int size = 0;
            if(graph->global_to_local.count(gid)) {
                int lid = graph->global_to_local.at(gid);
                size = graph->block_sizes[lid];
            }
            std::memcpy(buf_ptr, &gid, sizeof(int)); buf_ptr += sizeof(int);
            std::memcpy(buf_ptr, &size, sizeof(int)); buf_ptr += sizeof(int);
        }
        
        // Pass 2: Collect and Pack Blocks
        int total_blocks = 0;
        ptr = req_start;
        
        for(int r=0; r<num_rows; ++r) {
            int gid = *ptr++;
            int num_cols = *ptr++;
            std::set<int> req_cols(ptr, ptr + num_cols); ptr += num_cols;
            
            if(graph->global_to_local.count(gid)) {
                int lid = graph->global_to_local.at(gid);
                int start = row_ptr[lid];
                int end = row_ptr[lid+1];
                
                for(int k=start; k<end; ++k) {
                    int col_lid = col_ind[k];
                    int col_gid = graph->get_global_index(col_lid);
                    
                    if(req_cols.count(col_gid)) {
                        total_blocks++;
                        int r_dim = graph->block_sizes[lid];
                        int c_dim = graph->block_sizes[col_lid];
                        size_t size = blk_sizes[k];
                        
                        size_t old_size = resp_buffer.size();
                        resp_buffer.resize(old_size + 4*sizeof(int) + size*sizeof(T));
                        char* b_ptr = resp_buffer.data() + old_size;
                        
                        std::memcpy(b_ptr, &gid, sizeof(int)); b_ptr += sizeof(int);
                        std::memcpy(b_ptr, &col_gid, sizeof(int)); b_ptr += sizeof(int);
                        std::memcpy(b_ptr, &r_dim, sizeof(int)); b_ptr += sizeof(int);
                        std::memcpy(b_ptr, &c_dim, sizeof(int)); b_ptr += sizeof(int);
                        std::memcpy(b_ptr, arena.get_ptr(blk_handles[k]), size*sizeof(T));
                    }
                }
            }
        }
        
        // Write NumBlocks (at the end of the header section)
        // Header structure: [NumRows] [(GID, Size)...] [NumBlocks]
        // Offset for NumBlocks is: sizeof(int) + num_rows * 2 * sizeof(int)
        size_t num_blocks_offset = sizeof(int) + num_rows * 2 * sizeof(int);
        std::memcpy(resp_buffer.data() + num_blocks_offset, &total_blocks, sizeof(int));
    }

    // Fetch blocks for a batch of submatrices
    FetchContext fetch_blocks(const std::vector<std::vector<int>>& batch_indices) {
        int rank = graph->rank;
        FetchContext ctx;
        
        // 1. Analyze Requirements
        std::set<int> all_required_rows;
        for(const auto& indices : batch_indices) {
            all_required_rows.insert(indices.begin(), indices.end());
        }
        
        // 2. Identify Local vs Remote
        std::vector<int> local_rows;
        std::map<int, std::vector<int>> remote_rows_by_rank;
        
        for(int gid : all_required_rows) {
            int owner = graph->find_owner(gid);
            if(owner == graph->rank) {
                local_rows.push_back(gid);
            } else {
                remote_rows_by_rank[owner].push_back(gid);
            }
        }

        // Map global_row -> set of required global_cols
        std::map<int, std::set<int>> required_cols_per_row;
        for(const auto& indices : batch_indices) {
            for(int row_gid : indices) {
                required_cols_per_row[row_gid].insert(indices.begin(), indices.end());
            }
        }

        // 3. Local Fetch
        for(int gid : local_rows) {
            if(graph->global_to_local.find(gid) == graph->global_to_local.end()) continue;
            int lid = graph->global_to_local.at(gid);
            
            ctx.row_sizes[gid] = graph->block_sizes[lid];
            
            int start = row_ptr[lid];
            int end = row_ptr[lid+1];
            const auto& req_cols = required_cols_per_row[gid];
            
            for(int k=start; k<end; ++k) {
                int col_lid = col_ind[k];
                int col_gid = graph->get_global_index(col_lid);
                
                if(req_cols.count(col_gid)) {
                    BlockData bd;
                    bd.global_row = gid;
                    bd.global_col = col_gid;
                    bd.r_dim = graph->block_sizes[lid];
                    bd.c_dim = graph->block_sizes[col_lid];
                    
                    // long long offset = blk_ptr[k];
                    // size_t size = blk_ptr[k+1] - offset;
                    size_t size = blk_sizes[k];
                    bd.data.resize(size);
                    std::memcpy(bd.data.data(), arena.get_ptr(blk_handles[k]), size * sizeof(T));
                    
                    ctx.blocks.push_back(std::move(bd));
                }
            }
        }
        
        // 4. Remote Fetch
        // Prepare Requests
        std::vector<size_t> send_counts(graph->size, 0);
        std::vector<std::vector<int>> send_buffers(graph->size);
        
        for(auto& kv : remote_rows_by_rank) {
            int target = kv.first;
            auto& rows = kv.second;
            
            send_buffers[target].push_back(rows.size());
            for(int gid : rows) {
                send_buffers[target].push_back(gid);
                const auto& cols = required_cols_per_row[gid];
                send_buffers[target].push_back(cols.size());
                send_buffers[target].insert(send_buffers[target].end(), cols.begin(), cols.end());
            }
            send_counts[target] = send_buffers[target].size() * sizeof(int);
        }
        
        // Exchange Counts
        std::vector<size_t> recv_counts(graph->size);
        if (graph->size > 1) {
            MPI_Alltoall(send_counts.data(), sizeof(size_t), MPI_BYTE, recv_counts.data(), sizeof(size_t), MPI_BYTE, graph->comm);
        } else {
            recv_counts = send_counts;
        }
        
        // Exchange Requests
        std::vector<size_t> sdispls(graph->size + 1, 0);
        std::vector<size_t> rdispls(graph->size + 1, 0);
        for(int i=0; i<graph->size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
        }
        
        std::vector<char> send_blob(sdispls[graph->size]);
        for(int i=0; i<graph->size; ++i) {
             if (!send_buffers[i].empty())
                std::memcpy(send_blob.data() + sdispls[i], send_buffers[i].data(), send_buffers[i].size() * sizeof(int));
        }
        
        std::vector<char> recv_blob(rdispls[graph->size]);
        if (graph->size > 1) {
            safe_alltoallv(send_blob.data(), send_counts, sdispls, MPI_BYTE,
                          recv_blob.data(), recv_counts, rdispls, MPI_BYTE, graph->comm);
        } else {
            recv_blob = send_blob;
        }
        
        // Serve Requests
        std::vector<std::vector<char>> resp_buffers(graph->size);
        std::vector<size_t> resp_send_counts(graph->size, 0);
        
        for(int i=0; i<graph->size; ++i) {
            if(recv_counts[i] == 0) continue;
            serve_fetch_requests(recv_blob.data() + rdispls[i], resp_buffers[i]);
            resp_send_counts[i] = resp_buffers[i].size();
        }
        
        // Exchange Responses
        std::vector<size_t> resp_recv_counts(graph->size);
        if (graph->size > 1) {
            MPI_Alltoall(resp_send_counts.data(), sizeof(size_t), MPI_BYTE, resp_recv_counts.data(), sizeof(size_t), MPI_BYTE, graph->comm);
        } else {
            resp_recv_counts = resp_send_counts;
        }
        
        std::vector<size_t> resp_sdispls(graph->size + 1, 0);
        std::vector<size_t> resp_rdispls(graph->size + 1, 0);
        for(int i=0; i<graph->size; ++i) {
            resp_sdispls[i+1] = resp_sdispls[i] + resp_send_counts[i];
            resp_rdispls[i+1] = resp_rdispls[i] + resp_recv_counts[i];
        }
        
        std::vector<char> resp_send_blob(resp_sdispls[graph->size]);
        for(int i=0; i<graph->size; ++i) {
            if(!resp_buffers[i].empty()) {
                std::memcpy(resp_send_blob.data() + resp_sdispls[i], resp_buffers[i].data(), resp_buffers[i].size());
            }
        }
        
        std::vector<char> resp_recv_blob(resp_rdispls[graph->size]);
        if (graph->size > 1) {
            safe_alltoallv(resp_send_blob.data(), resp_send_counts, resp_sdispls, MPI_BYTE,
                          resp_recv_blob.data(), resp_recv_counts, resp_rdispls, MPI_BYTE, graph->comm);
        } else {
            resp_recv_blob = resp_send_blob;
        }
        
        // Unpack Responses
        for(int i=0; i<graph->size; ++i) {
            if(resp_recv_counts[i] == 0) continue;
            
            const char* ptr = resp_recv_blob.data() + resp_rdispls[i];
            
            // 1. Read Sizes
            int num_rows;
            std::memcpy(&num_rows, ptr, sizeof(int)); ptr += sizeof(int);
            for(int k=0; k<num_rows; ++k) {
                int gid, size;
                std::memcpy(&gid, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&size, ptr, sizeof(int)); ptr += sizeof(int);
                ctx.row_sizes[gid] = size;
            }
            
            // 2. Read Blocks
            int num_blocks;
            std::memcpy(&num_blocks, ptr, sizeof(int)); ptr += sizeof(int);
            
            for(int k=0; k<num_blocks; ++k) {
                int gid, col_gid, r_dim, c_dim;
                std::memcpy(&gid, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&col_gid, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&r_dim, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&c_dim, ptr, sizeof(int)); ptr += sizeof(int);
                
                BlockData bd;
                bd.global_row = gid;
                bd.global_col = col_gid;
                bd.r_dim = r_dim;
                bd.c_dim = c_dim;
                bd.data.resize(r_dim * c_dim);
                std::memcpy(bd.data.data(), ptr, bd.data.size() * sizeof(T)); ptr += bd.data.size() * sizeof(T);
                
                ctx.blocks.push_back(std::move(bd));
            }
        }
        
        return ctx;
    }

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
        auto ctx = fetch_blocks({global_indices});
        return construct_submatrix(global_indices, ctx);
    }

    // Extract multiple submatrices efficiently
    std::vector<BlockSpMat<T, Kernel>> extract_submatrix_batched(const std::vector<std::vector<int>>& batch_indices) {
        auto ctx = fetch_blocks(batch_indices);
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
        int n_rows = submat.row_ptr.size() - 1;
        for(int i=0; i<n_rows; ++i) {
            int r_dim = submat.graph->block_sizes[i];
            int start = submat.row_ptr[i];
            int end = submat.row_ptr[i+1];
            
            int global_row = global_indices[i];
            
            for(int k=start; k<end; ++k) {
                int col = submat.col_ind[k];
                int c_dim = submat.graph->block_sizes[col];
                int global_col = global_indices[col];
                
                const T* data = submat.arena.get_ptr(submat.blk_handles[k]);
                
                // Use add_block with INSERT mode. 
                // It handles local update and remote buffering.
                // Data is in ColMajor (internal storage of submat).
                this->add_block(global_row, global_col, data, r_dim, c_dim, AssemblyMode::INSERT, MatrixLayout::ColMajor);
            }
        }
        
        // Flush remote updates
        this->assemble();
    }

    // Convert to dense (Row-Major)
    // Convert to dense (Row-Major)
    // Returns dense matrix of size (owned_rows) x (all_local_cols)
    // This includes owned columns AND ghost columns present locally.
    // The columns are ordered by their local index (owned first, then ghosts).
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
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for(int k=start; k<end; ++k) {
                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                int col_offset = graph->block_offsets[col];
                
                // const T* data = val.data() + blk_ptr[k];
                const T* data = arena.get_ptr(blk_handles[k]);
                
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
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for(int k=start; k<end; ++k) {
                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                int col_offset = graph->block_offsets[col];
                
                // T* data = val.data() + blk_ptr[k];
                T* data = arena.get_ptr(blk_handles[k]);
                
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
        long long local_nnz = col_ind.size();
        long long global_nnz = 0;
        
        if (graph->size > 1) {
            MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_LONG_LONG, MPI_SUM, graph->comm);
        } else {
            global_nnz = local_nnz;
        }
        
        // Total global blocks N
        // graph->block_displs is size+1, last element is total blocks
        if (graph->block_displs.empty()) {
             // Should not happen if constructed, but safety check
             return 0.0;
        }
        long long N = graph->block_displs.back();
        
        if (N == 0) return 0.0;
        
        double density = (double)global_nnz / (double)(N * N);
        return density;
    }
};

#include "detail/axpby_ops.hpp"
#include "detail/bsr_spmm.hpp"
#include "detail/csr_spmm.hpp"
#include "detail/legacy_spmm.hpp"
#include "detail/transpose_ops.hpp"

template <typename T, typename Kernel>
BlockSpMat<T, Kernel> BlockSpMat<T, Kernel>::spmm_csr(const BlockSpMat& B, double threshold) const {
    return detail::CSRSpMMExecutor<BlockSpMat<T, Kernel>>::run(*this, B, threshold);
}

template <typename T, typename Kernel>
BlockSpMat<T, Kernel> BlockSpMat<T, Kernel>::spmm_bsr(const BlockSpMat& B, double threshold) const {
    return detail::BSRSpMMExecutor<BlockSpMat<T, Kernel>>::run(*this, B, threshold);
}

template <typename T, typename Kernel>
BlockSpMat<T, Kernel> LegacyMatrixBuilder<T, Kernel>::materialize() const {
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    graph_->get_matrix_structure(row_ptr, col_ind);
    const MatrixKind kind = BlockSpMat<T, Kernel>::detect_matrix_kind(graph_);
    auto backend = BlockSpMat<T, Kernel>::build_backend_for_structure(kind, graph_, row_ptr, col_ind);
    return BlockSpMat<T, Kernel>::from_parts(
        graph_,
        owns_graph_,
        kind,
        std::move(row_ptr),
        std::move(col_ind),
        std::move(backend));
}

template <typename T, typename Kernel>
void LegacyMatrixBuilder<T, Kernel>::write_transposed_conjugate_slot(
    BlockSpMat<T, Kernel>& matrix,
    int slot,
    const T* src,
    int src_rows,
    int src_cols) const {
    T* dest = matrix.mutable_block_data(slot);
    for (int col = 0; col < src_cols; ++col) {
        for (int row = 0; row < src_rows; ++row) {
            const T val = src[col * src_rows + row];
            dest[row * src_cols + col] = ConjHelper<T>::apply(val);
        }
    }
}

} // namespace vbcsr

#endif
