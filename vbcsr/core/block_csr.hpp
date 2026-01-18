#ifndef VBCSR_BLOCK_CSR_HPP
#define VBCSR_BLOCK_CSR_HPP

#include "dist_graph.hpp"
#include "dist_vector.hpp"
#include "dist_multivector.hpp"
#include "kernels.hpp"
#include "block_memory_pool.hpp" // Added
#include "mpi_utils.hpp"
#include <xmmintrin.h>
#include <vector>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <complex>
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

struct BlockMeta {
    int col;
    double norm;
};

struct BlockID {
    int row;
    int col;
    bool operator<(const BlockID& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

template <typename T, typename Kernel = DefaultKernel<T>>
class BlockSpMat {
public:
    DistGraph* graph;
    using KernelType = Kernel;
    
    // CSR structure (local indices)
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    
    // Values (flattened blocks)
    std::vector<uint64_t> blk_handles;
    std::vector<size_t> blk_sizes;
    BlockArena<T> arena;
    
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
    std::map<int, std::map<std::pair<int, int>, PendingBlock>> remote_blocks;

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
        int nnz = col_ind.size();
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

public:
    BlockSpMat(DistGraph* g) : graph(g) {
        allocate_from_graph();
    }

    ~BlockSpMat() {
        if (owns_graph && graph) {
            delete graph;
        }
    }

    // Move constructor
    BlockSpMat(BlockSpMat&& other) noexcept : 
        graph(other.graph), 
        row_ptr(std::move(other.row_ptr)),
        col_ind(std::move(other.col_ind)),
        blk_handles(std::move(other.blk_handles)),
        blk_sizes(std::move(other.blk_sizes)),
        arena(std::move(other.arena)),
        remote_blocks(std::move(other.remote_blocks)),
        owns_graph(other.owns_graph),

        block_norms(std::move(other.block_norms)),
        norms_valid(other.norms_valid)
    {
        other.graph = nullptr;
        other.owns_graph = false;
    }

    // Move assignment
    BlockSpMat& operator=(BlockSpMat&& other) noexcept {
        if (this != &other) {
            if (owns_graph && graph) delete graph;
            graph = other.graph;
            row_ptr = std::move(other.row_ptr);
            col_ind = std::move(other.col_ind);
            blk_handles = std::move(other.blk_handles);
            blk_sizes = std::move(other.blk_sizes);
            arena = std::move(other.arena);
            remote_blocks = std::move(other.remote_blocks);
            owns_graph = other.owns_graph;
            
            // Fix: Move norms state
            block_norms = std::move(other.block_norms);
            norms_valid = other.norms_valid;
            
            other.graph = nullptr;
            other.owns_graph = false;
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
        BlockSpMat<T, Kernel> new_mat(new_graph);
        new_mat.owns_graph = new_owns_graph;
        new_mat.copy_from(*this);
        if (norms_valid) {
            new_mat.block_norms = block_norms;
            new_mat.norms_valid = true;
        }
        return new_mat;
    }

    void allocate_from_graph() {
        // 1. Get CSR structure
        graph->get_matrix_structure(row_ptr, col_ind);
        
        // 2. Cache dimensions - REMOVED, use graph->block_sizes directly
        // row_dims = graph->block_sizes (prefix)
        // col_dims = graph->block_sizes (all)
        
        // 2.1 Compute offsets - REMOVED, use graph->block_offsets directly
        // y_offsets = graph->block_offsets (prefix)
        // x_offsets = graph->block_offsets (all)
        
        // 3. Calculate value size and block pointers
        int nnz = col_ind.size();
        blk_handles.resize(nnz);
        blk_sizes.resize(nnz);

        unsigned long long total_elements = 0;
        int n_owned = graph->owned_global_indices.size();
        
        // First pass: calculate size
        #pragma omp parallel for reduction(+ : total_elements)
        for (int i = 0; i < n_owned; ++i) {
            int r_dim = graph->block_sizes[i];
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            int row_total_elements = 0;
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                row_total_elements += (unsigned long long)r_dim * c_dim;
            }
            total_elements += row_total_elements;
        }
        
        arena.reserve(total_elements);
        
        // Second pass: allocate handles (Serial to ensure thread safety of arena.allocate)
        for (int i = 0; i < n_owned; ++i) {

            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int r_dim = graph->block_sizes[i];
                int c_dim = graph->block_sizes[col];
                int sz = r_dim * c_dim;
                blk_handles[k] = arena.allocate(sz);
                blk_sizes[k] = sz;
            }
        }
        
        norms_valid = false;
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

        auto& blocks_map = remote_blocks[owner];
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
        int size = graph->size;
        
        // 1. Counting pass
        std::vector<size_t> send_counts(size, 0);
        for (const auto& kv : remote_blocks) {
            int target = kv.first;
            size_t bytes = 0;
            for (const auto& inner_kv : kv.second) {
                bytes += 5 * sizeof(int) + inner_kv.second.data.size() * sizeof(T);
            }
            send_counts[target] = bytes;
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
        for (auto& kv : remote_blocks) {
            int target = kv.first;
            char* ptr = send_blob.data() + sdispls[target];
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
        
        remote_blocks.clear();
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

    void axpby(T alpha, const BlockSpMat<T, Kernel>& X, T beta) {
        // Optimization Checks
        if (alpha == T(0)) {
            this->scale(beta);
            return;
        }

        if (beta == T(0)) {
            if (this->graph == X.graph) {
                // Same graph, just copy values and scale
                #pragma omp parallel for
                for (size_t i = 0; i < this->blk_handles.size(); ++i) {
                    T* block = this->arena.get_ptr(this->blk_handles[i]);
                    const T* block_x = X.arena.get_ptr(X.blk_handles[i]);
                    for (size_t j = 0; j < this->blk_sizes[i]; ++j) {
                        block[j] = alpha * block_x[j];
                    }
                }
                this->block_norms = X.block_norms;
                this->norms_valid = false;
                bool update_norm = false;
                double alpha_real = 0;

                if (X.norms_valid) {
                    if (alpha == T(1)) {
                        this->norms_valid = true;
                    }

                    if (!this->norms_valid) {
                        update_norm = true;
                        if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
                            if (alpha.imag() != 0.0) {
                                update_norm = false;
                            } else {
                                alpha_real = alpha.real();
                            }
                        } else {
                            alpha_real = alpha;
                        }
                    }
                        
                    if (update_norm) {
                        #pragma omp parallel for
                        for (size_t i = 0; i < this->block_norms.size(); ++i) {
                            this->block_norms[i] = X.block_norms[i] * alpha_real;
                        }
                        this->norms_valid = true;
                    }
                }
                return;
            } else {
                 // Check if structures are identical
                bool same_structure = (this->row_ptr == X.row_ptr && this->col_ind == X.col_ind);
                if (same_structure) {
                    // If structure (topology) is same but graphs differ, block sizes might differ.
                    // if (this->val.size() != X.val.size()) {
                    //     throw std::runtime_error("Matrix dimension mismatch in axpby (same topology but different block sizes)");
                    // }
                    #pragma omp parallel for
                    for (size_t i = 0; i < this->blk_handles.size(); ++i) {
                        T* block = this->arena.get_ptr(this->blk_handles[i]);
                        const T* block_x = X.arena.get_ptr(X.blk_handles[i]);
                        for (size_t j = 0; j < this->blk_sizes[i]; ++j) {
                            block[j] = alpha * block_x[j];
                        }
                    }
                    this->block_norms = X.block_norms;
                    this->norms_valid = false;
                    bool update_norm = false;
                    double alpha_real = 0;

                    if (X.norms_valid) {
                        if (alpha == T(1)) {
                            this->norms_valid = true;
                        }

                        if (!this->norms_valid) {
                            update_norm = true;
                            if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
                                if (alpha.imag() != 0.0) {
                                    update_norm = false;
                                } else {
                                    alpha_real = alpha.real();
                                }
                            } else {
                                alpha_real = alpha;
                            }
                        }
                        
                        if (update_norm) {
                            #pragma omp parallel for
                            for (size_t i = 0; i < this->block_norms.size(); ++i) {
                                this->block_norms[i] = X.block_norms[i] * alpha_real;
                            }
                            this->norms_valid = true;
                        }
                    }
                    return;
                } else {
                    // Reallocation Required
                    // optimizable, we can just copy the data using the available mem without reallocation,
                    // and allocate only needed space, and then do iterative copy.
                    *this = X.duplicate(); // Deep copy
                    this->scale(alpha);
                    return;
                }
            }
        }
        
        if (this == &X) {
            this->scale(alpha + beta);
            return;
        }

        // Structure Comparison & Execution
        
        // Step 1: Fast Path (Same Object/Graph)
        bool same_graph = (this->graph == X.graph);
        bool same_structure = false;
        if (same_graph) {
             if (this->row_ptr.size() == X.row_ptr.size() && this->col_ind.size() == X.col_ind.size()) {
                 if (this->row_ptr == X.row_ptr && this->col_ind == X.col_ind) {
                     same_structure = true;
                 }
             }
        }

        if (same_structure) {
            #pragma omp parallel for
            for (size_t i = 0; i < this->blk_handles.size(); ++i) {
                T* block = this->arena.get_ptr(this->blk_handles[i]);
                const T* block_x = X.arena.get_ptr(X.blk_handles[i]);
                for (size_t j = 0; j < this->blk_sizes[i]; ++j) {
                    block[j] = alpha * block_x[j] + beta * block[j];
                }
            }
            this->norms_valid = false;
            return;
        }

        // Step 2: Robust Merge with Global Indices
        int n_rows = this->row_ptr.size() - 1;
        if (X.row_ptr.size() - 1 != n_rows) {
            throw std::runtime_error("Matrix row count mismatch in axpby");
        }

        // Build Translation Table: X_local -> this_local
        // If X has a column that this doesn't have, map to -1.
        int x_n_owned = X.graph->owned_global_indices.size();
        int x_n_ghost = X.graph->ghost_global_indices.size();
        int x_total_cols = x_n_owned + x_n_ghost;
        
        std::vector<int> x_to_this(x_total_cols, -1);
        bool x_is_subset = true;
        
        // Owned
        for(int i=0; i<x_n_owned; ++i) {
            int gid = X.graph->owned_global_indices[i];
            if (this->graph->global_to_local.count(gid)) {
                x_to_this[i] = this->graph->global_to_local.at(gid);
            } else {
                x_is_subset = false;
            }
        }
        // Ghost
        for(int i=0; i<x_n_ghost; ++i) {
            int gid = X.graph->ghost_global_indices[i];
            if (this->graph->global_to_local.count(gid)) {
                x_to_this[x_n_owned + i] = this->graph->global_to_local.at(gid);
            } else {
                x_is_subset = false;
            }
        }
        
        int local_subset = x_is_subset ? 1 : 0;
        int global_subset = 0;
        MPI_Allreduce(&local_subset, &global_subset, 1, MPI_INT, MPI_MIN, this->graph->comm);
        x_is_subset = (global_subset == 1);
        
        if (x_is_subset) {
            // Check sparsity subset
            bool sparsity_subset = true;
            #pragma omp parallel for reduction(&&:sparsity_subset)
            for (int i = 0; i < n_rows; ++i) {
                if (!sparsity_subset) continue;
                int y_start = this->row_ptr[i];
                int y_end = this->row_ptr[i+1];
                int x_start = X.row_ptr[i];
                int x_end = X.row_ptr[i+1];
                
                int y_k = y_start;
                for (int x_k = x_start; x_k < x_end; ++x_k) {
                    int x_col_local = X.col_ind[x_k];
                    int target_col = x_to_this[x_col_local];
                    
                    while (y_k < y_end && this->col_ind[y_k] < target_col) {
                        y_k++;
                    }
                    if (y_k == y_end || this->col_ind[y_k] != target_col) {
                        sparsity_subset = false;
                        break;
                    }
                }
            }
            
            // Check globally if we can use the fast path (subset)
            int local_ss = sparsity_subset ? 1 : 0;
            int global_ss = 0;
            MPI_Allreduce(&local_ss, &global_ss, 1, MPI_INT, MPI_MIN, this->graph->comm);
            // TODO: optimizable, seems we can merge if the local_ss is 1, and for mismatched case, we only need to reallocate locally
            // this is very tricky.
            if (global_ss == 1) {
                // Safe to proceed with in-place addition
                this->scale(beta);
                #pragma omp parallel for
                for (int i = 0; i < n_rows; ++i) {
                    int y_start = this->row_ptr[i];
                    int y_end = this->row_ptr[i+1];
                    int x_start = X.row_ptr[i];
                    int x_end = X.row_ptr[i+1];
                    
                    int y_k = y_start;
                    for (int x_k = x_start; x_k < x_end; ++x_k) {
                        int x_col_local = X.col_ind[x_k];
                        int target_col = x_to_this[x_col_local];
                        
                        while (y_k < y_end && this->col_ind[y_k] < target_col) {
                            y_k++;
                        }
                        // Guaranteed found
                        T* y_ptr = this->arena.get_ptr(this->blk_handles[y_k]);
                        T* x_ptr = X.arena.get_ptr(X.blk_handles[x_k]);
                        int size = this->blk_sizes[y_k];
                        
                        for (int j = 0; j < size; ++j) {
                            y_ptr[j] += alpha * x_ptr[j];
                        }
                    }
                }
                this->norms_valid = false;
                return;
            }
        }
        
        // Path C: General Case (Union)
        // We need to merge using GLOBAL indices to ensure correctness.
        
        // 1. Collect all global columns for each row
        std::vector<int> new_row_ptr(n_rows + 1);
        new_row_ptr[0] = 0;
        
        std::vector<int> row_nnz(n_rows);
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int y_start = this->row_ptr[i];
            int y_end = this->row_ptr[i+1];
            int x_start = X.row_ptr[i];
            int x_end = X.row_ptr[i+1];
            
            int count = 0;
            int y_k = y_start;
            int x_k = x_start;
            
            // We need to compare GLOBAL indices
            while (y_k < y_end && x_k < x_end) {
                int y_col_local = this->col_ind[y_k];
                int x_col_local = X.col_ind[x_k];
                int y_col_global = this->graph->get_global_index(y_col_local);
                int x_col_global = X.graph->get_global_index(x_col_local);
                
                if (y_col_global < x_col_global) {
                    count++; y_k++;
                } else if (x_col_global < y_col_global) {
                    count++; x_k++;
                } else {
                    count++; y_k++; x_k++;
                }
            }
            count += (y_end - y_k) + (x_end - x_k);
            row_nnz[i] = count;
        }
        
        for (int i = 0; i < n_rows; ++i) {
            new_row_ptr[i+1] = new_row_ptr[i] + row_nnz[i];
        }
        
        // 2. Construct new graph structure (Adjacency)
        std::vector<std::vector<int>> new_adj(n_rows);
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            std::vector<int> cols;
            // Reserve conservative estimate (sum of sizes)
            int y_count = this->row_ptr[i+1] - this->row_ptr[i];
            int x_count = X.row_ptr[i+1] - X.row_ptr[i];
            cols.reserve(y_count + x_count);
            
            // Collect A
            for(int k=this->row_ptr[i]; k<this->row_ptr[i+1]; ++k) {
                cols.push_back(this->graph->get_global_index(this->col_ind[k]));
            }
            // Collect B
            for(int k=X.row_ptr[i]; k<X.row_ptr[i+1]; ++k) {
                cols.push_back(X.graph->get_global_index(X.col_ind[k]));
            }
            
            // Sort and Unique
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            
            new_adj[i] = std::move(cols);
        }
        
        // 3. Create New Graph
        DistGraph* new_graph = new DistGraph(this->graph->comm);
        new_graph->construct_distributed(this->graph->owned_global_indices, this->graph->block_sizes, new_adj);

        // Manually populate block sizes for ghosts in new_graph using info from this and X
        // This is necessary because construct_distributed might not fetch ghost sizes for new ghosts immediately/correctly
        // and we already have the info locally.
        int new_total_cols = new_graph->global_to_local.size(); // Owned + Ghosts
        if (new_graph->block_sizes.size() < new_total_cols) {
            new_graph->block_sizes.resize(new_total_cols);
        }
        
        #pragma omp parallel for
        for (int i = 0; i < new_total_cols; ++i) {
            int gid = new_graph->get_global_index(i);
            int size = 0;
            
            // Try to find in 'this'
            auto it_this = this->graph->global_to_local.find(gid);
            if (it_this != this->graph->global_to_local.end()) {
                size = this->graph->block_sizes[it_this->second];
            } else {
                // Try to find in 'X'
                auto it_x = X.graph->global_to_local.find(gid);
                if (it_x != X.graph->global_to_local.end()) {
                    size = X.graph->block_sizes[it_x->second];
                } else {
                    // Should not happen if new_adj was constructed from this and X
                    // But if it does, we might have a problem. 
                    // However, for axpby, every column in new_graph comes from this or X.
                }
            }
            new_graph->block_sizes[i] = size;
        }
        
        // 4. Populate Values
        std::vector<int> new_col_ind;
        new_graph->get_matrix_structure(new_row_ptr, new_col_ind);
        
        int total_blocks_new = new_col_ind.size();
        std::vector<uint64_t> new_blk_handles(total_blocks_new);
        std::vector<size_t> new_blk_sizes(total_blocks_new);
        
        std::vector<size_t> row_val_size_new(n_rows);
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = new_row_ptr[i];
            int end = new_row_ptr[i+1];
            size_t sz = 0;
            int r_dim = new_graph->block_sizes[i];
            for(int k=start; k<end; ++k) {
                int col = new_col_ind[k];
                int c_dim = new_graph->block_sizes[col];
                sz += r_dim * c_dim;
            }
            row_val_size_new[i] = sz;
        }
        
        std::vector<long long> row_val_offset_new(n_rows + 1);
        row_val_offset_new[0] = 0;
        for(int i=0; i<n_rows; ++i) row_val_offset_new[i+1] = row_val_offset_new[i] + row_val_size_new[i];
        // opt: How to reuse the available memory without reallocating for them all?
        arena.reserve(row_val_offset_new.back());
        
        // Pass 1: Map existing handles and mark new ones (Parallel)
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = new_row_ptr[i];
            int end = new_row_ptr[i+1];
            int r_dim = new_graph->block_sizes[i];
            
            int this_start = this->row_ptr[i];
            int this_end = this->row_ptr[i+1];
            
            int tk = this_start;

            for(int k=start; k<end; ++k) {
                int col = new_col_ind[k];
                int c_dim = new_graph->block_sizes[col];
                new_blk_sizes[k] = r_dim * c_dim;
                
                int g_col_new = new_graph->get_global_index(col);
                
                // Advance tk to find match or go past
                // Both sequences are sorted by global column index
                while (tk < this_end) {
                    int this_col = this->col_ind[tk];
                    int g_col_old = this->graph->get_global_index(this_col);
                    if (g_col_old < g_col_new) {
                        tk++;
                    } else {
                        break;
                    }
                }
                
                bool found = false;
                if (tk < this_end) {
                    int this_col = this->col_ind[tk];
                    int g_col_old = this->graph->get_global_index(this_col);
                    if (g_col_old == g_col_new) {
                        new_blk_handles[k] = this->blk_handles[tk];
                        found = true;
                    }
                }
                
                if (!found) {
                    new_blk_handles[k] = UINT64_MAX; // Sentinel for "needs allocation"
                }
            }
        }

        // Pass 2: Allocate new blocks (Serial, to avoid race in arena)
        for (size_t k = 0; k < new_blk_handles.size(); ++k) {
            if (new_blk_handles[k] == UINT64_MAX) {
                new_blk_handles[k] = arena.allocate(new_blk_sizes[k]);
            }
        }
        
        // 5. Populate Values
        // We must respect the structure of new_graph (new_col_ind).
        // The merge loop below iterates in Global Index Order (or whatever order inputs are in).
        // We need to map each merged block to its correct position in new_val.
        
        // Initialize new_val to 0 because we might accumulate (e.g. if inputs have duplicate global cols)
        // std::fill(new_val.begin(), new_val.end(), T(0));
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int y_start = this->row_ptr[i];
            int y_end = this->row_ptr[i+1];
            int x_start = X.row_ptr[i];
            int x_end = X.row_ptr[i+1];
            
            int dest_start = new_row_ptr[i];
            int dest_end = new_row_ptr[i+1];
            
            // Build a lookup map: Global Col -> Index k in new_col_ind
            // Since block count per row is small, a simple vector of pairs is efficient.
            std::vector<std::pair<int, int>> col_map;
            col_map.reserve(dest_end - dest_start);
            for (int k = dest_start; k < dest_end; ++k) {
                int local_col = new_col_ind[k];
                int global_col = new_graph->get_global_index(local_col);
                col_map.push_back({global_col, k});
            }
            // Sort for binary search (or just linear scan if very small)
            std::sort(col_map.begin(), col_map.end());
            
            int y_k = y_start;
            int x_k = x_start;
            
            while (y_k < y_end || x_k < x_end) {
                int col_global = -1;
                bool use_y = false;
                bool use_x = false;
                
                // Determine next column to process
                if (y_k < y_end && x_k < x_end) {
                    int y_col_global = this->graph->get_global_index(this->col_ind[y_k]);
                    int x_col_global = X.graph->get_global_index(X.col_ind[x_k]);
                    
                    if (y_col_global < x_col_global) {
                        col_global = y_col_global; use_y = true;
                    } else if (x_col_global < y_col_global) {
                        col_global = x_col_global; use_x = true;
                    } else {
                        col_global = y_col_global; use_y = true; use_x = true;
                    }
                } else if (y_k < y_end) {
                    col_global = this->graph->get_global_index(this->col_ind[y_k]); use_y = true;
                } else {
                    col_global = X.graph->get_global_index(X.col_ind[x_k]); use_x = true;
                }
                
                // Find target index k in new_val
                auto it = std::lower_bound(col_map.begin(), col_map.end(), std::make_pair(col_global, -1));
                if (it == col_map.end() || it->first != col_global) {
                     // This implies the column exists in inputs but not in the graph?
                     // This shouldn't happen if graph construction was correct.
                     // But if it does, we skip to avoid crash.
                     if (use_y) y_k++;
                     if (use_x) x_k++;
                     continue;
                }
                
                int k = it->second;
                size_t blk_sz = new_blk_sizes[k];
                
                T* y_ptr = arena.get_ptr(new_blk_handles[k]);
                
                if (use_y && use_x) {
                    const T* x_ptr = X.arena.get_ptr(X.blk_handles[x_k]);
                    for(size_t j=0; j<blk_sz; ++j) y_ptr[j] = alpha * x_ptr[j] + beta * y_ptr[j];
                    y_k++; x_k++;
                } else if (use_y) {
                    for(size_t j=0; j<blk_sz; ++j) y_ptr[j] = beta * y_ptr[j];
                    y_k++;
                } else {
                    const T* x_ptr = X.arena.get_ptr(X.blk_handles[x_k]);
                    for(size_t j=0; j<blk_sz; ++j) y_ptr[j] = alpha * x_ptr[j];
                    x_k++;
                }
            }
        }
        
        // Update this
        this->row_ptr = std::move(new_row_ptr);
        this->col_ind = std::move(new_col_ind);
        this->blk_handles = std::move(new_blk_handles);
        this->blk_sizes = std::move(new_blk_sizes);
        this->norms_valid = false;
        
        if (this->owns_graph && this->graph) delete this->graph;
        this->graph = new_graph;
        this->owns_graph = true;
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
        
        // Swap
        row_ptr = std::move(new_row_ptr);
        col_ind = std::move(new_col_ind);
        blk_handles = std::move(new_blk_handles);
        blk_sizes = std::move(new_blk_sizes);
        block_norms = std::move(new_norms);
        norms_valid = true;
        
        // Sync Graph

        // Rebuild Graph to update communication pattern
        // 1. Collect global indices for the new adjacency list
        std::vector<std::vector<int>> new_adj_global(n_rows);
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            new_adj_global[i].reserve(end - start);
            for (int k = start; k < end; ++k) {
                int local_col = col_ind[k];
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
        for (size_t k = 0; k < col_ind.size(); ++k) {
            int old_local = col_ind[k];
            int global_col = graph->get_global_index(old_local);
            
            // Use .at() to ensure it exists (throws if not)
            // However, .at() is not const-qualified in all C++ versions? No, it is.
            // But to be safe in OMP, ensure no writes happen to map.
            col_ind[k] = new_graph->global_to_local.at(global_col);
        }
        
        // 4. Replace Graph
        if (this->owns_graph && this->graph) delete this->graph;
        this->graph = new_graph;
        this->owns_graph = true;
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
        
        // 1. Metadata Exchange
        GhostMetadata meta = exchange_ghost_metadata(B);
        
        // Ensure norms are ready
        const auto& A_norms = get_block_norms();
        const auto& B_local_norms = B.get_block_norms();
        
        // 2. Symbolic Phase (Structure Prediction)
        SymbolicResult sym = symbolic_multiply_filtered(B, meta, threshold);
        
        // 3. Data Exchange (Fetch Ghost Blocks)
        auto [ghost_data_map, ghost_sizes] = B.fetch_ghost_blocks(sym.required_blocks);
        
        // Reorganize ghost data for fast row access
        std::map<int, std::vector<GhostBlockRef>> ghost_rows;
        for (const auto& [bid, data] : ghost_data_map) {
            int c_dim = ghost_sizes.at(bid.col);
            
            double norm = 0.0;
            if (meta.count(bid.row)) {
                for(const auto& m : meta.at(bid.row)) {
                    if (m.col == bid.col) {
                        norm = m.norm;
                        break;
                    }
                }
            }
            
            ghost_rows[bid.row].push_back({bid.col, data.data(), c_dim, norm});
        }
        
        // 4. Construct Result Matrix Structure
        std::vector<std::vector<int>> adj(graph->owned_global_indices.size());
        int n_rows = row_ptr.size() - 1;
        for(int i=0; i<n_rows; ++i) {
            int start = sym.c_row_ptr[i];
            int end = sym.c_row_ptr[i+1];
            for(int k=start; k<end; ++k) {
                adj[i].push_back(sym.c_col_ind[k]);
            }
        }
        
        DistGraph* c_graph = new DistGraph(graph->comm);
        c_graph->construct_distributed(graph->owned_global_indices, graph->block_sizes, adj);

        // [FIX START] Backfill block sizes for ghost columns in C
        // if (graph->rank == 0) {
        //     std::cout << "Backfill block sizes for ghost columns in C" << std::endl;
        // }
        {
            // ghost_sizes contains {global_col -> size} for remote blocks
            // B.graph contains sizes for local blocks
            
            int c_n_cols = c_graph->global_to_local.size();
            // Resize if necessary (construct_distributed usually only sizes for owned)
            if (c_graph->block_sizes.size() < c_n_cols) {
                c_graph->block_sizes.resize(c_n_cols, 0);
            }

            #pragma omp parallel for
            for(int i = 0; i < c_n_cols; ++i) {
                // If size is already set (owned col), skip
                if (c_graph->block_sizes[i] > 0) continue;

                int g_col = c_graph->get_global_index(i);
                
                // 1. Try Ghost Map (from fetch_ghost_blocks)
                if (ghost_sizes.count(g_col)) {
                    c_graph->block_sizes[i] = ghost_sizes.at(g_col);
                }
                // 2. Try B's Local Graph (if the column exists locally in B)
                else if (B.graph->global_to_local.count(g_col)) {
                    int b_local = B.graph->global_to_local.at(g_col);
                    c_graph->block_sizes[i] = B.graph->block_sizes[b_local];
                }
                // If neither, we have a logic error (C columns must come from B)
            }

            // Recompute offsets for consistency
            c_graph->block_offsets.resize(c_n_cols + 1);
            c_graph->block_offsets[0] = 0;
            for(int i=0; i<c_n_cols; ++i) {
                c_graph->block_offsets[i+1] = c_graph->block_offsets[i] + c_graph->block_sizes[i];
            }
        }
        // [FIX END]
        
        BlockSpMat C(c_graph);
        C.owns_graph = true;
        
        // 5. Numeric Phase (GEMM)
        numeric_multiply(B, ghost_rows, C, threshold, A_norms, B_local_norms);
        
        // 6. Post-filtering
        C.filter_blocks(threshold);
        
        return C;
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
        int size = graph->size;
        int rank = graph->rank;
        
        // 1. Counting pass
        std::vector<size_t> send_counts(size, 0);
        std::vector<size_t> send_data_counts(size, 0);
        
        int n_rows = row_ptr.size() - 1;
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            int r_dim = graph->block_sizes[i];
            for (int k = start; k < end; ++k) {
                int g_col = graph->get_global_index(col_ind[k]);
                int owner = graph->find_owner(g_col);
                int c_dim = graph->block_sizes[col_ind[k]];
                
                send_counts[owner] += 4; // g_row, g_col, r_dim, c_dim
                send_data_counts[owner] += (size_t)r_dim * c_dim;
            }
        }
        
        // 2. Exchange counts
        std::vector<size_t> recv_counts(size);
        std::vector<size_t> recv_data_counts(size);
        if (graph->size > 1) {
            MPI_Alltoall(send_counts.data(), sizeof(size_t), MPI_BYTE, recv_counts.data(), sizeof(size_t), MPI_BYTE, graph->comm);
            MPI_Alltoall(send_data_counts.data(), sizeof(size_t), MPI_BYTE, recv_data_counts.data(), sizeof(size_t), MPI_BYTE, graph->comm);
        } else {
            recv_counts = send_counts;
            recv_data_counts = send_data_counts;
        }
        
        // 3. Setup displacements
        std::vector<size_t> sdispls(size + 1, 0), rdispls(size + 1, 0);
        std::vector<size_t> sdispls_data(size + 1, 0), rdispls_data(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
            sdispls_data[i+1] = sdispls_data[i] + send_data_counts[i];
            rdispls_data[i+1] = rdispls_data[i] + recv_data_counts[i];
        }
        
        // 4. Pack flat buffers
        std::vector<int> send_buf(sdispls[size]);
        std::vector<T> send_val(sdispls_data[size]);
        std::vector<size_t> current_counts(size, 0);
        std::vector<size_t> current_data_counts(size, 0);
        
        for (int i = 0; i < n_rows; ++i) {
            int g_row = graph->owned_global_indices[i];
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            int r_dim = graph->block_sizes[i];
            for (int k = start; k < end; ++k) {
                int g_col = graph->get_global_index(col_ind[k]);
                int owner = graph->find_owner(g_col);
                int c_dim = graph->block_sizes[col_ind[k]];
                
                int* meta_ptr = send_buf.data() + sdispls[owner] + current_counts[owner];
                meta_ptr[0] = g_col;
                meta_ptr[1] = g_row;
                meta_ptr[2] = c_dim;
                meta_ptr[3] = r_dim;
                current_counts[owner] += 4;
                
                size_t count = (size_t)r_dim * c_dim;
                std::memcpy(send_val.data() + sdispls_data[owner] + current_data_counts[owner], arena.get_ptr(blk_handles[k]), count * sizeof(T));
                current_data_counts[owner] += count;
            }
        }
        
        // 5. Exchange data
        std::vector<int> recv_buf(rdispls[size]);
        if (graph->size > 1) {
            safe_alltoallv(send_buf.data(), send_counts, sdispls, MPI_INT,
                          recv_buf.data(), recv_counts, rdispls, MPI_INT, graph->comm);
        } else {
            recv_buf = send_buf;
        }
                      
        std::vector<T> recv_val(rdispls_data[size]);
        std::vector<size_t> send_data_bytes(size), recv_data_bytes(size), sdispls_data_bytes(size + 1), rdispls_data_bytes(size + 1);
        for(int i=0; i<size; ++i) {
            send_data_bytes[i] = send_data_counts[i] * sizeof(T);
            recv_data_bytes[i] = recv_data_counts[i] * sizeof(T);
            sdispls_data_bytes[i] = sdispls_data[i] * sizeof(T);
            rdispls_data_bytes[i] = rdispls_data[i] * sizeof(T);
        }
        sdispls_data_bytes[size] = sdispls_data[size] * sizeof(T);
        rdispls_data_bytes[size] = rdispls_data[size] * sizeof(T);

        if (graph->size > 1) {
            safe_alltoallv(send_val.data(), send_data_bytes, sdispls_data_bytes, MPI_BYTE,
                          recv_val.data(), recv_data_bytes, rdispls_data_bytes, MPI_BYTE, graph->comm);
        } else {
            recv_val = send_val;
        }
                      
        // 6. Construct C
        std::vector<std::vector<int>> my_adj(graph->owned_global_indices.size());
        std::map<int, int> ghost_dims; // Map Global Col -> Block Size
        
        int* ptr = recv_buf.data();
        for(int i=0; i<size; ++i) {
            size_t count = recv_counts[i];
            int* end = ptr + count;
            while(ptr < end) {
                int g_row = *ptr++; // C's row
                int g_col = *ptr++; // C's col
                int r_dim = *ptr++; // C's row dim
                int c_dim = *ptr++; // C's col dim
                
                if (graph->global_to_local.count(g_row)) {
                    int l_row = graph->global_to_local.at(g_row);
                    my_adj[l_row].push_back(g_col);
                    
                    // Store dimension for ghost backfilling
                    ghost_dims[g_col] = c_dim;
                }
            }
        }
        
        for(auto& neighbors : my_adj) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
        
        DistGraph* graph_C = new DistGraph(graph->comm);
        graph_C->construct_distributed(graph->owned_global_indices, graph->block_sizes, my_adj);
        
        // [FIX START] Backfill block sizes for ghost columns in C (Transpose)
        {
            int c_n_cols = graph_C->global_to_local.size();
            if (graph_C->block_sizes.size() < c_n_cols) {
                graph_C->block_sizes.resize(c_n_cols, 0);
            }
            
            #pragma omp parallel for
            for(int i = 0; i < c_n_cols; ++i) {
                if (graph_C->block_sizes[i] > 0) continue; // Skip owned
                
                int g_col = graph_C->get_global_index(i);
                // Use .at() for thread-safe read-only access (operator[] is not const)
                if (ghost_dims.count(g_col)) {
                    graph_C->block_sizes[i] = ghost_dims.at(g_col);
                } else {
                     // Should not happen if all columns came from recv_buf
                     throw std::runtime_error("Missing block size in transpose for ghost col");
                }
            }

            // Recompute offsets
            graph_C->block_offsets.resize(c_n_cols + 1);
            graph_C->block_offsets[0] = 0;
            for(int i=0; i<c_n_cols; ++i) {
                graph_C->block_offsets[i+1] = graph_C->block_offsets[i] + graph_C->block_sizes[i];
            }
        }
        
        BlockSpMat C(graph_C);
        C.owns_graph = true;
        
        // 7. Unpack and Insert
        ptr = recv_buf.data();
        T* val_ptr = recv_val.data();
        for(int i=0; i<size; ++i) {
            size_t count = recv_counts[i];
            int* end = ptr + count;
            while(ptr < end) {
                int g_row = *ptr++; 
                int g_col = *ptr++; 
                int r_dim = *ptr++; 
                int c_dim = *ptr++; 
                size_t n_elem = (size_t)r_dim * c_dim;
                
                std::vector<T> block(n_elem);
                int rows_A = c_dim; 
                int cols_A = r_dim;
                for(int c=0; c<cols_A; ++c) {
                    for(int r=0; r<rows_A; ++r) {
                        T val = val_ptr[r + c * rows_A];
                        val = ConjHelper<T>::apply(val);
                        block[c + r * r_dim] = val;
                    }
                }
                C.add_block(g_row, g_col, block.data(), r_dim, c_dim, AssemblyMode::ADD);
                val_ptr += n_elem;
            }
        }
        C.assemble();
        return C;
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
    struct GhostBlockRef { int col; const T* data; int c_dim; double norm; };
    using GhostBlockData = std::map<BlockID, std::vector<T>>;
    using GhostSizes = std::map<int, int>;

    using GhostMetadata = std::map<int, std::vector<BlockMeta>>;

    struct SymbolicResult {
        std::vector<int> c_row_ptr;
        std::vector<int> c_col_ind;
        std::vector<BlockID> required_blocks;
    };

    // SpMM Phase 1: Metadata Exchange
    GhostMetadata exchange_ghost_metadata(const BlockSpMat& B) const {
        GhostMetadata metadata;
        int size = graph->size;
        int rank = graph->rank;
        
        // 1. Identify needed rows of B (column indices of A)
        std::set<int> needed_rows;
        int n_rows = row_ptr.size() - 1;
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            for (int k = start; k < end; ++k) {
                int global_col = graph->get_global_index(col_ind[k]);
                if (B.graph->find_owner(global_col) != rank) {
                    needed_rows.insert(global_col);
                }
            }
        }

        // 2. Count requests per rank
        std::vector<int> send_req_counts(size, 0);
        for (int global_row : needed_rows) {
            int owner = B.graph->find_owner(global_row);
            send_req_counts[owner]++;
        }

        // 3. Exchange request counts
        std::vector<int> recv_req_counts(size);
        if (graph->size > 1) {
            MPI_Alltoall(send_req_counts.data(), 1, MPI_INT, recv_req_counts.data(), 1, MPI_INT, graph->comm);
        } else {
            recv_req_counts = send_req_counts;
        }

        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_req_counts[i];
            rdispls[i+1] = rdispls[i] + recv_req_counts[i];
        }

        // 4. Pack request buffer
        std::vector<int> send_req_buf(sdispls[size]);
        std::vector<int> current_req_counts(size, 0);
        for (int global_row : needed_rows) {
            int owner = B.graph->find_owner(global_row);
            send_req_buf[sdispls[owner] + current_req_counts[owner]++] = global_row;
        }

        std::vector<int> recv_req_buf(rdispls[size]);
        if (graph->size > 1) {
            MPI_Alltoallv(send_req_buf.data(), send_req_counts.data(), sdispls.data(), MPI_INT,
                          recv_req_buf.data(), recv_req_counts.data(), rdispls.data(), MPI_INT, graph->comm);
        } else {
            recv_req_buf = send_req_buf;
        }

        // 5. Counting pass for replies
        std::vector<double> B_norms = B.compute_block_norms();
        std::vector<size_t> send_reply_bytes(size, 0);
        int* ptr = recv_req_buf.data();
        for(int i=0; i<size; ++i) {
            int count = recv_req_counts[i];
            int* end = ptr + count;
            while(ptr < end) {
                int global_row = *ptr++;
                if (B.graph->global_to_local.count(global_row)) {
                    int local_row = B.graph->global_to_local.at(global_row);
                    int n_blocks = B.row_ptr[local_row+1] - B.row_ptr[local_row];
                    send_reply_bytes[i] += 2 * sizeof(int) + n_blocks * (sizeof(int) + sizeof(double));
                }
            }
        }

        // 6. Exchange reply counts
        std::vector<size_t> recv_reply_bytes(size);
        if (graph->size > 1) {
            MPI_Alltoall(send_reply_bytes.data(), sizeof(size_t), MPI_BYTE, recv_reply_bytes.data(), sizeof(size_t), MPI_BYTE, graph->comm);
        } else {
            recv_reply_bytes = send_reply_bytes;
        }

        std::vector<size_t> sdispls_reply(size + 1, 0), rdispls_reply(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls_reply[i+1] = sdispls_reply[i] + send_reply_bytes[i];
            rdispls_reply[i+1] = rdispls_reply[i] + recv_reply_bytes[i];
        }

        // 7. Pack reply buffer
        std::vector<char> send_reply_blob(sdispls_reply[size]);
        ptr = recv_req_buf.data();
        for(int i=0; i<size; ++i) {
            char* blob_ptr = send_reply_blob.data() + sdispls_reply[i];
            int count = recv_req_counts[i];
            int* end = ptr + count;
            while(ptr < end) {
                int global_row = *ptr++;
                if (B.graph->global_to_local.count(global_row)) {
                    int local_row = B.graph->global_to_local.at(global_row);
                    int start = B.row_ptr[local_row];
                    int end_row = B.row_ptr[local_row+1];
                    int n_blocks = end_row - start;
                    
                    std::memcpy(blob_ptr, &global_row, sizeof(int)); blob_ptr += sizeof(int);
                    std::memcpy(blob_ptr, &n_blocks, sizeof(int)); blob_ptr += sizeof(int);
                    for(int k=start; k<end_row; ++k) {
                        int col = B.graph->get_global_index(B.col_ind[k]);
                        double norm = B_norms[k];
                        std::memcpy(blob_ptr, &col, sizeof(int)); blob_ptr += sizeof(int);
                        std::memcpy(blob_ptr, &norm, sizeof(double)); blob_ptr += sizeof(double);
                    }
                }
            }
        }

        // 8. Exchange replies
        std::vector<char> recv_reply_blob(rdispls_reply[size]);
        if (graph->size > 1) {
            safe_alltoallv(send_reply_blob.data(), send_reply_bytes, sdispls_reply, MPI_BYTE,
                          recv_reply_blob.data(), recv_reply_bytes, rdispls_reply, MPI_BYTE, graph->comm);
        } else {
            recv_reply_blob = send_reply_blob;
        }

        // 9. Unpack replies
        for(int i=0; i<size; ++i) {
            char* blob_ptr = recv_reply_blob.data() + rdispls_reply[i];
            char* end = recv_reply_blob.data() + rdispls_reply[i+1];
            while(blob_ptr < end) {
                int global_row, n_blocks;
                std::memcpy(&global_row, blob_ptr, sizeof(int)); blob_ptr += sizeof(int);
                std::memcpy(&n_blocks, blob_ptr, sizeof(int)); blob_ptr += sizeof(int);
                auto& list = metadata[global_row];
                list.reserve(n_blocks);
                for(int k=0; k<n_blocks; ++k) {
                    BlockMeta meta;
                    std::memcpy(&meta.col, blob_ptr, sizeof(int)); blob_ptr += sizeof(int);
                    std::memcpy(&meta.norm, blob_ptr, sizeof(double)); blob_ptr += sizeof(double);
                    list.push_back(meta);
                }
            }
        }
        return metadata;
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
        GhostBlockData ghost_data;
        GhostSizes ghost_sizes;
        int size = graph->size;
        int rank = graph->rank;
        
        // 1. Counting pass for requests
        std::vector<size_t> send_req_counts(size, 0);
        for (const auto& bid : required_blocks) {
            int owner = graph->find_owner(bid.row);
            send_req_counts[owner] += 2 * sizeof(int); // row, col
        }
        
        // 2. Exchange request counts
        std::vector<size_t> recv_req_counts(size);
        if (graph->size > 1) {
            MPI_Alltoall(send_req_counts.data(), sizeof(size_t), MPI_BYTE, recv_req_counts.data(), sizeof(size_t), MPI_BYTE, graph->comm);
        } else {
            recv_req_counts = send_req_counts;
        }
        
        // 3. Setup request displacements
        std::vector<size_t> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_req_counts[i];
            rdispls[i+1] = rdispls[i] + recv_req_counts[i];
        }
        
        // 4. Pack request buffer
        std::vector<int> send_req_buf(sdispls[size] / sizeof(int));
        std::vector<size_t> current_req_counts(size, 0);
        for (const auto& bid : required_blocks) {
            int owner = graph->find_owner(bid.row);
            int* ptr = send_req_buf.data() + (sdispls[owner] + current_req_counts[owner]) / sizeof(int);
            ptr[0] = bid.row;
            ptr[1] = bid.col;
            current_req_counts[owner] += 2 * sizeof(int);
        }
        
        // 5. Exchange requests
        std::vector<int> recv_req_buf(rdispls[size] / sizeof(int));
        if (graph->size > 1) {
            safe_alltoallv(send_req_buf.data(), send_req_counts, sdispls, MPI_BYTE,
                          recv_req_buf.data(), recv_req_counts, rdispls, MPI_BYTE, graph->comm);
        } else {
            recv_req_buf = send_req_buf;
        }
                      
        // 6. Counting pass for replies
        std::vector<size_t> send_reply_bytes(size, 0);
        int* ptr = recv_req_buf.data();
        for(int i=0; i<size; ++i) {
            size_t count_bytes = recv_req_counts[i];
            int* end = ptr + count_bytes / sizeof(int);
            while(ptr < end) {
                int g_row = *ptr++;
                int g_col = *ptr++;
                if (graph->global_to_local.count(g_row)) {
                    int l_row = graph->global_to_local.at(g_row);
                    int start = row_ptr[l_row];
                    int end_row = row_ptr[l_row+1];
                    for(int k=start; k<end_row; ++k) {
                        if (graph->get_global_index(col_ind[k]) == g_col) {
                            int r_dim = graph->block_sizes[l_row];
                            int c_dim = graph->block_sizes[col_ind[k]];
                            send_reply_bytes[i] += 4 * sizeof(int) + r_dim * c_dim * sizeof(T);
                            break;
                        }
                    }
                }
            }
        }
        
        // 7. Exchange reply counts
        std::vector<size_t> recv_reply_bytes(size);
        if (graph->size > 1) {
            MPI_Alltoall(send_reply_bytes.data(), sizeof(size_t), MPI_BYTE, recv_reply_bytes.data(), sizeof(size_t), MPI_BYTE, graph->comm);
        } else {
            recv_reply_bytes = send_reply_bytes;
        }
        
        // 8. Setup reply displacements
        std::vector<size_t> sdispls_reply(size + 1, 0), rdispls_reply(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls_reply[i+1] = sdispls_reply[i] + send_reply_bytes[i];
            rdispls_reply[i+1] = rdispls_reply[i] + recv_reply_bytes[i];
        }
        
        // 9. Pack reply buffer
        std::vector<char> send_reply_blob(sdispls_reply[size]);
        ptr = recv_req_buf.data();
        for(int i=0; i<size; ++i) {
            char* blob_ptr = send_reply_blob.data() + sdispls_reply[i];
            size_t count_bytes = recv_req_counts[i];
            int* end = ptr + count_bytes / sizeof(int);
            while(ptr < end) {
                int g_row = *ptr++;
                int g_col = *ptr++;
                if (graph->global_to_local.count(g_row)) {
                    int l_row = graph->global_to_local.at(g_row);
                    int start = row_ptr[l_row];
                    int end_row = row_ptr[l_row+1];
                    for(int k=start; k<end_row; ++k) {
                        if (graph->get_global_index(col_ind[k]) == g_col) {
                            int r_dim = graph->block_sizes[l_row];
                            int c_dim = graph->block_sizes[col_ind[k]];
                            // long long offset = blk_ptr[k];
                            size_t n_elem = r_dim * c_dim;
                            
                            std::memcpy(blob_ptr, &g_row, sizeof(int)); blob_ptr += sizeof(int);
                            std::memcpy(blob_ptr, &g_col, sizeof(int)); blob_ptr += sizeof(int);
                            std::memcpy(blob_ptr, &r_dim, sizeof(int)); blob_ptr += sizeof(int);
                            std::memcpy(blob_ptr, &c_dim, sizeof(int)); blob_ptr += sizeof(int);
                            std::memcpy(blob_ptr, arena.get_ptr(blk_handles[k]), n_elem * sizeof(T)); blob_ptr += n_elem * sizeof(T);
                            break;
                        }
                    }
                }
            }
        }
        
        // 10. Exchange replies
        std::vector<char> recv_reply_blob(rdispls_reply[size]);
        if (graph->size > 1) {
            safe_alltoallv(send_reply_blob.data(), send_reply_bytes, sdispls_reply, MPI_BYTE,
                          recv_reply_blob.data(), recv_reply_bytes, rdispls_reply, MPI_BYTE, graph->comm);
        } else {
            recv_reply_blob = send_reply_blob;
        }
                      
        // 11. Unpack
        for(int i=0; i<size; ++i) {
            char* blob_ptr = recv_reply_blob.data() + rdispls_reply[i];
            char* end = recv_reply_blob.data() + rdispls_reply[i+1];
            while(blob_ptr < end) {
                int g_row, g_col, r_dim, c_dim;
                std::memcpy(&g_row, blob_ptr, sizeof(int)); blob_ptr += sizeof(int);
                std::memcpy(&g_col, blob_ptr, sizeof(int)); blob_ptr += sizeof(int);
                std::memcpy(&r_dim, blob_ptr, sizeof(int)); blob_ptr += sizeof(int);
                std::memcpy(&c_dim, blob_ptr, sizeof(int)); blob_ptr += sizeof(int);
                
                size_t n_elem = r_dim * c_dim;
                std::vector<T> data(n_elem);
                std::memcpy(data.data(), blob_ptr, n_elem * sizeof(T)); blob_ptr += n_elem * sizeof(T);
                
                ghost_data[{g_row, g_col}] = std::move(data);
                ghost_sizes[g_col] = c_dim;
            }
        }
        
        return {ghost_data, ghost_sizes};
    }

    // SpMM Phase 4: Numerical Multiplication
    void numeric_multiply(const BlockSpMat& B, 
                          const std::map<int, std::vector<GhostBlockRef>>& ghost_rows,
                          BlockSpMat& C,
                          double threshold,
                          const std::vector<double>& A_norms,
                          const std::vector<double>& B_local_norms) const {
        int n_rows = row_ptr.size() - 1;
        
        int max_threads = omp_get_max_threads();
        const size_t HASH_SIZE = 8192; 
        const size_t HASH_MASK = HASH_SIZE - 1;
        
        struct HashEntry {
            int key; 
            uint64_t value; // Changed from long long to uint64_t for handle
            int tag; 
        };
        
        std::vector<std::vector<HashEntry>> thread_tables(max_threads, std::vector<HashEntry>(HASH_SIZE, {-1, 0, 0}));
        std::vector<int> thread_tags(max_threads, 0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& table = thread_tables[tid];
            int& tag = thread_tags[tid];
            
            #pragma omp for
            for (int i = 0; i < n_rows; ++i) {
                tag++;
                if (tag == 0) { 
                    for(auto& e : table) e.tag = 0;
                    tag = 1;
                }

                int c_start = C.row_ptr[i];
                int c_end = C.row_ptr[i+1];
                
                for(int k=c_start; k<c_end; ++k) {
                    int l_col = C.col_ind[k];
                    int g_col = C.graph->get_global_index(l_col);
                    uint64_t offset = C.blk_handles[k]; // Store handle instead of offset
                    
                    size_t h = (size_t)g_col & HASH_MASK;
                    size_t count = 0;
                    while(table[h].tag == tag) {
                        h = (h + 1) & HASH_MASK;
                        if (++count > HASH_SIZE) {
                            throw std::runtime_error("Hash table is full during SpMM population");
                        }
                    }
                    table[h] = {g_col, offset, tag};
                }
                
                int a_start = row_ptr[i];
                int a_end = row_ptr[i+1];
                int r_dim = graph->block_sizes[i];
                
                for (int k = a_start; k < a_end; ++k) {

                    // Dynamic threshold for this row
                    // row_count is number of blocks in A's row i
                    int row_count = a_end - a_start;
                    double row_eps = threshold / std::max(1, row_count);
                    
                    int l_col_A = col_ind[k];
                    int g_col_A = graph->get_global_index(l_col_A);
                    const T* a_val = arena.get_ptr(blk_handles[k]);
                    int inner_dim = graph->block_sizes[l_col_A];
                    double norm_A = A_norms[k];
                    
                    if (graph->find_owner(g_col_A) == graph->rank) {
                        int l_row_B = graph->global_to_local.at(g_col_A);
                        int b_start = B.row_ptr[l_row_B];
                        int b_end = B.row_ptr[l_row_B+1];
                        for(int j=b_start; j<b_end; ++j) {
                            double norm_B = B_local_norms[j];
                            
                            int l_col_B = B.col_ind[j];
                            int g_col_B = B.graph->get_global_index(l_col_B);
                            const T* b_val = B.arena.get_ptr(B.blk_handles[j]);

                            if (norm_A * norm_B < row_eps) continue;

                            int c_dim = B.graph->block_sizes[l_col_B];

                            size_t h = (size_t)g_col_B & HASH_MASK;
                            size_t count = 0;
                            while(table[h].tag == tag) {
                                if (table[h].key == g_col_B) {
                                    T* c_val = C.arena.get_ptr(table[h].value);
                                    SmartKernel<T>::gemm(r_dim, c_dim, inner_dim, T(1), a_val, r_dim, b_val, inner_dim, T(1), c_val, r_dim);
                                    break;
                                }
                                h = (h + 1) & HASH_MASK;
                                if (++count > HASH_SIZE) {
                                    throw std::runtime_error("Hash table infinite loop detected during SpMM numeric phase (local)");
                                }
                            }
                        }
                    } else {
                        auto it = ghost_rows.find(g_col_A);
                        if (it != ghost_rows.end()) {
                            for (const auto& block : it->second) {
                                int g_col_B = block.col;
                                const T* b_val = block.data;
                                int c_dim = block.c_dim;
                                double norm_B = block.norm;
                                
                                if (norm_A * norm_B < row_eps) continue;
                                
                                size_t h = (size_t)g_col_B & HASH_MASK;
                                size_t count = 0;
                                while(table[h].tag == tag) {
                                    if (table[h].key == g_col_B) {
                                        T* c_val = C.arena.get_ptr(table[h].value);
                                        SmartKernel<T>::gemm(r_dim, c_dim, inner_dim, T(1), a_val, r_dim, b_val, inner_dim, T(1), c_val, r_dim);
                                        break;
                                    }
                                    h = (h + 1) & HASH_MASK;
                                    if (++count > HASH_SIZE) {
                                        throw std::runtime_error("Hash table infinite loop detected during SpMM numeric phase (ghost)");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
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
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
        DistGraph* sub_graph = new DistGraph(MPI_COMM_SELF); // this make this method thread safe
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
        
        MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_LONG_LONG, MPI_SUM, graph->comm);
        
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

} // namespace vbcsr

#endif

