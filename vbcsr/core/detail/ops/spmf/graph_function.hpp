#ifndef VBCSR_DETAIL_OPS_SPMF_GRAPH_FUNCTION_HPP
#define VBCSR_DETAIL_OPS_SPMF_GRAPH_FUNCTION_HPP

#include "../../../block_csr.hpp"
#include "../../../dist_multivector.hpp"
#include "../../kernels/blas_api.hpp"
#include "../../kernels/lapack_api.hpp"
#include "subspace.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <omp.h>

namespace vbcsr {


template <typename T>
void graph_matrix_function(
    BlockSpMat<T>& A,
    BlockSpMat<T>* Result,
    std::function<T(double)> func,
    bool verbose) {
    
    DistGraph* graph = A.graph;
    int rank = graph->rank;
    int size = graph->size;
    MPI_Comm comm = graph->comm;

    // 1. Initialize Result
    if (rank == 0 && verbose) std::cout << "graph_matrix_function: Initializing (subgraph dense diagonalization)..." << std::endl;

    if (Result == nullptr) {
        throw std::invalid_argument("graph_matrix_function requires a valid output matrix pointer");
    }

    if (Result->graph != A.graph) {
        // If pointers differ, check basic properties
        if (Result->graph->comm != A.graph->comm || 
            Result->graph->owned_global_indices.size() != A.graph->owned_global_indices.size()) {
            throw std::runtime_error("graph_matrix_function: Result matrix has incompatible graph structure");
        }
    }
    
    // Clear Result
    Result->fill(T(0));
    
    int n_owned = graph->owned_global_indices.size();
    int n_owned_max = 0;
    if (comm != MPI_COMM_NULL && comm != MPI_COMM_SELF) {
        MPI_Allreduce(&n_owned, &n_owned_max, 1, MPI_INT, MPI_MAX, comm);
    } else {
        n_owned_max = n_owned;
    }
    
    // batch size
    int batch_size = std::max(1, omp_get_max_threads() / size);
    
    int nbatch = n_owned_max / batch_size;
    if (n_owned_max % batch_size != 0) nbatch++;

    std::vector<std::vector<int>> batch_indices(batch_size);

    for (int b = 0; b < nbatch; ++b) {
        // Avoid thread oversubscription in BLAS/LAPACK inside the parallel
        // subgraph loop (shared threading policy: dense_kernels.hpp).
        BLASKernel::configure_native_threading();

        batch_indices.clear();
        batch_indices.resize(batch_size);
        
        // Store neighbors for parallel phase
        std::vector<std::vector<int>> batch_neighbors(batch_size);

        for (int i=0; i < batch_size; ++i) {
            int idx = b * batch_size + i;
            
            if (idx < n_owned) {
                int global_row = graph->owned_global_indices[idx];
                // Identify Neighborhood C_i
                std::vector<int> neighbors;
                int start = A.row_ptr()[idx]; // Use local index idx
                int end = A.row_ptr()[idx+1];
                for (int k = start; k < end; ++k) {
                    int col_lid = A.col_ind()[k];
                    int col_gid = graph->get_global_index(col_lid);
                    neighbors.push_back(col_gid);
                }
                // Ensure global_row is in neighbors (diagonal)
                bool has_diag = false;
                for(int gid : neighbors) if(gid == global_row) has_diag = true;
                if(!has_diag) neighbors.push_back(global_row);
                
                // Sort for consistency
                std::sort(neighbors.begin(), neighbors.end());

                batch_indices[i] = neighbors;
            } else {
                // Padding: Empty request
                batch_indices[i] = {};
            }
        }
        
        auto batch_blocks = detail::fetch_batched_block_payloads(A, batch_indices);

        #pragma omp parallel for schedule(dynamic)
        for (int i=0; i < batch_size; ++i) {
            int idx = b * batch_size + i;
            if (idx >= n_owned) continue;
            
            int global_row = graph->owned_global_indices[idx];
            const auto& neighbors = batch_indices[i];

            // Find block index of global_row
            auto it = std::find(neighbors.begin(), neighbors.end(), global_row);
            int block_idx = std::distance(neighbors.begin(), it);
            BlockSpMat<T> sub_mat = A.construct_submatrix(neighbors, batch_blocks);
            
            int r_dim = sub_mat.graph->block_sizes[block_idx]; // getting from submat, where index is the neighbours order, safe

            // Convert to Dense
            std::vector<T> M = sub_mat.to_dense();
        
            // M is row-major, size (total_dim) x (total_dim)
            int total_dim = 0;
            for(size_t k=0; k<neighbors.size(); ++k) total_dim += sub_mat.graph->block_sizes[k];

            if (rank==0 && verbose) {
                std::cout << "working on atom: "<< idx << "/" << n_owned_max << " total dim: " << total_dim << std::endl;
            }
            
            // Calculate offset in dense matrix M
            int row_offset = 0;
            for(int k=0; k<block_idx; ++k) row_offset += sub_mat.graph->block_sizes[k];
            
            DistMultiVector<T> X(sub_mat.graph, r_dim);

            // Dense diagonalization of the subgraph matrix (the only route:
            // the Lanczos variant was removed after measuring consistently
            // worse efficiency than the direct dense diagonalization).
            // Only the needed columns [row_offset, row_offset + r_dim) of
            // f(M) are formed.
            dense_matrix_function(total_dim, M, func, r_dim, row_offset);

            // M is now f(M)[:, row_offset:row_offset+r_dim]
            // Size: total_dim x r_dim
            // Layout: ColMajor (from dense_gemm)

            X.bind_to_graph(sub_mat.graph);
            // X is row-major (padded ld): transpose-copy the column-major
            // dense result across the boundary.
            for (int r = 0; r < total_dim; ++r) {
                T* dst = X.row_data(r);
                for (int c = 0; c < r_dim; ++c) {
                    dst[c] = M[static_cast<size_t>(c) * total_dim + r];
                }
            }
            
            // Iterate over columns (neighbors)
            int col_offset = 0;
            for(size_t k=0; k<neighbors.size(); ++k) {
                int col_gid = neighbors[k];
                int c_dim = sub_mat.graph->block_sizes[k];
                
                // Extract block (r_dim x c_dim) from X
                // X contains f(M)(:, row_offset:row_offset+r_dim)
                // We want block (global_row, col_gid) of f(A)
                // In sub_mat, this is block (block_idx, k).
                // f(M)(row_offset:row_offset+r_dim, col_offset:col_offset+c_dim)
                // By symmetry = f(M)(col_offset:col_offset+c_dim, row_offset:row_offset+r_dim)^T
                // X has rows col_offset:col_offset+c_dim
                
                std::vector<T> block_data(r_dim * c_dim);
                for(int r=0; r<r_dim; ++r) {
                    for(int c=0; c<c_dim; ++c) {
                        // We want X(col_offset + c, r); the accessor hides the
                        // (row-major, padded-ld) multivector storage. The block
                        // is packed in canonical row-major order.
                        block_data[r * c_dim + c] = X(col_offset + c, r);
                    }
                }

                Result->add_block(global_row, col_gid, block_data.data(), r_dim, c_dim, AssemblyMode::ADD, kCanonicalBlockLayout);
                
                col_offset += c_dim;
            }
        }
    }
    
    Result->assemble();

}

}

#endif // VBCSR_DETAIL_OPS_SPMF_GRAPH_FUNCTION_HPP
