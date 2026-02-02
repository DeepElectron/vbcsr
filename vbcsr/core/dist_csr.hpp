#ifndef VBCSR_DIST_CSR_HPP
#define VBCSR_DIST_CSR_HPP

#include "block_csr.hpp"
#include <vector>
#include <mpi.h>
#include <algorithm>
#include "mpi_utils.hpp"
#include <numeric>

namespace vbcsr {

template <typename T>
struct DistCSR {
    // Local CSR structure
    std::vector<int> row_ptr;
    // PARDISO expects standard CSR with 0-based indexing (iparm[34]=1)
    // For distributed PARDISO, col_ind must be GLOBAL indices.
    std::vector<int> col_ind; 
    std::vector<T> values;
    
    // Metadata
    int n_local_rows = 0;
    int n_global_rows = 0;
    int n_global_cols = 0; // Usually square matrix
    int local_row_start = 0; // Global index of first local row
    
    MPI_Comm comm = MPI_COMM_NULL;
    
    // Constructor
    DistCSR() = default;
};

template <typename T, typename Kernel>
DistCSR<T> convert_to_csr(const BlockSpMat<T, Kernel>& mat) {
    DistCSR<T> csr;
    csr.comm = mat.graph->comm;
    
    int rank, size;
    MPI_Comm_rank(csr.comm, &rank);
    MPI_Comm_size(csr.comm, &size);
    
    // 1. Calculate local scalar sizes and offsets
    const auto& block_sizes = mat.graph->block_sizes; // Contains owned + ghost sizes
    int n_owned_blocks = mat.graph->owned_global_indices.size();
    
    int my_local_rows = 0;
    // Offsets of each owned block relative to start of this rank's rows
    std::vector<int> block_local_offsets(n_owned_blocks + 1, 0); 
    
    for (int i = 0; i < n_owned_blocks; ++i) {
        block_local_offsets[i] = my_local_rows;
        my_local_rows += block_sizes[i];
    }
    block_local_offsets[n_owned_blocks] = my_local_rows;
    csr.n_local_rows = my_local_rows;
    
    // 2. Calculate global offsets (start row for each rank)
    std::vector<int> rank_global_offsets(size + 1, 0);
    std::vector<int> all_local_rows(size);
    
    MPI_Allgather(&my_local_rows, 1, MPI_INT, all_local_rows.data(), 1, MPI_INT, csr.comm);
    
    for (int i = 0; i < size; ++i) {
        rank_global_offsets[i+1] = rank_global_offsets[i] + all_local_rows[i];
    }
    csr.n_global_rows = rank_global_offsets[size];
    csr.n_global_cols = csr.n_global_rows; // Assume square
    csr.local_row_start = rank_global_offsets[rank];
    
    // 3. Resolve Global Scalar Offsets for Ghost Blocks
    // We need to know where each ghost block starts in the global scalar indexing.
    // Ghost block G is owned by rank Owner(G). 
    // GlobalScalarStart(G) = rank_global_offsets[Owner(G)] + local_offset_on_owner(G)
    
    // 3.1 Identify ghosts and their owners
    const auto& ghost_gids = mat.graph->ghost_global_indices;
    int n_ghosts = ghost_gids.size();
    
    // We use the same communication pattern as fetch_ghost_block_sizes but fetch offsets
    // Map: Owner -> [List of GIDs to request]
    std::map<int, std::vector<int>> ghosts_by_rank;
    for (int i = 0; i < n_ghosts; ++i) {
        int gid = ghost_gids[i];
        int owner = mat.graph->find_owner(gid);
        ghosts_by_rank[owner].push_back(gid);
    }
    
    // 3.2 Exchange Requests (How many GIDs from each)
    std::vector<int> req_counts(size, 0);
    for (const auto& kv : ghosts_by_rank) req_counts[kv.first] = kv.second.size();
    
    std::vector<int> incom_req_counts(size);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, incom_req_counts.data(), 1, MPI_INT, csr.comm);
    
    // 3.3 Exchange GIDs
    // Prepare Send Buffers
    std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        sdispls[i+1] = sdispls[i] + req_counts[i];
        rdispls[i+1] = rdispls[i] + incom_req_counts[i];
    }
    
    std::vector<int> req_send_buf(sdispls[size]);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        if (ghosts_by_rank.count(i)) {
            std::copy(ghosts_by_rank[i].begin(), ghosts_by_rank[i].end(), req_send_buf.begin() + offset);
        }
        offset += req_counts[i];
    }
    
    std::vector<int> req_recv_buf(rdispls[size]);
    MPI_Alltoallv(req_send_buf.data(), req_counts.data(), sdispls.data(), MPI_INT,
                  req_recv_buf.data(), incom_req_counts.data(), rdispls.data(), MPI_INT, csr.comm);
                  
    // 3.4 Lookup Local Offsets for requested GIDs
    std::vector<int> resp_send_buf(rdispls[size]); // Responding with offsets
    for (int i = 0; i < rdispls[size]; ++i) {
        int gid = req_recv_buf[i];
        // Convert global block ID to local block ID
        if (mat.graph->global_to_local.find(gid) == mat.graph->global_to_local.end()) {
             throw std::runtime_error("DistCSR: Requested GID not found in local map");
        }
        int lid = mat.graph->global_to_local.at(gid);
        // lid must be an owned block
        if (lid >= n_owned_blocks) {
             throw std::runtime_error("DistCSR: Requested GID is not an owned block");
        }
        resp_send_buf[i] = block_local_offsets[lid];
    }
    
    // 3.5 Receive Local Offsets
    std::vector<int> resp_recv_buf(sdispls[size]);
    MPI_Alltoallv(resp_send_buf.data(), incom_req_counts.data(), rdispls.data(), MPI_INT,
                  resp_recv_buf.data(), req_counts.data(), sdispls.data(), MPI_INT, csr.comm);
                  
    // 3.6 Map Ghost LID -> Global Scalar Start Index
    // LID is relative to n_owned_blocks
    std::vector<int> ghost_scalar_starts(n_ghosts);
    offset = 0;
    for (int i = 0; i < size; ++i) {
        if (ghosts_by_rank.count(i)) {
            int rank_base_offset = rank_global_offsets[i];
            const auto& gids = ghosts_by_rank[i];
            for (size_t k = 0; k < gids.size(); ++k) {
                int gid = gids[k];
                int lid = mat.graph->global_to_local.at(gid); // This is the local ID on *my* process (index in ghost part)
                int ghost_idx = lid - n_owned_blocks;
                
                int remote_local_offset = resp_recv_buf[offset + k];
                ghost_scalar_starts[ghost_idx] = rank_base_offset + remote_local_offset;
            }
        }
        offset += req_counts[i];
    }
    
    // 4. Build CSR
    // 4.1 Count NNZ
    int total_nnz = 0;
    std::vector<int> row_nnz(my_local_rows, 0);
    
    for (int i = 0; i < n_owned_blocks; ++i) {
        int r_dim = block_sizes[i];
        int start_row = block_local_offsets[i]; // relative to my_local_rows
        
        int blk_start = mat.row_ptr[i];
        int blk_end = mat.row_ptr[i+1];
        
        for (int k = blk_start; k < blk_end; ++k) {
            int col_blk = mat.col_ind[k]; // local index of block
            int c_dim = block_sizes[col_blk];
            
            uint64_t handle = mat.blk_handles[k];
            const T* data = mat.arena.get_ptr(handle);
            
            // Check non-zeros in block
            // Assume dense blocks are fully non-zero for now, OR check explicit zeros?
            // Usually in BlockSparse, we treat the whole block as structural non-zeros.
            // But for PARDISO efficiency, we might want to drop explicit zeros?
            // Standard CSR conversion usually keeps stored values.
            
            for (int r = 0; r < r_dim; ++r) {
                 row_nnz[start_row + r] += c_dim;
            }
        }
    }
    
    // Prefix sum for row_ptr
    csr.row_ptr.resize(my_local_rows + 1);
    csr.row_ptr[0] = 0;
    for (int i = 0; i < my_local_rows; ++i) {
        csr.row_ptr[i+1] = csr.row_ptr[i] + row_nnz[i];
    }
    
    total_nnz = csr.row_ptr[my_local_rows];
    csr.col_ind.resize(total_nnz);
    csr.values.resize(total_nnz);
    
    // 4.2 Fill CSR
    std::fill(row_nnz.begin(), row_nnz.end(), 0); // Reset to use as current offset
    
    for (int i = 0; i < n_owned_blocks; ++i) {
        int r_dim = block_sizes[i];
        int start_row = block_local_offsets[i];
        
        int blk_start = mat.row_ptr[i];
        int blk_end = mat.row_ptr[i+1];
        
        // Sort blocks by column index? 
        // VBCSR col_ind is sorted by local block ID?
        // No, adj_ind in graph is sorted by LocalID, which usually doesn't mean GlobalID sorted.
        // PARDISO requires column indices to be sorted within each row.
        // We need to fetch blocks in Global Column order.
        
        // Let's collect blocks for this row block
        struct BlockInfo {
            int col_blk_idx;
            int global_scalar_start;
            const T* data;
            int c_dim;
        };
        std::vector<BlockInfo> row_blocks;
        
        for (int k = blk_start; k < blk_end; ++k) {
            int col_blk = mat.col_ind[k];
            int c_dim = block_sizes[col_blk];
            
            int global_col_start;
            if (col_blk < n_owned_blocks) {
                // Owned
                global_col_start = rank_global_offsets[rank] + block_local_offsets[col_blk];
            } else {
                // Ghost
                global_col_start = ghost_scalar_starts[col_blk - n_owned_blocks];
            }
            
            row_blocks.push_back({col_blk, global_col_start, mat.arena.get_ptr(mat.blk_handles[k]), c_dim});
        }
        
        // Sort by global column start
        std::sort(row_blocks.begin(), row_blocks.end(), [](const BlockInfo& a, const BlockInfo& b) {
            return a.global_scalar_start < b.global_scalar_start;
        });
        
        // Fill
        for (const auto& blk : row_blocks) {
            int c_dim = blk.c_dim;
            
            for (int r = 0; r < r_dim; ++r) {
                int row_idx = start_row + r;
                int base_ptr = csr.row_ptr[row_idx] + row_nnz[row_idx];
                
                for (int c = 0; c < c_dim; ++c) {
                    csr.col_ind[base_ptr + c] = blk.global_scalar_start + c;
                    // VBCSR is ColMajor
                    // value is data[c * r_dim + r]
                    csr.values[base_ptr + c] = blk.data[c * r_dim + r];
                }
                row_nnz[row_idx] += c_dim;
            }
        }
    }
    
    return csr;
}

} // namespace vbcsr

#endif
