#ifndef VBCSR_DETAIL_DISTRIBUTED_MPI_UTILS_HPP
#define VBCSR_DETAIL_DISTRIBUTED_MPI_UTILS_HPP

#include <mpi.h>
#include <vector>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace vbcsr {

/**
 * @brief A safe wrapper for MPI_Alltoallv that handles counts and displacements exceeding INT_MAX.
 * 
 * This function uses point-to-point communication (MPI_Isend/MPI_Irecv) to transfer data in 
 * chunks of at most INT_MAX bytes.
 */
inline void safe_alltoallv(const void* sendbuf, const std::vector<size_t>& sendcounts, 
                          const std::vector<size_t>& sdispls, MPI_Datatype sendtype,
                          void* recvbuf, const std::vector<size_t>& recvcounts, 
                          const std::vector<size_t>& rdispls, MPI_Datatype recvtype,
                          MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int send_type_size, recv_type_size;
    MPI_Type_size(sendtype, &send_type_size);
    MPI_Type_size(recvtype, &recv_type_size);

    const size_t MAX_INT = static_cast<size_t>(std::numeric_limits<int>::max());
    
    std::vector<MPI_Request> requests;
    requests.reserve(size * 2);

    // Post receives
    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] == 0) continue;
        
        size_t total_bytes = recvcounts[i] * recv_type_size;
        char* ptr = static_cast<char*>(recvbuf) + rdispls[i] * recv_type_size;
        
        if (total_bytes <= MAX_INT) {
            MPI_Request req;
            MPI_Irecv(ptr, static_cast<int>(total_bytes), MPI_BYTE, i, 0, comm, &req);
            requests.push_back(req);
        } else {
            // Split into chunks
            size_t offset = 0;
            while (offset < total_bytes) {
                size_t chunk_size = std::min(total_bytes - offset, MAX_INT);
                MPI_Request req;
                MPI_Irecv(ptr + offset, static_cast<int>(chunk_size), MPI_BYTE, i, 0, comm, &req);
                requests.push_back(req);
                offset += chunk_size;
            }
        }
    }

    // Post sends
    for (int i = 0; i < size; ++i) {
        if (sendcounts[i] == 0) continue;
        
        size_t total_bytes = sendcounts[i] * send_type_size;
        const char* ptr = static_cast<const char*>(sendbuf) + sdispls[i] * send_type_size;
        
        if (total_bytes <= MAX_INT) {
            MPI_Request req;
            MPI_Isend(ptr, static_cast<int>(total_bytes), MPI_BYTE, i, 0, comm, &req);
            requests.push_back(req);
        } else {
            // Split into chunks
            size_t offset = 0;
            while (offset < total_bytes) {
                size_t chunk_size = std::min(total_bytes - offset, MAX_INT);
                MPI_Request req;
                MPI_Isend(ptr + offset, static_cast<int>(chunk_size), MPI_BYTE, i, 0, comm, &req);
                requests.push_back(req);
                offset += chunk_size;
            }
        }
    }

    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

} // namespace vbcsr

#endif // VBCSR_DETAIL_DISTRIBUTED_MPI_UTILS_HPP
