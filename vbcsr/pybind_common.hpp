#pragma once
#include <pybind11/pybind11.h>
#include <mpi.h>
#include <cstdlib>

namespace py = pybind11;

// Helper to get MPI_Comm from python object (mpi4py)
inline MPI_Comm get_mpi_comm(py::object comm_obj) {
    int initialized;
    MPI_Initialized(&initialized);

    if (comm_obj.is_none()) {
        if (initialized) return MPI_COMM_WORLD;
        
        // If not initialized, we initialize it to support serial execution (size=1)
        // or execution where mpi4py is not used but MPI is needed by C++.
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
        return MPI_COMM_WORLD;
    }

    // If a communicator is provided, we MUST be initialized
    if (!initialized) {
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    }

    // Try to get 'py2f' method if it's an mpi4py communicator
    if (py::hasattr(comm_obj, "py2f")) {
        MPI_Fint f_handle = (MPI_Fint)comm_obj.attr("py2f")().cast<intptr_t>();
        return MPI_Comm_f2c(f_handle);
    }
    
    // Assume it's an integer handle
    MPI_Fint f_handle = (MPI_Fint)comm_obj.cast<intptr_t>();
    return MPI_Comm_f2c(f_handle);
}

inline void finalize_mpi() {
    int initialized, finalized;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized) {
        MPI_Finalize();
    }
}
