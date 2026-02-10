from . import vbcsr_core
import numpy as np
import ase.data
import ase.io
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class AtomicData(vbcsr_core.AtomicData):
    """
    Wrapper class for vbcsr_core.AtomicData with enhanced Python support.
    Handles dictionary inputs for r_max/type_norb with support for atomic numbers
    and chemical symbols.
    """

    @classmethod
    def from_points(cls, pos, z, cell, pbc, r_max, type_norb=1, comm=None):
        """
        Create AtomicData from atomic positions and numbers.
        
        Args:
            pos: (N, 3) positions
            z: (N,) atomic numbers
            cell: (3, 3) unit cell
            pbc: (3,) PBC flags
            r_max: float or dict. Cutoff radius. Dict keys can be atomic number (int) or symbol (str).
            type_norb: int or dict. Orbitals per type. Dict keys similar to r_max.
            comm: MPI communicator (mpi4py). If None, defaults to MPI.COMM_WORLD if installed.
        """
        if comm is None and MPI is not None:
            comm = MPI.COMM_WORLD
            
        # Ensure imports
        
        # 1. Determine Unique Zs to map inputs to vectors
        # Logic: We need to match C++'s mapping: sorted unique Zs.
        # Data might be distributed or on Rank 0.
        # vbcsr C++ from_points typically expects data on Rank 0 for initial partition?
        # Or if passed distributed, it gathers?
        # Standard usage: Pass global on Rank 0.
        # But we verify locally.
        
        local_z = np.asarray(z, dtype=np.int32).reshape(-1)
        
        # We need global unique Zs
        if comm is not None:
             # Gather all Zs? Expensive if large.
             # Gather only unique local Zs.
             local_unique = np.unique(local_z)
             all_unique = comm.allgather(local_unique) # list of arrays
             global_unique = np.unique(np.concatenate(all_unique))
             sorted_unique_z = np.sort(global_unique)
        else:
             sorted_unique_z = np.unique(local_z)
             
        # 2. Parse r_max and type_norb into vectors
        def parse_param(param, name, output_type=float):
            vec = np.zeros(len(sorted_unique_z), dtype=output_type)
            if np.isscalar(param) or (isinstance(param, list) and len(param) == 1):
                vec[:] = param
            elif isinstance(param, dict):
                # Check default
                default_val = param.get("default", None)
                if default_val is not None:
                    vec[:] = default_val
                    
                for i, z_val in enumerate(sorted_unique_z):
                    # Try keys: Z (int), Symbol (str)
                    val = None
                    if z_val in param:
                        val = param[z_val]
                    else:
                        sym = ase.data.chemical_symbols[z_val]
                        if sym in param:
                            val = param[sym]
                    
                    if val is not None:
                        vec[i] = val
                    elif default_val is None:
                        raise ValueError(f"{name} missing for Z={z_val} ({ase.data.chemical_symbols[z_val]})")
            elif isinstance(param, (list, np.ndarray, tuple)):
                 # Assign directly if size matches?? 
                 # Risky if user doesn't know mapping.
                 # Better to allow only if user explicitly asks? 
                 # But previous C++ binding allowed list.
                 # We'll allow if len matches.
                 if len(param) == len(sorted_unique_z):
                     vec[:] = param
                 else:
                     raise ValueError(f"{name} list length {len(param)} != number of unique types {len(sorted_unique_z)}")
            else:
                 raise ValueError(f"{name} invalid type: {type(param)}")
            return vec

        r_max_vec = parse_param(r_max, "r_max", float)
        type_norb_vec = parse_param(type_norb, "type_norb", int)
        
        # 3. Call C++ static method
        # We pass python objects checkable by Pybind
        # C++ binding expects: pos, z, cell, pbc, r_max_vec (list), type_norb_vec (list), comm
        return super().from_points(pos, z, cell, pbc, r_max_vec, type_norb_vec, comm)


    @classmethod
    def from_ase(cls, atoms, r_max, type_norb=1, comm=None):
        """
        Create AtomicData from ASE Atoms.
        """
        pos = atoms.get_positions()
        z = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.pbc
        # Handle pbc being bool or array
        if isinstance(pbc, bool): pbc = [pbc]*3
        
        # Note: input atoms should be same on all ranks OR on Rank 0 (with others empty/None?)
        # ase.io.read usually returns atoms on all ranks if run on all.
        # But usually we run read on Rank 0.
        # We assume `atoms` is valid locally.
        
        return cls.from_points(pos, z, cell, pbc, r_max, type_norb, comm)

    @classmethod
    def from_file(cls, filename, r_max, type_norb=1, comm=None, format=None):
        """
        Create AtomicData from file using ASE.
        """
        if comm is None and MPI is not None:
            comm = MPI.COMM_WORLD
            
        rank = 0
        if comm is not None:
            rank = comm.Get_rank()
            
        atoms = None
        if rank == 0:
            atoms = ase.io.read(filename, format=format)
            
        # Broadcast atoms to all ranks? 
        # C++ from_points usually handles Rank 0 -> All distribution.
        # So we can pass `atoms` on Rank 0 and None on others?
        # Python `from_ase` calls `from_points`.
        # My `from_points` wrapper calculates `unique_z`.
        # If `atoms` is None on rank 1, `z` is None/empty.
        # `unique_z` logic: `allgather` of local unique.
        # If Rank 1 has no atoms, it contributes empty set.
        # Rank 0 contributes all types.
        # Union is correct.
        
        # But `super().from_points` expects valid numpy arrays or compatible.
        # If `z` is None on Rank 1, Pybind might complain if it expects array?
        # `numpy_to_vector` handles empty?
        # I should ensure on non-root ranks, we pass empty arrays.
        
        if comm is not None:
             # If atoms is None (non-root), make dummy empty atoms or pass empty arrays
             if rank != 0:
                 # Need to pass empty arrays to from_points
                 pos = np.empty((0,3))
                 z = np.empty((0,), dtype=int)
                 cell = np.zeros((3,3)) # Cell? Cell usually needed globally?
                 pbc = [False]*3
             else:
                 pos = atoms.get_positions()
                 z = atoms.get_atomic_numbers()
                 cell = atoms.get_cell()
                 pbc = atoms.pbc
                 # Broadacst cell and pbc? 
                 # C++ AtomicData constructor (partitioning) usually needs cell on all ranks?
                 # construct_final_object (line 1213) broadcasts cell.
                 # process_input_rank0 uses cell.
                 # So maybe only Rank 0 needs cell?
                 
             # Safer to broadcast cell/pbc from Rank 0 so all ranks pass consistent metadata
             cell = comm.bcast(cell, root=0)
             pbc = comm.bcast(pbc, root=0)
             
             return cls.from_points(pos, z, cell, pbc, r_max, type_norb, comm)
        else:
             return cls.from_ase(atoms, r_max, type_norb, comm)
