import ase.data
import ase.io
import numpy as np

from . import vbcsr_core

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def _default_comm(comm):
    if comm is not None or MPI is None:
        return comm
    return MPI.COMM_WORLD


def _normalize_pbc(pbc):
    if isinstance(pbc, bool):
        return [pbc, pbc, pbc]
    values = list(pbc)
    if len(values) == 1:
        return [bool(values[0]), bool(values[0]), bool(values[0])]
    if len(values) != 3:
        raise ValueError("pbc must be a bool or a sequence of length 1 or 3")
    return [bool(value) for value in values]


class AtomicData(vbcsr_core.AtomicData):
    """
    Python wrapper for ``vbcsr_core.AtomicData``.

    Dictionary inputs for ``r_max`` and ``type_norb`` may use atomic numbers,
    chemical symbols, or a ``"default"`` fallback.
    """

    @classmethod
    def from_points(cls, pos, z, cell, pbc, r_max, type_norb=1, comm=None):
        comm = _default_comm(comm)

        positions = np.ascontiguousarray(np.asarray(pos, dtype=np.float64).reshape(-1, 3))
        atomic_numbers = np.ascontiguousarray(np.asarray(z, dtype=np.int32).reshape(-1))
        if positions.shape[0] != atomic_numbers.size:
            raise ValueError("pos and z must describe the same number of atoms")

        if comm is not None:
            local_unique = np.unique(atomic_numbers)
            gathered = comm.allgather(local_unique)
            sorted_unique_z = np.unique(np.concatenate(gathered)) if gathered else np.empty((0,), dtype=np.int32)
        else:
            sorted_unique_z = np.unique(atomic_numbers)

        def parse_param(param, name, dtype):
            values = np.zeros(len(sorted_unique_z), dtype=dtype)

            if np.isscalar(param):
                values.fill(param)
                return values

            if isinstance(param, dict):
                default_value = param.get("default")
                if default_value is not None:
                    values.fill(default_value)

                for idx, atomic_number in enumerate(sorted_unique_z):
                    if atomic_number in param:
                        values[idx] = param[atomic_number]
                        continue

                    symbol = ase.data.chemical_symbols[int(atomic_number)]
                    if symbol in param:
                        values[idx] = param[symbol]
                        continue

                    if default_value is None:
                        raise ValueError(
                            f"{name} missing for Z={atomic_number} ({ase.data.chemical_symbols[int(atomic_number)]})"
                        )
                return values

            array = np.asarray(param)
            if array.ndim == 1 and array.size == 1:
                values.fill(array.reshape(-1)[0])
                return values
            if array.ndim != 1 or array.size != len(sorted_unique_z):
                raise ValueError(
                    f"{name} length {array.size} does not match the number of unique atomic types {len(sorted_unique_z)}"
                )
            return np.ascontiguousarray(array.astype(dtype, copy=False))

        r_max_vec = parse_param(r_max, "r_max", np.float64)
        type_norb_vec = parse_param(type_norb, "type_norb", np.int32)
        cell_array = np.ascontiguousarray(np.asarray(cell, dtype=np.float64).reshape(3, 3))

        return super().from_points(
            positions,
            atomic_numbers,
            cell_array,
            _normalize_pbc(pbc),
            r_max_vec,
            type_norb_vec,
            comm,
        )

    @classmethod
    def from_distributed(
        cls,
        n_atom,
        N_atom,
        atom_offset,
        n_edge,
        N_edge,
        atom_index,
        atom_type,
        edge_index,
        type_norb,
        edge_shift,
        cell,
        pos,
        atomic_numbers=None,
        comm=None,
    ):
        comm = _default_comm(comm)

        atom_index = np.ascontiguousarray(np.asarray(atom_index, dtype=np.int32).reshape(-1))
        atom_type = np.ascontiguousarray(np.asarray(atom_type, dtype=np.int32).reshape(-1))
        edge_index = np.ascontiguousarray(np.asarray(edge_index, dtype=np.int32).reshape(-1, 2))
        type_norb = np.ascontiguousarray(np.asarray(type_norb, dtype=np.int32).reshape(-1))
        edge_shift = np.ascontiguousarray(np.asarray(edge_shift, dtype=np.int32).reshape(-1, 3))
        cell = np.ascontiguousarray(np.asarray(cell, dtype=np.float64).reshape(3, 3))
        pos = np.ascontiguousarray(np.asarray(pos, dtype=np.float64).reshape(-1, 3))

        if atom_index.size != int(n_atom):
            raise ValueError("atom_index size must equal n_atom")
        if atom_type.size != int(n_atom):
            raise ValueError("atom_type size must equal n_atom")
        if pos.shape[0] != int(n_atom):
            raise ValueError("pos must have shape (n_atom, 3)")
        if edge_index.shape[0] != int(n_edge):
            raise ValueError("edge_index must have shape (n_edge, 2)")
        if edge_shift.shape[0] != int(n_edge):
            raise ValueError("edge_shift must have shape (n_edge, 3)")

        atomic_number_array = None
        if atomic_numbers is not None:
            atomic_number_array = np.ascontiguousarray(np.asarray(atomic_numbers, dtype=np.int32).reshape(-1))
            if atomic_number_array.size != int(n_atom):
                raise ValueError("atomic_numbers size must equal n_atom")

        return super().from_distributed(
            int(n_atom),
            int(N_atom),
            int(atom_offset),
            int(n_edge),
            int(N_edge),
            atom_index,
            atom_type,
            edge_index,
            type_norb,
            edge_shift,
            cell,
            pos,
            atomic_number_array,
            comm,
        )

    @classmethod
    def from_ase(cls, atoms, r_max, type_norb=1, comm=None):
        pbc = atoms.pbc
        return cls.from_points(
            atoms.get_positions(),
            atoms.get_atomic_numbers(),
            atoms.get_cell(),
            _normalize_pbc(pbc),
            r_max,
            type_norb,
            _default_comm(comm),
        )

    @classmethod
    def from_file(cls, filename, r_max, type_norb=1, comm=None, format=None):
        comm = _default_comm(comm)
        if comm is None:
            atoms = ase.io.read(filename, format=format)
            return cls.from_ase(atoms, r_max, type_norb, comm=None)

        rank = comm.Get_rank()
        atoms = ase.io.read(filename, format=format) if rank == 0 else None

        if rank == 0:
            pos = atoms.get_positions()
            atomic_numbers = atoms.get_atomic_numbers()
            cell = atoms.get_cell()
            pbc = _normalize_pbc(atoms.pbc)
        else:
            pos = np.empty((0, 3), dtype=np.float64)
            atomic_numbers = np.empty((0,), dtype=np.int32)
            cell = np.zeros((3, 3), dtype=np.float64)
            pbc = [False, False, False]

        cell = comm.bcast(np.asarray(cell, dtype=np.float64), root=0)
        pbc = comm.bcast(_normalize_pbc(pbc), root=0)
        return cls.from_points(pos, atomic_numbers, cell, pbc, r_max, type_norb, comm=comm)
