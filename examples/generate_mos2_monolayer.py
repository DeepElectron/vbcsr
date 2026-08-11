#!/usr/bin/env python3
"""Generate device-scale 1H-MoS2 structures for the VBCSR benchmark.

The default is a process-inspired MoS2 FET footprint: approximately 50 nm
along the transport direction by 650 nm across the device width, with a
28 nm channel and smooth tensile-strain peaks at the source/drain contact
edges.  The Gaussian strain profile is representative rather than a fit to a
specific device measurement.

Three structures for the paper can be generated with the same script::

    # Case A: flat reference
    python examples/generate_mos2_monolayer.py --profile flat -o mos2_fet_flat.extxyz

    # Case B: uniform 0.5% uniaxial (add --biaxial if desired)
    python examples/generate_mos2_monolayer.py --profile uniform --strain 0.005

    # Case C: contact-induced strain, 0.8% Gaussian amplitude (default)
    python examples/generate_mos2_monolayer.py -o mos2_fet_contact_strain.extxyz

    # Conservative periodic control: |epsilon_xx| <= 1%
    python examples/generate_mos2_monolayer.py --profile periodic --strain 0.01

Use ``--dry-run`` to inspect the commensurate size, atom count, and tight-
binding dimension without allocating the structure.  Lengths supplied on the
command line are in nm; ASE and the extended-XYZ output use angstrom.

The structure is periodic in-plane by default.  In the transport direction,
this represents a repeated contacted-device pitch and avoids artificial MoS2
edge states.  The contact Gaussians are periodized to numerical precision.
Use ``--finite`` only when explicit, under-coordinated edges are intended.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.io import write
from scipy.special import erf


ANGSTROM_PER_NM = 10.0
MO_ATOMIC_NUMBER = 42
S_ATOMIC_NUMBER = 16
VALID_PROFILES = ("contact", "periodic", "uniform", "flat")


@dataclass(frozen=True)
class StructurePlan:
    """Integer supercell dimensions and deformation parameters."""

    nx: int
    ny: int
    reference_length_x_angstrom: float
    reference_length_y_angstrom: float
    deformed_length_x_angstrom: float
    deformed_length_y_angstrom: float
    profile: str
    strain: float
    biaxial: bool
    source_edge_angstrom: float = 0.0
    drain_edge_angstrom: float = 0.0
    contact_sigma_angstrom: float = 0.0

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def n_atoms(self) -> int:
        return 6 * self.n_cells

    @property
    def n_mo(self) -> int:
        return 2 * self.n_cells

    @property
    def n_s(self) -> int:
        return 4 * self.n_cells

    @property
    def spinful_hamiltonian_dimension(self) -> int:
        """Dimension for the spinful 10-orbital Mo/6-orbital S model."""

        return 10 * self.n_mo + 6 * self.n_s

    @property
    def channel_length_angstrom(self) -> float:
        if self.profile != "contact":
            return 0.0
        return self.drain_edge_angstrom - self.source_edge_angstrom


def _integrated_gaussian(
    x: float, center: float, sigma: float
) -> float:
    """Integral from zero to x of a unit Gaussian centered at ``center``."""

    scale = math.sqrt(2.0) * sigma
    return sigma * math.sqrt(math.pi / 2.0) * (
        math.erf((x - center) / scale) - math.erf(-center / scale)
    )


def _periodic_image_count(length: float, sigma: float) -> int:
    """Images needed to suppress omitted Gaussian tails below roundoff."""

    return max(1, int(math.ceil(1.0 + 8.0 * sigma / length)))


def make_plan(
    size_nm: Sequence[float] = (50.0, 650.0),
    *,
    profile: str = "contact",
    strain: float = 0.008,
    biaxial: bool = False,
    channel_length_nm: float = 28.0,
    contact_sigma_nm: float = 3.0,
    contact_edges_nm: Optional[Sequence[float]] = None,
    lattice_constant_angstrom: float = 3.16,
) -> StructurePlan:
    """Plan the nearest commensurate orthorhombic MoS2 supercell.

    The first sheet dimension is the transport length and the second is the
    device width.  The rectangular six-atom cell has side lengths
    ``sqrt(3) * a`` and ``a``.

    For the contact profile, ``strain`` is the amplitude epsilon_0 in the
    periodized profile

        epsilon_xx(x) = epsilon_0 * sum_n [G(x - x_S - nL)
                                           + G(x - x_D - nL)],

    where each ``G`` is a unit-height Gaussian with standard deviation
    ``contact_sigma_nm``.  Images are retained until omitted tails are below
    floating-point roundoff.  Atomic x coordinates are displaced by the
    analytic integral, so ``du_x/dx = epsilon_xx`` and the periodic cell length
    changes consistently.

    The ``periodic`` profile is the zero-mean control

        u_x(x) = epsilon_max * L / (2 pi) * sin(2 pi x / L),

    for which ``epsilon_xx = epsilon_max * cos(2 pi x / L)``.
    """

    if len(size_nm) == 1:
        requested_x_nm = requested_y_nm = float(size_nm[0])
    elif len(size_nm) == 2:
        requested_x_nm, requested_y_nm = map(float, size_nm)
    else:
        raise ValueError("size_nm must contain one value or two values")

    if profile not in VALID_PROFILES:
        raise ValueError("profile must be one of: " + ", ".join(VALID_PROFILES))
    if requested_x_nm <= 0.0 or requested_y_nm <= 0.0:
        raise ValueError("sheet dimensions must be positive")
    if lattice_constant_angstrom <= 0.0:
        raise ValueError("lattice constant must be positive")
    if strain < 0.0:
        raise ValueError("only non-negative tensile strain is supported")
    if biaxial and profile != "uniform":
        raise ValueError("--biaxial is only meaningful for the uniform profile")

    cell_x = math.sqrt(3.0) * lattice_constant_angstrom
    cell_y = lattice_constant_angstrom
    requested_x = requested_x_nm * ANGSTROM_PER_NM
    requested_y = requested_y_nm * ANGSTROM_PER_NM
    nx = max(1, int(math.floor(requested_x / cell_x + 0.5)))
    ny = max(1, int(math.floor(requested_y / cell_y + 0.5)))
    reference_x = nx * cell_x
    reference_y = ny * cell_y

    applied_strain = 0.0 if profile == "flat" else strain
    source_edge = 0.0
    drain_edge = 0.0
    sigma = 0.0

    if profile == "contact":
        sigma = contact_sigma_nm * ANGSTROM_PER_NM
        if sigma <= 0.0:
            raise ValueError("contact Gaussian width must be positive")

        if contact_edges_nm is not None:
            if len(contact_edges_nm) != 2:
                raise ValueError("contact_edges_nm must contain source and drain edges")
            source_edge, drain_edge = (
                float(value) * ANGSTROM_PER_NM for value in contact_edges_nm
            )
        else:
            channel_length = channel_length_nm * ANGSTROM_PER_NM
            if channel_length <= 0.0 or channel_length >= reference_x:
                raise ValueError(
                    "channel length must be positive and shorter than the "
                    "commensurate transport length"
                )
            source_edge = 0.5 * (reference_x - channel_length)
            drain_edge = 0.5 * (reference_x + channel_length)

        if not 0.0 < source_edge < drain_edge < reference_x:
            raise ValueError(
                "contact edges must satisfy 0 < source < drain < transport length"
            )

        image_count = _periodic_image_count(reference_x, sigma)
        total_extension = 0.0
        for center in (source_edge, drain_edge):
            for image in range(-image_count, image_count + 1):
                total_extension += applied_strain * _integrated_gaussian(
                    reference_x, center + image * reference_x, sigma
                )
        deformed_x = reference_x + total_extension
        deformed_y = reference_y
    elif profile == "uniform":
        deformed_x = reference_x * (1.0 + applied_strain)
        deformed_y = reference_y * (
            1.0 + applied_strain if biaxial else 1.0
        )
    else:
        deformed_x = reference_x
        deformed_y = reference_y

    return StructurePlan(
        nx=nx,
        ny=ny,
        reference_length_x_angstrom=reference_x,
        reference_length_y_angstrom=reference_y,
        deformed_length_x_angstrom=deformed_x,
        deformed_length_y_angstrom=deformed_y,
        profile=profile,
        strain=applied_strain,
        biaxial=biaxial,
        source_edge_angstrom=source_edge,
        drain_edge_angstrom=drain_edge,
        contact_sigma_angstrom=sigma,
    )


def _strain_and_displacement(
    x: np.ndarray, plan: StructurePlan
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the local epsilon_xx and compatible x displacement."""

    if plan.profile == "flat":
        return np.zeros_like(x), np.zeros_like(x)
    if plan.profile == "uniform":
        return np.full_like(x, plan.strain), plan.strain * x
    if plan.profile == "periodic":
        wavevector = 2.0 * math.pi / plan.reference_length_x_angstrom
        phase = wavevector * x
        strain_xx = plan.strain * np.cos(phase)
        displacement_x = plan.strain * np.sin(phase) / wavevector
        return strain_xx, displacement_x

    sigma = plan.contact_sigma_angstrom
    source = plan.source_edge_angstrom
    drain = plan.drain_edge_angstrom
    scale = math.sqrt(2.0) * sigma
    prefactor = sigma * math.sqrt(math.pi / 2.0)
    image_count = _periodic_image_count(
        plan.reference_length_x_angstrom, sigma
    )
    strain_xx = np.zeros_like(x)
    displacement_x = np.zeros_like(x)
    for center in (source, drain):
        for image in range(-image_count, image_count + 1):
            image_center = center + image * plan.reference_length_x_angstrom
            strain_xx += plan.strain * np.exp(
                -0.5 * ((x - image_center) / sigma) ** 2
            )
            displacement_x += plan.strain * prefactor * (
                erf((x - image_center) / scale)
                - erf(-image_center / scale)
            )
    return strain_xx, displacement_x


def build_mos2_monolayer(
    plan: StructurePlan,
    *,
    lattice_constant_angstrom: float = 3.16,
    sulfur_half_height_angstrom: float = 1.565,
    vacuum_nm: float = 2.0,
    in_plane_pbc: bool = True,
    include_reference_positions: bool = False,
) -> Atoms:
    """Build the strained atomistic S-Mo-S trilayer described by ``plan``."""

    if sulfur_half_height_angstrom <= 0.0:
        raise ValueError("sulfur half-height must be positive")
    if vacuum_nm < 0.0:
        raise ValueError("vacuum thickness cannot be negative")

    rectangular_x = math.sqrt(3.0) * lattice_constant_angstrom
    rectangular_y = lattice_constant_angstrom
    expected_x = plan.nx * rectangular_x
    expected_y = plan.ny * rectangular_y
    if not math.isclose(
        expected_x, plan.reference_length_x_angstrom
    ) or not math.isclose(expected_y, plan.reference_length_y_angstrom):
        raise ValueError("plan and lattice constant are inconsistent")

    # Two primitive cells form this orthorhombic cell.  Top and bottom S
    # atoms have identical in-plane projections in a 1H monolayer (one layer
    # of the common 2H bulk polytype).
    basis_xy = np.array(
        [
            [0.0, 0.0],
            [rectangular_x / 3.0, 0.0],
            [rectangular_x / 3.0, 0.0],
            [rectangular_x / 2.0, rectangular_y / 2.0],
            [5.0 * rectangular_x / 6.0, rectangular_y / 2.0],
            [5.0 * rectangular_x / 6.0, rectangular_y / 2.0],
        ],
        dtype=np.float64,
    )
    basis_z = np.array(
        [
            0.0,
            sulfur_half_height_angstrom,
            -sulfur_half_height_angstrom,
            0.0,
            sulfur_half_height_angstrom,
            -sulfur_half_height_angstrom,
        ],
        dtype=np.float64,
    )
    basis_numbers = np.array(
        [
            MO_ATOMIC_NUMBER,
            S_ATOMIC_NUMBER,
            S_ATOMIC_NUMBER,
            MO_ATOMIC_NUMBER,
            S_ATOMIC_NUMBER,
            S_ATOMIC_NUMBER,
        ],
        dtype=np.int32,
    )
    basis_sublayer = np.array([0, 1, -1, 0, 1, -1], dtype=np.int8)

    ix, iy = np.meshgrid(
        np.arange(plan.nx, dtype=np.float64),
        np.arange(plan.ny, dtype=np.float64),
        indexing="ij",
    )
    origins = np.column_stack(
        (ix.ravel() * rectangular_x, iy.ravel() * rectangular_y)
    )
    reference_xy = (origins[:, None, :] + basis_xy[None, :, :]).reshape(-1, 2)
    intrinsic_z = np.tile(basis_z, plan.n_cells)

    strain_xx, displacement_x = _strain_and_displacement(
        reference_xy[:, 0], plan
    )
    if plan.profile == "uniform" and plan.biaxial:
        strain_yy = np.full_like(strain_xx, plan.strain)
        deformed_y = reference_xy[:, 1] * (1.0 + plan.strain)
    else:
        strain_yy = np.zeros_like(strain_xx)
        deformed_y = reference_xy[:, 1]

    vacuum_angstrom = vacuum_nm * ANGSTROM_PER_NM
    midplane_z = vacuum_angstrom + sulfur_half_height_angstrom
    positions = np.column_stack(
        (
            reference_xy[:, 0] + displacement_x,
            deformed_y,
            midplane_z + intrinsic_z,
        )
    )
    cell_z = 2.0 * vacuum_angstrom + 2.0 * sulfur_half_height_angstrom
    atoms = Atoms(
        numbers=np.tile(basis_numbers, plan.n_cells),
        positions=positions,
        cell=np.diag(
            [
                plan.deformed_length_x_angstrom,
                plan.deformed_length_y_angstrom,
                cell_z,
            ]
        ),
        pbc=[in_plane_pbc, in_plane_pbc, False],
    )

    # Store the in-plane strain tensor explicitly for LDOS/strain maps.  The
    # current profiles have no shear; local_strain is the maximum absolute
    # normal component and provides a convenient scalar plotting field.
    strain_xy = np.zeros_like(strain_xx)
    local_strain = np.maximum(np.abs(strain_xx), np.abs(strain_yy))
    atoms.new_array("local_strain", local_strain)
    atoms.new_array("strain_xx", strain_xx)
    atoms.new_array("strain_yy", strain_yy)
    atoms.new_array("strain_xy", strain_xy)
    atoms.new_array("sublayer", np.tile(basis_sublayer, plan.n_cells))

    device_region = np.zeros(plan.n_atoms, dtype=np.int8)
    if plan.profile == "contact":
        device_region[reference_xy[:, 0] <= plan.source_edge_angstrom] = -1
        device_region[reference_xy[:, 0] >= plan.drain_edge_angstrom] = 1
    atoms.new_array("device_region", device_region)

    if include_reference_positions:
        reference_positions = np.column_stack(
            (reference_xy, midplane_z + intrinsic_z)
        )
        atoms.new_array("reference_position", reference_positions)

    atoms.info.update(
        {
            "structure": "1H-MoS2_FET_footprint",
            "deformation_profile": plan.profile,
            "strain_mode": "biaxial" if plan.biaxial else "uniaxial_x",
            "strain_amplitude": plan.strain,
            "lattice_constant_A": lattice_constant_angstrom,
            "sulfur_half_height_A": sulfur_half_height_angstrom,
            "reference_length_x_A": plan.reference_length_x_angstrom,
            "reference_length_y_A": plan.reference_length_y_angstrom,
        }
    )
    if plan.profile == "contact":
        atoms.info.update(
            {
                "source_edge_A": plan.source_edge_angstrom,
                "drain_edge_A": plan.drain_edge_angstrom,
                "contact_sigma_A": plan.contact_sigma_angstrom,
                "contact_profile": "periodized_gaussian_image_sum",
                "profile_note": "representative_not_experimental_fit",
            }
        )
    return atoms


def _format_summary(plan: StructurePlan) -> str:
    lines = [
        "MoS2 FET structure",
        f"  profile            : {plan.profile}",
        f"  reference footprint: {plan.reference_length_x_angstrom / 10.0:.6f} x "
        f"{plan.reference_length_y_angstrom / 10.0:.6f} nm",
        f"  deformed footprint : {plan.deformed_length_x_angstrom / 10.0:.6f} x "
        f"{plan.deformed_length_y_angstrom / 10.0:.6f} nm",
        f"  rectangular cells : {plan.nx} x {plan.ny} = {plan.n_cells:,}",
        f"  atoms              : {plan.n_atoms:,} "
        f"(Mo {plan.n_mo:,}, S {plan.n_s:,})",
    ]
    if plan.profile == "contact":
        lines.extend(
            [
                f"  S/D edges          : {plan.source_edge_angstrom / 10.0:.6f}, "
                f"{plan.drain_edge_angstrom / 10.0:.6f} nm",
                f"  channel length     : {plan.channel_length_angstrom / 10.0:.6f} nm",
                f"  Gaussian sigma     : {plan.contact_sigma_angstrom / 10.0:.6f} nm",
                f"  strain amplitude   : {100.0 * plan.strain:.6f}%",
            ]
        )
    elif plan.profile == "uniform":
        mode = "biaxial" if plan.biaxial else "uniaxial x"
        lines.append(f"  uniform strain     : {100.0 * plan.strain:.6f}% ({mode})")
    elif plan.profile == "periodic":
        lines.append(
            f"  max |strain|       : {100.0 * plan.strain:.6f}% "
            "(sinusoidal displacement)"
        )
    else:
        lines.append("  strain             : 0.000000%")
    lines.append(
        f"  spinful TB size    : {plan.spinful_hamiltonian_dimension:,} "
        "(10 orbitals/Mo, 6 orbitals/S)"
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profile",
        choices=VALID_PROFILES,
        default="contact",
        help="strain profile to generate (default: contact)",
    )
    parser.add_argument(
        "--size-nm",
        nargs="+",
        type=float,
        default=[50.0, 650.0],
        metavar="L",
        help="transport length and width in nm (default: 50 650)",
    )
    parser.add_argument(
        "--strain",
        type=float,
        help="strain fraction (defaults: contact 0.008, periodic 0.01, uniform 0.005)",
    )
    parser.add_argument(
        "--biaxial",
        action="store_true",
        help="apply uniform strain along both x and y (uniform profile only)",
    )
    parser.add_argument(
        "--channel-length-nm",
        type=float,
        default=28.0,
        help="distance between centered contact edges in nm (default: 28)",
    )
    parser.add_argument(
        "--contact-edges-nm",
        nargs=2,
        type=float,
        metavar=("SOURCE", "DRAIN"),
        help="explicit source and drain edge positions in nm",
    )
    parser.add_argument(
        "--contact-sigma-nm",
        type=float,
        default=3.0,
        help="standard deviation of each contact strain peak (default: 3 nm)",
    )
    parser.add_argument(
        "--lattice-constant-angstrom",
        type=float,
        default=3.16,
        help="MoS2 in-plane lattice constant (default: 3.16 A)",
    )
    parser.add_argument(
        "--sulfur-half-height-angstrom",
        type=float,
        default=1.565,
        help="distance of each S plane from the Mo plane (default: 1.565 A)",
    )
    parser.add_argument(
        "--vacuum-nm",
        type=float,
        default=2.0,
        help="vacuum below and above the sheet (default: 2 nm)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="extended-XYZ output path (default: derived from profile and size)",
    )
    parser.add_argument(
        "--include-reference-positions",
        action="store_true",
        help="store pristine coordinates as a per-atom reference_position array",
    )
    parser.add_argument(
        "--finite",
        action="store_true",
        help="use exposed finite edges; their chemistry is not reparameterized",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report dimensions and atom counts without allocating or writing atoms",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(args.size_nm) not in (1, 2):
        raise SystemExit("--size-nm accepts either one value or two values")

    if args.strain is None:
        if args.profile == "uniform":
            strain = 0.005
        elif args.profile == "periodic":
            strain = 0.01
        else:
            strain = 0.008
    else:
        strain = args.strain
    if args.profile == "flat":
        strain = 0.0

    plan = make_plan(
        size_nm=args.size_nm,
        profile=args.profile,
        strain=strain,
        biaxial=args.biaxial,
        channel_length_nm=args.channel_length_nm,
        contact_sigma_nm=args.contact_sigma_nm,
        contact_edges_nm=args.contact_edges_nm,
        lattice_constant_angstrom=args.lattice_constant_angstrom,
    )
    print(_format_summary(plan))
    if args.finite:
        print(
            "warning: finite MoS2 edges are under-coordinated; bulk-fitted SK "
            "parameters may not describe their edge states quantitatively",
            file=sys.stderr,
        )
    if args.dry_run:
        return

    output = args.output
    if output is None:
        requested_x = args.size_nm[0]
        requested_y = args.size_nm[-1]
        mode = "_biaxial" if args.biaxial else ""
        output = Path(
            f"mos2_fet_{args.profile}{mode}_{requested_x:g}x{requested_y:g}nm.extxyz"
        )
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing file: {output} (use --force)")

    atoms = build_mos2_monolayer(
        plan,
        lattice_constant_angstrom=args.lattice_constant_angstrom,
        sulfur_half_height_angstrom=args.sulfur_half_height_angstrom,
        vacuum_nm=args.vacuum_nm,
        in_plane_pbc=not args.finite,
        include_reference_positions=args.include_reference_positions,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, atoms, format="extxyz")
    sampled_max_strain = float(atoms.arrays["local_strain"].max(initial=0.0))
    print(f"  sampled max strain : {100.0 * sampled_max_strain:.6f}%")
    print(f"  wrote               : {output.resolve()}")


if __name__ == "__main__":
    main()
