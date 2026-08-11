"""Sapra--Viswanatha--Sarma InP tight-binding parameters.

The source is Table I of J. Phys. D 36, 1595 (2003),
https://arxiv.org/abs/cond-mat/0308038.

The orbital order is chosen to match the real-harmonic rotation used here:
``p = (py, pz, px)`` and
``d = (dxy, dyz, dz2, dzx, dx2-y2)``.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np


H = "H"
P = "P"
IN = "In"

ATOMIC_NUMBERS: Mapping[str, int] = {H: 1, P: 15, IN: 49}
BASIS: Mapping[str, Tuple[str, ...]] = {
    H: ("s",),
    P: ("s", "py", "pz", "px", "dxy", "dyz", "dz2", "dzx", "dx2-y2"),
    IN: ("s", "py", "pz", "px"),
}
NORB: Mapping[str, int] = {symbol: len(labels) for symbol, labels in BASIS.items()}
NORB_TO_SYMBOL: Mapping[int, str] = {value: key for key, value in NORB.items()}


@dataclass(frozen=True)
class InPParameters:
    lattice_constant: float
    in_h_bond_length: float
    onsite: Mapping[str, np.ndarray]
    pair_channels: Mapping[Tuple[str, str], Mapping[Tuple[int, int, int], float]]
    paper_in_h_values: Tuple[float, float, float]
    passivation_note: str


def _onsite(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


# A channel key is (row angular momentum, column angular momentum, |m|),
# with |m| = 0/1/2 denoting sigma/pi/delta.
PAIR_CHANNELS: Dict[Tuple[str, str], Dict[Tuple[int, int, int], float]] = {
    (IN, P): {
        (0, 0, 0): -1.43,  # ss sigma
        (0, 1, 0): 2.19,   # s(In)-p(P) sigma
        (0, 2, 0): -2.72,  # s(In)-d(P) sigma
        # Table I gives the aligned p(In)-s(P) matrix element as -1.63 eV.
        # The real-harmonic builder applies the odd-parity p-s direction sign,
        # so its radial channel is +1.63 eV.
        (1, 0, 0): 1.63,
        (1, 1, 0): 3.35,   # pp sigma
        (1, 1, 1): -0.66,  # pp pi
        (1, 2, 0): -3.38,  # pd sigma
        (1, 2, 1): 3.35,   # pd pi
    },
    (IN, IN): {
        (0, 0, 0): -0.21,
        (0, 1, 0): 0.00,
        (1, 0, 0): 0.00,
        (1, 1, 0): 0.14,
        (1, 1, 1): -0.01,
    },
    (P, P): {
        (0, 0, 0): -0.01,
        (0, 1, 0): 0.14,
        (1, 0, 0): 0.14,
        (1, 1, 0): 0.70,
        (1, 1, 1): -0.02,
    },
    # Table I prints the passivation row under ss/sp/sd headings even though H
    # has only s and surface In has only sp3.  We interpret the first two as
    # host s/p-to-H-s channels; the source does not remove this ambiguity.
    (IN, H): {
        (0, 0, 0): -2.944,
        (1, 0, 0): 2.76,
    },
}


PAPER_PARAMETERS = InPParameters(
    lattice_constant=5.861,
    # The paper does not specify a geometric In-H distance.  Hoppings are
    # distance independent, so this nominal value only places the passivant
    # along the missing tetrahedral bond and builds the neighbour graph.
    in_h_bond_length=1.80,
    onsite={
        IN: _onsite(-1.53, 3.92, 3.92, 3.92),
        P: _onsite(-10.24, -0.63, -0.63, -0.63, 16.62, 16.62, 16.62, 16.62, 16.62),
        H: _onsite(-0.7412),
    },
    pair_channels=PAIR_CHANNELS,
    paper_in_h_values=(-2.944, 2.76, -1.36),
    passivation_note=(
        "Table I prints three In-H values under ss/sp/sd although H has only s "
        "and In has no d. This implementation reads the first two as host "
        "s/p-to-H-s (-2.944 and 2.76 eV); the paper does not define the "
        "rectangular passivation block unambiguously."
    ),
)


def atomwise_cutoff_radii(parameters: InPParameters = PAPER_PARAMETERS) -> Mapping[str, float]:
    """Tight atom radii for the AtomicData pair-sum cutoff convention."""
    second_neighbor = parameters.lattice_constant / np.sqrt(2.0)
    bulk_radius = 0.5 * second_neighbor + 0.01
    # H + In must include the nominal In-H bond but avoid H-H edges.
    h_radius = max(0.01, parameters.in_h_bond_length - bulk_radius + 0.01)
    return {H: h_radius, P: bulk_radius, IN: bulk_radius}
