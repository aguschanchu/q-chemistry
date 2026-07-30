"""RDM engine for the H2O/STO-3G reproduction

Conventions:

- Interleaved spin orbitals (OpenFermion order): spatial p maps 
  to 2p (up) and 2p + 1 (down).
- The M_S = 0 sector basis holds the 441 = C(7,5)^2 determinants in the order
  inherited from fermionic_mbody.FixedBasis(14, 10), index maps are always
  derived from the stored bitmasks.
- rho1[i, j] = <c^dag_j c_i>, pair basis (masks {i1 < i2}, {j1 < j2}):
  rho2[I, J] = <c^dag_j1 c^dag_j2 c_i2 c_i1>.
- RDMs are unnormalized: Tr rho_up = 1, Tr rho1_up = 5, Tr rho2_upup = 10
  (21-dim i < j), Tr rho2_updown = 25 (49-dim product basis).
- Entropies in bits, -sum lambda log2 lambda.
- Negativities are nonnegative: minus the sum of the negative eigenvalues of
  the corresponding partial transpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import scipy.sparse as sp
from fermionic_mbody import FixedBasis, rho_m, rho_m_gen

N_SPATIAL = 7
N_SPIN_ORBITALS = 14
N_ELECTRONS = 10
N_UP = 5
N_DOWN = 5
DIM_UP = 21          # C(7, 5): up (or down) Schmidt factor dimension
DIM_SECTOR = 441     # C(7,5)^2: fixed M_S = 0 sector
DIM_FULL = 1001      # C(14, 10): fixed-N sector 
DIM_PAIR = 91        # C(14, 2): antisymmetric two-body basis
DIM_UPUP = 21        # C(7, 2): same-spin block
DIM_UPDOWN = 49      # 7 * 7: opposite-spin product block
UP_BIT_MASK = sum(1 << (2 * p) for p in range(N_SPATIAL))

EIG_CUTOFF = 1e-12   # spectral cutoff for entropies / negativities

__all__ = [
    "SectorBasis", "SpinSplitMaps", "PairBlockMaps", "RDMGenerators",
    "build_sector_basis", "build_spin_split_maps", "build_pair_block_maps",
    "build_generators", "split_det", "coeff_matrix", "rho_up_from_states",
    "rho_down_from_states", "sector_density_product", "total_negativity",
    "pure_total_negativity", "reduced_blocks", "rho2_updown_product",
    "same_spin_extended", "same_spin_negativity", "opposite_spin_negativity",
    "hermitian_spectrum", "spectral_entropy_bits", "shannon_bits",
    "weight_entropy_bits", "expected_traces",
]


# --------------------------------------------------------------------------- #
# Bases and index maps
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SectorBasis:
    """Determinant bases for the fixed M_S = 0 sector and its up factor."""

    basis_full: FixedBasis   # FixedBasis(14, 10), size 1001
    basis: FixedBasis        # M_S = 0 subset, size 441
    basis_up: FixedBasis     # FixedBasis(7, 5), size 21 (spatial masks)
    mask_to_index: Mapping[int, int]      # sector bitmask -> sector index
    up_mask_to_index: Mapping[int, int]   # 7-bit spatial mask -> up index


def build_sector_basis() -> SectorBasis:
    """Build the bases and public mask->index dictionaries."""
    basis_full = FixedBasis(N_SPIN_ORBITALS, N_ELECTRONS)
    if basis_full.size != DIM_FULL:
        raise AssertionError(f"unexpected fixed-N dimension {basis_full.size}")
    keep = [
        index for index, mask in enumerate(basis_full.num_ele)
        if bin(int(mask) & UP_BIT_MASK).count("1") == N_UP
    ]
    basis = FixedBasis.from_subset(basis_full, keep)
    if basis.size != DIM_SECTOR:
        raise AssertionError(f"unexpected M_S=0 dimension {basis.size}")
    basis_up = FixedBasis(N_SPATIAL, N_UP)
    if basis_up.size != DIM_UP:
        raise AssertionError(f"unexpected up-factor dimension {basis_up.size}")
    mask_to_index = {int(mask): k for k, mask in enumerate(basis.num_ele)}
    up_mask_to_index = {int(mask): k for k, mask in enumerate(basis_up.num_ele)}
    return SectorBasis(basis_full, basis, basis_up, mask_to_index, up_mask_to_index)


def split_det(mask: int) -> tuple[int, int, int]:
    """
    Split an interleaved determinant into (up_mask, down_mask, sign).

    up_mask/down_mask are 7-bit spatial occupation masks. sign is
    the fermionic phase of reordering the ascending interleaved creation string
    into (all up, ascending)(all down, ascending).
    """
    up_mask = 0
    down_mask = 0
    inversions = 0
    ups_seen = 0
    total_ups = bin(mask & UP_BIT_MASK).count("1")
    for bit in range(N_SPIN_ORBITALS):
        if not (mask >> bit) & 1:
            continue
        if bit % 2 == 0:
            up_mask |= 1 << (bit // 2)
            ups_seen += 1
        else:
            down_mask |= 1 << (bit // 2)
            inversions += total_ups - ups_seen  # up bits still to the right
    sign = -1 if inversions % 2 else 1
    return up_mask, down_mask, sign


@dataclass(frozen=True)
class SpinSplitMaps:
    """Bijection between the 441 sector determinants and (up, down) pairs."""

    k_of: np.ndarray      # (21, 21) int   sector index of (i_up, j_down)
    sign_of: np.ndarray   # (21, 21) float split phase of that determinant
    reorder: np.ndarray   # (441,) int     reorder[i*21 + j] = k_of[i, j]
    signs: np.ndarray     # (441,) float   flattened sign_of


def build_spin_split_maps(sector: SectorBasis) -> SpinSplitMaps:
    k_of = np.full((DIM_UP, DIM_UP), -1, dtype=np.int64)
    sign_of = np.zeros((DIM_UP, DIM_UP))
    for k, mask in enumerate(sector.basis.num_ele):
        up_mask, down_mask, sign = split_det(int(mask))
        i = sector.up_mask_to_index[up_mask]
        j = sector.up_mask_to_index[down_mask]
        if k_of[i, j] != -1:
            raise AssertionError("determinant split is not a bijection")
        k_of[i, j] = k
        sign_of[i, j] = sign
    if (k_of < 0).any():
        raise AssertionError("determinant split does not cover the product basis")
    return SpinSplitMaps(
        k_of=k_of,
        sign_of=sign_of,
        reorder=k_of.reshape(-1).copy(),
        signs=sign_of.reshape(-1).copy(),
    )


# --------------------------------------------------------------------------- #
# Schmidt route: rho_up / rho_down / total negativity
# --------------------------------------------------------------------------- #

def coeff_matrix(state: np.ndarray, maps: SpinSplitMaps) -> np.ndarray:
    """Schmidt coefficient matrix Gamma: |Psi> = sum Gamma_ij |i>|j>."""
    return (maps.signs * np.asarray(state)[maps.reorder]).reshape(DIM_UP, DIM_UP)


def _as_state_list(states: np.ndarray) -> np.ndarray:
    array = np.atleast_2d(np.asarray(states))
    if array.shape[1] != DIM_SECTOR:
        raise ValueError(f"states must have {DIM_SECTOR} components")
    return array


def rho_up_from_states(states: np.ndarray, weights: Iterable[float],
                       maps: SpinSplitMaps) -> np.ndarray:
    """Up-factor reduced state, trace 1: sum_n w_n Gamma_n Gamma_n^dagger."""
    result = np.zeros((DIM_UP, DIM_UP), dtype=np.complex128)
    for weight, state in zip(weights, _as_state_list(states)):
        gamma = coeff_matrix(state, maps)
        result += weight * (gamma @ gamma.conj().T)
    return result.real if np.abs(result.imag).max() < 1e-14 else result


def rho_down_from_states(states: np.ndarray, weights: Iterable[float],
                         maps: SpinSplitMaps) -> np.ndarray:
    """Down-factor reduced state, trace 1: sum_n w_n Gamma_n^T Gamma_n^*."""
    result = np.zeros((DIM_UP, DIM_UP), dtype=np.complex128)
    for weight, state in zip(weights, _as_state_list(states)):
        gamma = coeff_matrix(state, maps)
        result += weight * (gamma.T @ gamma.conj())
    return result.real if np.abs(result.imag).max() < 1e-14 else result


def sector_density_product(states: np.ndarray, weights: Iterable[float],
                           maps: SpinSplitMaps) -> np.ndarray:
    """
    Density matrix in the (i_up * 21 + j_down) product order.
    """
    vectors = np.array([
        maps.signs * state[maps.reorder] for state in _as_state_list(states)
    ])
    weighted = vectors * np.sqrt(np.asarray(list(weights)))[:, None]
    return weighted.T @ weighted.conj()


def total_negativity(states: np.ndarray, weights: Iterable[float],
                     maps: SpinSplitMaps) -> float:
    """Eq. (24): N_updown = -sum(neg eigs) of the down partial transpose."""
    density = sector_density_product(states, weights, maps)
    transposed = (
        density.reshape(DIM_UP, DIM_UP, DIM_UP, DIM_UP)
        .transpose(0, 3, 2, 1)
        .reshape(DIM_SECTOR, DIM_SECTOR)
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (transposed + transposed.conj().T))
    return float(-eigenvalues[eigenvalues < -EIG_CUTOFF].sum()) + 0.0


def pure_total_negativity(state: np.ndarray, maps: SpinSplitMaps) -> float:
    """Eq. (25) pure-state: 0.5 * ((sum_n Gamma_n)^2 - 1)."""
    singular_values = np.linalg.svd(coeff_matrix(state, maps), compute_uv=False)
    return float(0.5 * (singular_values.sum() ** 2 - 1.0))


# --------------------------------------------------------------------------- #
# One- and two-body RDM blocks
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RDMGenerators:
    """State-independent contraction tensors from fermionic_mbody.rho_m_gen."""

    tensor_1: sp.coo_array   # (14, 14, 441, 441)
    tensor_2: sp.coo_array   # (91, 91, 441, 441)


def build_generators(sector: SectorBasis) -> RDMGenerators:
    tensor_1 = rho_m_gen(sector.basis, 1, n_workers=1)
    tensor_2 = rho_m_gen(sector.basis, 2, n_workers=1)
    return RDMGenerators(tensor_1=tensor_1, tensor_2=tensor_2)


@dataclass(frozen=True)
class PairBlockMaps:
    """
    Index/sign maps from the 91-dim pair basis to spin blocks.

    Derived from the FixedBasis(14, 2) bitmasks. updown_sign is +1 when 
    the up bit precedes the down bit in the wedge creation string, 
    -1 otherwise.
    """

    upup_indices: np.ndarray       # (21,) rows of the 91 basis with two even bits
    upup_pairs: np.ndarray         # (21, 2) spatial (p < q) per row
    dndn_indices: np.ndarray       # (21,)
    dndn_pairs: np.ndarray         # (21, 2)
    updown_indices: np.ndarray     # (49,)
    updown_prod_index: np.ndarray  # (49,) destination p_up * 7 + q_down
    updown_sign: np.ndarray        # (49,) +/- 1


def build_pair_block_maps() -> PairBlockMaps:
    pair_basis = FixedBasis(N_SPIN_ORBITALS, 2)
    upup_indices, upup_pairs = [], []
    dndn_indices, dndn_pairs = [], []
    updown_indices, updown_prod_index, updown_sign = [], [], []
    for row, mask in enumerate(pair_basis.num_ele):
        bits = [b for b in range(N_SPIN_ORBITALS) if (int(mask) >> b) & 1]
        first, second = bits  
        if first % 2 == 0 and second % 2 == 0:
            upup_indices.append(row)
            upup_pairs.append((first // 2, second // 2))
        elif first % 2 == 1 and second % 2 == 1:
            dndn_indices.append(row)
            dndn_pairs.append((first // 2, second // 2))
        else:
            even_bit = first if first % 2 == 0 else second
            odd_bit = second if first % 2 == 0 else first
            updown_indices.append(row)
            updown_prod_index.append((even_bit // 2) * N_SPATIAL + odd_bit // 2)
            updown_sign.append(1.0 if even_bit < odd_bit else -1.0)
    maps = PairBlockMaps(
        upup_indices=np.array(upup_indices),
        upup_pairs=np.array(upup_pairs),
        dndn_indices=np.array(dndn_indices),
        dndn_pairs=np.array(dndn_pairs),
        updown_indices=np.array(updown_indices),
        updown_prod_index=np.array(updown_prod_index),
        updown_sign=np.array(updown_sign),
    )
    if len(maps.upup_indices) != DIM_UPUP or len(maps.updown_indices) != DIM_UPDOWN:
        raise AssertionError("unexpected spin-block dimensions in the pair basis")
    return maps


def rho2_updown_product(rho2_wedge: np.ndarray, maps: PairBlockMaps) -> np.ndarray:
    """
    Opposite-spin block in the 49-dim product basis, trace N_up * N_down.

    out[p*7+q, r*7+s] = <a^dag_r b^dag_s b_q a_p> with a = up, b = down.
    """
    sub = rho2_wedge[np.ix_(maps.updown_indices, maps.updown_indices)]
    dressed = sub * np.outer(maps.updown_sign, maps.updown_sign)
    out = np.zeros((DIM_UPDOWN, DIM_UPDOWN), dtype=dressed.dtype)
    out[np.ix_(maps.updown_prod_index, maps.updown_prod_index)] = dressed
    return out


def reduced_blocks(states: np.ndarray, weights: Iterable[float],
                   generators: RDMGenerators,
                   maps: PairBlockMaps) -> dict[str, np.ndarray]:
    """
    Weight-mixed unnormalized RDM blocks of a spectral mixture.

    Returns rho1_up/rho1_down (7x7, trace 5), rho2_upup/rho2_dndn (21x21 wedge,
    trace 10) and rho2_updown (49x49 product, trace 25), plus the full rho1
    (14x14) and wedge rho2 (91x91).
    """
    weight_list = np.asarray(list(weights), dtype=float)
    state_list = _as_state_list(states)
    rho1_full = np.zeros((N_SPIN_ORBITALS, N_SPIN_ORBITALS))
    rho2_full = np.zeros((DIM_PAIR, DIM_PAIR))
    for weight, state in zip(weight_list, state_list):
        rho1_full += weight * np.asarray(rho_m(state, generators.tensor_1))
        rho2_full += weight * np.asarray(rho_m(state, generators.tensor_2))
    return {
        "rho1_full": rho1_full,
        "rho2_full": rho2_full,
        "rho1_up": rho1_full[0::2, 0::2],
        "rho1_down": rho1_full[1::2, 1::2],
        "rho2_upup": rho2_full[np.ix_(maps.upup_indices, maps.upup_indices)],
        "rho2_dndn": rho2_full[np.ix_(maps.dndn_indices, maps.dndn_indices)],
        "rho2_updown": rho2_updown_product(rho2_full, maps),
    }


# --------------------------------------------------------------------------- #
# Negativities of the two-body blocks
# --------------------------------------------------------------------------- #

def opposite_spin_negativity(rho2_updown_prod: np.ndarray) -> float:
    """
    Eq. (26): N2_updown = 0.5 [Tr|PT| - N_up N_down] = -sum(neg eigs of PT).
    """
    matrix = np.asarray(rho2_updown_prod)
    transposed = (
        matrix.reshape(N_SPATIAL, N_SPATIAL, N_SPATIAL, N_SPATIAL)
        .transpose(0, 3, 2, 1)
        .reshape(DIM_UPDOWN, DIM_UPDOWN)
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (transposed + transposed.conj().T))
    negative_sum = float(-eigenvalues[eigenvalues < -EIG_CUTOFF].sum())
    trace_norm_formula = 0.5 * (np.abs(eigenvalues).sum() - matrix.trace().real)
    if abs(negative_sum - trace_norm_formula) > 5e-7:
        raise AssertionError("opposite-spin negativity formulas disagree")
    return negative_sum + 0.0


def same_spin_extended(block: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Antisymmetrized ordered representation ext[i,j,k,l] = <c^dag_k c^dag_l c_j c_i>.
    """
    extended = np.zeros((N_SPATIAL,) * 4, dtype=np.asarray(block).dtype)
    col_p, col_q = pairs[:, 0], pairs[:, 1]
    for row, (first, second) in enumerate(pairs):
        values = np.asarray(block)[row]
        extended[first, second][col_p, col_q] = values
        extended[second, first][col_p, col_q] = -values
        extended[first, second][col_q, col_p] = -values
        extended[second, first][col_q, col_p] = values
    return extended


def same_spin_negativity(block: np.ndarray, pairs: np.ndarray) -> float:
    """
    Eqs. (28)-(31): fermionic negativity of a same-spin two-body block.
    """
    if np.max(np.abs(np.imag(block))) > 2e-11:
        raise AssertionError("same-spin witness requires a real representation")
    extended = np.real(same_spin_extended(block, pairs))
    tp4 = extended.transpose(0, 3, 2, 1) - extended.transpose(0, 3, 1, 2)
    scaled = 0.5 * tp4.reshape(DIM_UPDOWN, DIM_UPDOWN)
    asymmetry = np.linalg.norm(scaled - scaled.T)
    if asymmetry > 5e-7:
        raise AssertionError(f"same-spin PT not symmetric: {asymmetry:.3e}")
    eigenvalues = np.linalg.eigvalsh(0.5 * (scaled + scaled.T))
    negative_sum = float(-eigenvalues[eigenvalues < -EIG_CUTOFF].sum())
    trace_norm_formula = 0.5 * (np.abs(eigenvalues).sum() - np.trace(block).real)
    if abs(negative_sum - trace_norm_formula) > 5e-7:
        raise AssertionError("same-spin negativity formulas disagree")
    return negative_sum + 0.0


# --------------------------------------------------------------------------- #
# Spectra, entropies, traces
# --------------------------------------------------------------------------- #

def hermitian_spectrum(matrix: np.ndarray, *, psd_tolerance: float = 5e-9,
                       label: str = "") -> np.ndarray:
    """Descending eigenvalues of a Hermitian PSD matrix."""
    matrix = np.asarray(matrix)
    hermiticity = np.linalg.norm(matrix - matrix.conj().T)
    if hermiticity > 1e-9:
        raise AssertionError(f"{label or 'matrix'} not Hermitian: {hermiticity:.3e}")
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
    if eigenvalues.min() < -psd_tolerance:
        raise AssertionError(
            f"{label or 'matrix'} materially negative: {eigenvalues.min():.3e}")
    return np.clip(eigenvalues, 0.0, None)[::-1]


def spectral_entropy_bits(matrix_or_eigs: np.ndarray, *,
                          cutoff: float = EIG_CUTOFF) -> float:
    """-sum lambda log2 lambda over eigenvalues > cutoff."""
    array = np.asarray(matrix_or_eigs)
    eigenvalues = np.linalg.eigvalsh(0.5 * (array + array.conj().T)) \
        if array.ndim == 2 else array
    positive = eigenvalues[eigenvalues > cutoff]
    return float(-(positive * np.log2(positive)).sum()) + 0.0


def shannon_bits(probabilities: np.ndarray, *, cutoff: float = 1e-15) -> float:
    """Shannon entropy in bits of a normalized distribution (sum p = 1)."""
    probabilities = np.asarray(probabilities, dtype=float)
    if abs(probabilities.sum() - 1.0) > 1e-9:
        raise AssertionError(f"weights not normalized: sum = {probabilities.sum()}")
    positive = probabilities[probabilities > cutoff]
    return float(-(positive * np.log2(positive)).sum()) + 0.0


def weight_entropy_bits(weights: np.ndarray, *, cutoff: float = 1e-300) -> float:
    """-sum q log2 q of weights q_n = exp[-beta(E_n - E_0)] (Fig. 7)."""
    weights = np.asarray(weights, dtype=float)
    positive = weights[weights > cutoff]
    return float(-(positive * np.log2(positive)).sum()) + 0.0


def expected_traces() -> dict[str, float]:
    return {
        "rho_up": 1.0,
        "rho1_up": float(N_UP),
        "rho1_down": float(N_DOWN),
        "rho1_full": float(N_ELECTRONS),
        "rho2_upup": float(N_UP * (N_UP - 1) // 2),
        "rho2_dndn": float(N_DOWN * (N_DOWN - 1) // 2),
        "rho2_updown": float(N_UP * N_DOWN),
        "rho2_full": float(N_ELECTRONS * (N_ELECTRONS - 1) // 2),
    }
