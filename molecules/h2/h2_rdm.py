"""
Scalable compact particle-RDM contractions.

The many-electron basis stores only determinant masks in PySCF CI address order,
and an RDMContractionPlan caches the state-independent action of compact
annihilation strings.  In the conventions of fermionic_mbody:

    V[r, I] = <r| C_I |Psi>
    Gamma[I, J] = sum_r V[r, I] conj(V[r, J])
"""

from __future__ import annotations

from itertools import combinations, product
from math import comb
from typing import Sequence

import numpy as np


__all__ = [
    "FixedSpinBasis",
    "PySCFCIAdapter",
    "PySCFInterleavedAdapter",
    "RDMContractionPlan",
    "MemoryBudgetError",
    "pyscf_ci_to_interleaved",
    "interleaved_to_pyscf_ci",
    "rho_m_direct",
    "alpha_beta_wedge_to_product",
    "alpha_beta_product_to_wedge",
]


DEFAULT_MEMORY_BUDGET_BYTES = 512 * 1024**2


class MemoryBudgetError(MemoryError):
    """Raised before a plan-owned allocation would exceed its memory budget."""


def _parse_nelec(nelec: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(nelec, (int, np.integer)):
        n = int(nelec)
        if n < 0:
            raise ValueError("electron count must be non-negative")
        return (n + 1) // 2, n // 2
    if len(nelec) != 2:
        raise ValueError("nelec must be an integer or an (nalpha, nbeta) pair")
    nalpha, nbeta = (int(nelec[0]), int(nelec[1]))
    if nalpha < 0 or nbeta < 0:
        raise ValueError("electron counts must be non-negative")
    return nalpha, nbeta


def _spread_spatial_masks(strings: np.ndarray, norb: int, spin: int) -> np.ndarray:
    """Spread spatial bits into even (alpha) or odd (beta) spin-orbital bits."""
    out = np.zeros(len(strings), dtype=np.uint64)
    src = np.asarray(strings, dtype=np.uint64)
    for p in range(norb):
        out |= ((src >> np.uint64(p)) & np.uint64(1)) << np.uint64(2 * p + spin)
    return out


def _pyscf_phase(alpha_mask: int, beta_mask: int) -> int:
    inversions = 0
    a = int(alpha_mask)
    while a:
        low = a & -a
        p = low.bit_length() - 1
        inversions += (int(beta_mask) & ((1 << p) - 1)).bit_count()
        a ^= low
    return -1 if inversions & 1 else 1


class FixedSpinBasis:
    """
    Mask-only fixed-(Nalpha,Nbeta) determinant basis in PySCF address order
    """

    __slots__ = (
        "norb",
        "nalpha",
        "nbeta",
        "d",
        "num",
        "pairs",
        "size",
        "alpha_strings",
        "beta_strings",
        "bitmasks",
        "pyscf_phases",
    )

    def __init__(self, norb: int, nalpha: int, nbeta: int) -> None:
        norb = int(norb)
        nalpha = int(nalpha)
        nbeta = int(nbeta)
        if norb < 0 or not (0 <= nalpha <= norb and 0 <= nbeta <= norb):
            raise ValueError("require norb >= 0 and 0 <= nalpha,nbeta <= norb")
        # The pinned helper represents determinant masks with signed int64.
        if 2 * norb > 63:
            raise ValueError("at most 63 interleaved spin orbitals are supported")

        try:
            from pyscf.fci import cistring
        except ImportError as exc:  
            raise ImportError("FixedSpinBasis requires PySCF") from exc

        alpha_strings = np.asarray(
            cistring.make_strings(range(norb), nalpha), dtype=np.int64
        )
        beta_strings = np.asarray(
            cistring.make_strings(range(norb), nbeta), dtype=np.int64
        )
        alpha_so = _spread_spatial_masks(alpha_strings, norb, spin=0)
        beta_so = _spread_spatial_masks(beta_strings, norb, spin=1)

        # PySCF CI tensors have shape (n_alpha_strings, n_beta_strings).
        bitmasks = (
            np.repeat(alpha_so, len(beta_so)) | np.tile(beta_so, len(alpha_so))
        ).astype(np.int64, copy=False)
        phases = np.empty(len(bitmasks), dtype=np.int8)
        k = 0
        for alpha_mask in alpha_strings:
            for beta_mask in beta_strings:
                phases[k] = _pyscf_phase(int(alpha_mask), int(beta_mask))
                k += 1

        self.norb = norb
        self.nalpha = nalpha
        self.nbeta = nbeta
        self.d = 2 * norb
        self.num = nalpha + nbeta
        self.pairs = False
        self.size = len(bitmasks)
        self.alpha_strings = alpha_strings
        self.beta_strings = beta_strings
        self.bitmasks = bitmasks
        self.pyscf_phases = phases

        for array in (
            self.alpha_strings,
            self.beta_strings,
            self.bitmasks,
            self.pyscf_phases,
        ):
            array.setflags(write=False)

    @property
    def num_ele(self) -> np.ndarray:
        """Backward-compatible alias used by fermionic_mbody.FixedBasis"""
        return self.bitmasks

    @property
    def ci_shape(self) -> tuple[int, int]:
        return len(self.alpha_strings), len(self.beta_strings)


class PySCFInterleavedAdapter:
    """Centralized, reversible PySCF-CI/interleaved determinant conversion."""

    __slots__ = ("basis", "ci_shape", "phases")

    def __init__(
        self,
        norb: int | FixedSpinBasis,
        nalpha: int | Sequence[int] | None = None,
        nbeta: int | None = None,
    ) -> None:
        if isinstance(norb, FixedSpinBasis):
            if nalpha is not None or nbeta is not None:
                raise ValueError("electron counts must not be repeated with a basis")
            self.basis = norb
        elif nalpha is None:
            raise TypeError("nalpha/nelec is required when constructing from norb")
        elif nbeta is None:
            nalpha_i, nbeta_i = _parse_nelec(nalpha) 
            self.basis = FixedSpinBasis(norb, nalpha_i, nbeta_i)
        else:
            nalpha_i, nbeta_i = int(nalpha), int(nbeta)
            self.basis = FixedSpinBasis(norb, nalpha_i, nbeta_i)
        self.ci_shape = self.basis.ci_shape
        self.phases = self.basis.pyscf_phases

    def to_interleaved(self, ci: np.ndarray) -> np.ndarray:
        """Return coefficients aligned with self.basis.bitmasks"""
        ci_array = np.asarray(ci)
        if ci_array.shape != self.ci_shape:
            raise ValueError(
                f"CI tensor has shape {ci_array.shape}; expected {self.ci_shape}"
            )
        return ci_array.reshape(-1) * self.phases

    def to_pyscf(self, psi: np.ndarray) -> np.ndarray:
        """Invert to_interleaved and restore the PySCF CI tensor."""
        psi_array = np.asarray(psi)
        if psi_array.ndim != 1 or len(psi_array) != self.basis.size:
            raise ValueError(
                f"psi must be one-dimensional with length {self.basis.size}"
            )
        # The phase is real and self-inverse.
        return (psi_array * self.phases).reshape(self.ci_shape)

    def interleaved_to_spin_product_matrix(self, psi: np.ndarray) -> np.ndarray:
        """
        Map an interleaved determinant vector to the alpha x beta product.
        """

        psi_array = np.asarray(psi)
        if psi_array.ndim != 1 or len(psi_array) != self.basis.size:
            raise ValueError(
                f"psi must be one-dimensional with length {self.basis.size}"
            )
        return (psi_array * self.phases).reshape(self.ci_shape)

    def to_spin_product_matrix(self, ci: np.ndarray) -> np.ndarray:
        """PySCF CI -> interleaved determinant -> alpha x beta product map."""

        return self.interleaved_to_spin_product_matrix(self.to_interleaved(ci))

    to_mbody = to_interleaved
    from_mbody = to_pyscf


PySCFCIAdapter = PySCFInterleavedAdapter


def pyscf_ci_to_interleaved(
    ci: np.ndarray,
    norb: int,
    nelec: int | Sequence[int],
    *,
    adapter: PySCFInterleavedAdapter | None = None,
) -> np.ndarray:
    nalpha, nbeta = _parse_nelec(nelec)
    if adapter is None:
        adapter = PySCFInterleavedAdapter(norb, nalpha, nbeta)
    elif (
        adapter.basis.norb,
        adapter.basis.nalpha,
        adapter.basis.nbeta,
    ) != (int(norb), nalpha, nbeta):
        raise ValueError("the supplied adapter does not match norb/nelec")
    return adapter.to_interleaved(ci)


def interleaved_to_pyscf_ci(
    psi: np.ndarray,
    norb: int,
    nelec: int | Sequence[int],
    *,
    adapter: PySCFInterleavedAdapter | None = None,
) -> np.ndarray:
    """One-shot inverse of pyscf_ci_to_interleaved."""
    nalpha, nbeta = _parse_nelec(nelec)
    if adapter is None:
        adapter = PySCFInterleavedAdapter(norb, nalpha, nbeta)
    elif (
        adapter.basis.norb,
        adapter.basis.nalpha,
        adapter.basis.nbeta,
    ) != (int(norb), nalpha, nbeta):
        raise ValueError("the supplied adapter does not match norb/nelec")
    return adapter.to_pyscf(psi)


def _compact_masks(d: int, m: int) -> np.ndarray:
    """Masks in the pinned FixedBasis(d,m) excitation-rank ordering."""
    dimension = comb(d, m)
    masks = np.empty(dimension, dtype=np.uint64)
    reference = (1 << m) - 1
    k = 0
    occupied = range(m)
    virtual = range(m, d)
    for rank in range(m + 1):
        for holes, particles in product(
            combinations(occupied, rank), combinations(virtual, rank)
        ):
            mask = reference
            for orbital in holes:
                mask ^= 1 << orbital
            for orbital in particles:
                mask |= 1 << orbital
            masks[k] = mask
            k += 1
    if k != dimension:  # defensive assertion against ordering regressions
        raise RuntimeError("internal compact-basis enumeration error")
    return masks


_SECTOR_ALIASES = {
    "full": "full",
    "all": "full",
    "alpha": "alpha",
    "a": "alpha",
    "up": "alpha",
    "α": "alpha",
    "beta": "beta",
    "b": "beta",
    "down": "beta",
    "β": "beta",
    "aa": "aa",
    "alphaalpha": "aa",
    "alpha-alpha": "aa",
    "αα": "aa",
    "ab": "ab",
    "alphabeta": "ab",
    "alpha-beta": "ab",
    "αβ": "ab",
    "bb": "bb",
    "betabeta": "bb",
    "beta-beta": "bb",
    "ββ": "bb",
}


def _normalise_sector(sector: str, m: int) -> str:
    key = str(sector).lower().replace("_", "").replace(" ", "")
    try:
        normal = _SECTOR_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unknown RDM sector {sector!r}") from exc
    allowed = {0: {"full"}, 1: {"full", "alpha", "beta"}, 2: {"full", "aa", "ab", "bb"}}
    if m in allowed and normal not in allowed[m]:
        raise ValueError(f"sector {normal!r} is incompatible with m={m}")
    if m not in allowed and normal != "full":
        raise ValueError("targeted spin sectors are implemented only for m=1 and m=2")
    return normal


def _mask_spin_counts(mask: int) -> tuple[int, int]:
    alpha = 0
    beta = 0
    value = int(mask)
    while value:
        low = value & -value
        orbital = low.bit_length() - 1
        if orbital & 1:
            beta += 1
        else:
            alpha += 1
        value ^= low
    return alpha, beta


def _sector_accepts(mask: int, sector: str) -> bool:
    if sector == "full":
        return True
    nalpha, nbeta = _mask_spin_counts(mask)
    return {
        "alpha": (1, 0),
        "beta": (0, 1),
        "aa": (2, 0),
        "ab": (1, 1),
        "bb": (0, 2),
    }[sector] == (nalpha, nbeta)


def _index_dtype(maximum_index: int) -> np.dtype:
    if maximum_index <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum_index <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _occupied_orbitals(mask: int) -> list[int]:
    result: list[int] = []
    value = int(mask)
    while value:
        low = value & -value
        result.append(low.bit_length() - 1)
        value ^= low
    return result


def _annihilation_sign(det_mask: int, orbitals: Sequence[int]) -> int:
    """Sign from applying annihilators in increasing compact-string order."""
    sign = 1
    current = int(det_mask)
    for orbital in orbitals:
        if (current & ((1 << orbital) - 1)).bit_count() & 1:
            sign = -sign
        current ^= 1 << orbital
    return sign


class RDMContractionPlan:
    """
    Cached factorized contraction plan for a compact particle RDM.

    Parameters
    ----------
    basis
        Any mask basis exposing d, num, size and bitmasks
    m
        Particle rank.
    sector
        fulll, alpha/beta for m=1, or aa/ab/bb for m=2.
    memory_budget_bytes
        Maximum bytes owned by the plan during construction or contraction.
    """

    def __init__(
        self,
        basis: object,
        m: int,
        *,
        sector: str = "full",
        memory_budget_bytes: int | None = DEFAULT_MEMORY_BUDGET_BYTES,
    ) -> None:
        self.basis = basis
        self.d = int(getattr(basis, "d"))
        number = getattr(basis, "num", None)
        if number is None:
            raise ValueError("basis.num (the fixed particle number) must be set")
        self.N = int(number)
        self.m = int(m)
        if not (0 <= self.m <= self.d):
            raise ValueError("require 0 <= m <= basis.d")
        if self.d > 63:
            raise ValueError("at most 63 spin orbitals are supported")
        if bool(getattr(basis, "pairs", False)):
            raise ValueError("pair-compressed bases are not particle determinant bases")

        self.sector = _normalise_sector(sector, self.m)
        if memory_budget_bytes is not None:
            memory_budget_bytes = int(memory_budget_bytes)
            if memory_budget_bytes <= 0:
                raise ValueError("memory_budget_bytes must be positive or None")
        self.memory_budget_bytes = memory_budget_bytes

        self.full_dimension = comb(self.d, self.m)
        self._ensure_budget(
            8 * self.full_dimension + 64 * self.full_dimension,
            "compact-basis construction",
        )

        raw_masks = getattr(basis, "bitmasks", None)
        if raw_masks is None:
            raw_masks = getattr(basis, "num_ele", None)
        if raw_masks is None:
            raise TypeError("basis must expose determinant bitmasks")
        state_masks = np.asarray(raw_masks)
        expected_size = int(getattr(basis, "size", len(state_masks)))
        if state_masks.ndim != 1 or len(state_masks) != expected_size:
            raise ValueError("basis masks and basis.size are inconsistent")
        self._ensure_budget(
            64 * len(state_masks) + 72 * self.full_dimension,
            "determinant-basis validation",
        )
        if len({int(mask) for mask in state_masks}) != len(state_masks):
            raise ValueError("basis determinant masks must be unique")
        upper_mask = (1 << self.d) - 1
        for mask in state_masks:
            value = int(mask)
            if value < 0 or value & ~upper_mask or value.bit_count() != self.N:
                raise ValueError("basis contains an invalid fixed-N determinant mask")
        self.state_dimension = len(state_masks)

        full_masks = _compact_masks(self.d, self.m)
        selected = np.fromiter(
            (_sector_accepts(int(mask), self.sector) for mask in full_masks),
            count=len(full_masks),
            dtype=bool,
        )
        self.full_indices = np.flatnonzero(selected).astype(np.intp, copy=False)
        self.rdm_masks = full_masks[selected].copy()
        self.system_masks = self.rdm_masks
        self.output_dimension = len(self.rdm_masks)
        local_index = {int(mask): i for i, mask in enumerate(self.rdm_masks)}

        # Exact connection count, obtained without allocating a connection list.
        nnz = 0
        if self.m <= self.N:
            for det_mask_raw in state_masks:
                occupied = _occupied_orbitals(int(det_mask_raw))
                for orbitals in combinations(occupied, self.m):
                    removed = sum(1 << orbital for orbital in orbitals)
                    if removed in local_index:
                        nnz += 1
        self.connectivity_nnz = nnz

        ci_dtype = _index_dtype(max(self.state_dimension - 1, 0))
        rdm_dtype = _index_dtype(max(self.output_dimension - 1, 0))
        entry_bytes = 8 + ci_dtype.itemsize + rdm_dtype.itemsize + 1
        self.estimated_build_peak_bytes = int(
            (2 * entry_bytes + 32) * nnz
            + 80 * self.full_dimension
            + 16 * (nnz + 1)
        )
        self._ensure_budget(self.estimated_build_peak_bytes, "connectivity construction")

        residual_per_connection = np.empty(nnz, dtype=np.uint64)
        rdm_indices = np.empty(nnz, dtype=rdm_dtype)
        ci_indices = np.empty(nnz, dtype=ci_dtype)
        signs = np.empty(nnz, dtype=np.int8)

        cursor = 0
        if self.m <= self.N:
            for ci_index, det_mask_raw in enumerate(state_masks):
                det_mask = int(det_mask_raw)
                occupied = _occupied_orbitals(det_mask)
                for orbitals in combinations(occupied, self.m):
                    removed = sum(1 << orbital for orbital in orbitals)
                    local = local_index.get(removed)
                    if local is None:
                        continue
                    residual_per_connection[cursor] = det_mask ^ removed
                    rdm_indices[cursor] = local
                    ci_indices[cursor] = ci_index
                    signs[cursor] = _annihilation_sign(det_mask, orbitals)
                    cursor += 1
        if cursor != nnz:
            raise RuntimeError("internal connectivity count mismatch")

        if nnz:
            order = np.argsort(residual_per_connection, kind="quicksort")
            residual_per_connection = residual_per_connection[order]
            rdm_indices = rdm_indices[order]
            ci_indices = ci_indices[order]
            signs = signs[order]
            del order

            is_start = np.empty(nnz, dtype=bool)
            is_start[0] = True
            is_start[1:] = residual_per_connection[1:] != residual_per_connection[:-1]
            starts = np.flatnonzero(is_start).astype(np.int64, copy=False)
            residual_masks = residual_per_connection[starts].copy()
            residual_ptr = np.empty(len(starts) + 1, dtype=np.int64)
            residual_ptr[:-1] = starts
            residual_ptr[-1] = nnz
        else:
            residual_masks = np.empty(0, dtype=np.uint64)
            residual_ptr = np.array([0], dtype=np.int64)

        self.residual_masks = residual_masks
        self.residual_ptr = residual_ptr
        self.rdm_indices = rdm_indices
        self.ci_indices = ci_indices
        self.signs = signs

        self.indptr = self.residual_ptr
        self.indices_ii = self.rdm_indices
        self.data_c_idx = self.ci_indices
        self.data_sign = self.signs

        for array in (
            self.full_indices,
            self.rdm_masks,
            self.residual_masks,
            self.residual_ptr,
            self.rdm_indices,
            self.ci_indices,
            self.signs,
        ):
            array.setflags(write=False)

        self.cache_nbytes = int(
            sum(
                array.nbytes
                for array in (
                    self.full_indices,
                    self.rdm_masks,
                    self.residual_masks,
                    self.residual_ptr,
                    self.rdm_indices,
                    self.ci_indices,
                    self.signs,
                )
            )
        )
        group_sizes = np.diff(self.residual_ptr)
        self.max_residual_degree = int(group_sizes.max(initial=0))

    def _ensure_budget(self, required: int, operation: str) -> None:
        budget = self.memory_budget_bytes
        if budget is not None and int(required) > budget:
            raise MemoryBudgetError(
                f"{operation} requires an estimated {int(required):,} bytes, "
                f"exceeding the explicit {budget:,}-byte budget"
            )

    def _validate_state(self, state: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(state)
        if array.ndim != 1 or len(array) != self.state_dimension:
            raise ValueError(
                f"{name} must be one-dimensional with length {self.state_dimension}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must have a numeric dtype")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains a non-finite coefficient")
        return array

    def estimated_contract_peak_bytes(
        self, dtype: np.dtype | type = np.complex128, *, batch_size: int = 1
    ) -> int:
        itemsize = np.dtype(dtype).itemsize
        out = int(batch_size) * self.output_dimension**2 * itemsize
        degree = self.max_residual_degree
        temporary = (2 * degree + 3 * degree**2) * itemsize
        return int(self.cache_nbytes + out + temporary)

    def _accumulate(
        self,
        out: np.ndarray,
        phi: np.ndarray,
        psi: np.ndarray,
        *,
        scale: float = 1.0,
    ) -> None:
        for r in range(len(self.residual_masks)):
            start = int(self.residual_ptr[r])
            stop = int(self.residual_ptr[r + 1])
            local = self.rdm_indices[start:stop]
            ci = self.ci_indices[start:stop]
            sign = self.signs[start:stop]
            v_psi = psi[ci] * sign
            v_phi = phi[ci] * sign
            contribution = np.multiply.outer(v_psi, np.conj(v_phi))
            if scale != 1.0:
                contribution *= scale
            out[np.ix_(local, local)] += contribution

    def transition(self, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Return Gamma[phi,psi][I,J] = <phi|C_J^dag C_I|psi>"""
        phi_array = self._validate_state(phi, "phi")
        psi_array = self._validate_state(psi, "psi")
        dtype = np.result_type(phi_array.dtype, psi_array.dtype, np.float64)
        self._ensure_budget(
            self.estimated_contract_peak_bytes(dtype), "transition-RDM contraction"
        )
        out = np.zeros((self.output_dimension, self.output_dimension), dtype=dtype)
        self._accumulate(out, phi_array, psi_array)
        return out

    def contract(self, psi: np.ndarray) -> np.ndarray:
        """Contract one pure state without constructing |psi><psi|."""
        psi_array = self._validate_state(psi, "psi")
        dtype = np.result_type(psi_array.dtype, np.float64)
        self._ensure_budget(
            self.estimated_contract_peak_bytes(dtype), "pure-state RDM contraction"
        )
        out = np.zeros((self.output_dimension, self.output_dimension), dtype=dtype)
        self._accumulate(out, psi_array, psi_array)
        if not np.iscomplexobj(psi_array):
            return np.real(out)
        return out

    def batch(self, states: np.ndarray) -> np.ndarray:
        """Contract a (batch,state_dimension) array of pure states."""
        array = np.asarray(states)
        if array.ndim != 2 or array.shape[1] != self.state_dimension:
            raise ValueError(
                "states must have shape (batch, " f"{self.state_dimension})"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("states must have a numeric dtype")
        if not np.all(np.isfinite(array)):
            raise ValueError("states contain a non-finite coefficient")
        dtype = np.result_type(array.dtype, np.float64)
        self._ensure_budget(
            self.estimated_contract_peak_bytes(dtype, batch_size=len(array)),
            "batch RDM contraction",
        )
        out = np.zeros(
            (len(array), self.output_dimension, self.output_dimension), dtype=dtype
        )
        for index, state in enumerate(array):
            self._accumulate(out[index], state, state)
        if not np.iscomplexobj(array):
            return np.real(out)
        return out

    contract_batch = batch

    def contract_mixture(
        self,
        states: Sequence[np.ndarray] | np.ndarray,
        weights: Sequence[float] | np.ndarray,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Stream sum_a weights[a] * Gamma(states[a])"""
        weight_array_raw = np.asarray(weights)
        if np.iscomplexobj(weight_array_raw) and np.any(weight_array_raw.imag != 0):
            raise ValueError("mixture weights must be real")
        weight_array = np.asarray(weight_array_raw.real, dtype=np.float64)
        if weight_array.ndim != 1 or not np.all(np.isfinite(weight_array)):
            raise ValueError("weights must be a finite one-dimensional array")
        if np.any(weight_array < 0):
            raise ValueError("mixture weights must be non-negative")

        if isinstance(states, np.ndarray):
            if states.ndim != 2 or states.shape[1] != self.state_dimension:
                raise ValueError(
                    f"states must have shape (rank, {self.state_dimension})"
                )
            state_sequence = [states[i] for i in range(len(states))]
        else:
            state_sequence = list(states)
        if len(state_sequence) != len(weight_array) or not state_sequence:
            raise ValueError("states and weights must have the same non-zero length")
        total_weight = float(weight_array.sum())
        if total_weight <= 0:
            raise ValueError("mixture must have positive total weight")
        if normalize:
            weight_array = weight_array / total_weight

        checked = [self._validate_state(state, f"states[{i}]") for i, state in enumerate(state_sequence)]
        dtype = np.result_type(np.float64, *[state.dtype for state in checked])
        self._ensure_budget(
            self.estimated_contract_peak_bytes(dtype), "mixture RDM contraction"
        )
        out = np.zeros((self.output_dimension, self.output_dimension), dtype=dtype)
        for state, weight in zip(checked, weight_array):
            if weight:
                self._accumulate(out, state, state, scale=float(weight))
        if not any(np.iscomplexobj(state) for state in checked):
            return np.real(out)
        return out

    mixture = contract_mixture

    def extract(self, full_rdm: np.ndarray) -> np.ndarray:
        """Extract this plan's targeted block from a full compact RDM."""
        array = np.asarray(full_rdm)
        expected = (self.full_dimension, self.full_dimension)
        if array.shape != expected:
            raise ValueError(f"full_rdm has shape {array.shape}; expected {expected}")
        return array[np.ix_(self.full_indices, self.full_indices)]

    def embed(self, local_rdm: np.ndarray) -> np.ndarray:
        """Embed a targeted block into an otherwise-zero full compact matrix."""
        array = np.asarray(local_rdm)
        expected = (self.output_dimension, self.output_dimension)
        if array.shape != expected:
            raise ValueError(f"local_rdm has shape {array.shape}; expected {expected}")
        required = self.cache_nbytes + self.full_dimension**2 * array.dtype.itemsize
        self._ensure_budget(required, "targeted-block embedding")
        out = np.zeros((self.full_dimension, self.full_dimension), dtype=array.dtype)
        out[np.ix_(self.full_indices, self.full_indices)] = array
        return out


def rho_m_direct(
    basis_N: object,
    m: int,
    psi: np.ndarray,
    *,
    n_workers: int | None = None,
    _statistics: str = "fermionic",
    sector: str = "full",
    memory_budget_bytes: int | None = DEFAULT_MEMORY_BUDGET_BYTES,
) -> np.ndarray:
    """Backward-compatible one-shot wrapper around RDMContractionPlan"""
    if bool(getattr(basis_N, "pairs", False)) or _statistics != "fermionic":
        try:
            from fermionic_mbody import rho_m_direct as helper_rho_m_direct
        except ImportError as exc:  
            raise ValueError(
                "pair-compressed/bosonic compatibility requires fermionic_mbody"
            ) from exc
        return helper_rho_m_direct(
            basis_N,
            m,
            psi,
            n_workers=n_workers,
            _statistics=_statistics,
        )
    del n_workers
    return RDMContractionPlan(
        basis_N,
        m,
        sector=sector,
        memory_budget_bytes=memory_budget_bytes,
    ).contract(psi)


def _alpha_beta_wedge_map(
    wedge_masks: Sequence[int] | np.ndarray, norb: int
) -> tuple[np.ndarray, np.ndarray]:
    masks = np.asarray(wedge_masks)
    if masks.ndim != 1 or len(masks) != norb * norb:
        raise ValueError("a complete alpha-beta compact mask list is required")
    product_indices = np.empty(len(masks), dtype=np.intp)
    phases = np.empty(len(masks), dtype=np.int8)
    seen: set[int] = set()
    for local, mask_raw in enumerate(masks):
        occupied = _occupied_orbitals(int(mask_raw))
        alpha_modes = [mode for mode in occupied if mode % 2 == 0]
        beta_modes = [mode for mode in occupied if mode % 2 == 1]
        if len(alpha_modes) != 1 or len(beta_modes) != 1:
            raise ValueError("wedge_masks contains a non-alpha-beta pair")
        p = alpha_modes[0] // 2
        q = beta_modes[0] // 2
        if p >= norb or q >= norb:
            raise ValueError("wedge mask is outside the requested orbital space")
        target = p * norb + q
        if target in seen:
            raise ValueError("duplicate alpha-beta wedge mask")
        seen.add(target)
        product_indices[local] = target
        phases[local] = 1 if p <= q else -1
    return product_indices, phases


def alpha_beta_wedge_to_product(
    wedge_rdm: np.ndarray,
    wedge_masks: Sequence[int] | np.ndarray,
    norb: int,
) -> np.ndarray:
    """Signed compact-wedge -> alpha-p/beta-q product-basis transformation."""
    array = np.asarray(wedge_rdm)
    dimension = int(norb) ** 2
    if array.shape != (dimension, dimension):
        raise ValueError("wedge_rdm has the wrong alpha-beta block shape")
    indices, phases = _alpha_beta_wedge_map(wedge_masks, int(norb))
    out = np.empty_like(array)
    out[np.ix_(indices, indices)] = array * np.multiply.outer(phases, phases)
    return out


def alpha_beta_product_to_wedge(
    product_rdm: np.ndarray,
    wedge_masks: Sequence[int] | np.ndarray,
    norb: int,
) -> np.ndarray:
    """Inverse signed permutation of alpha_beta_wedge_to_product"""
    array = np.asarray(product_rdm)
    dimension = int(norb) ** 2
    if array.shape != (dimension, dimension):
        raise ValueError("product_rdm has the wrong alpha-beta block shape")
    indices, phases = _alpha_beta_wedge_map(wedge_masks, int(norb))
    return array[np.ix_(indices, indices)] * np.multiply.outer(phases, phases)
