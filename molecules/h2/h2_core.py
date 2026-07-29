"""
Core for the H2/cc-pVDZ entanglement notebook

* RHF + complete fixed-sector FCI of H2 (PySCF): the full M_S=0 Hamiltonian
  (100x100 in cc-pVDZ) is assembled by Hamiltonian action and diagonalized
  exactly
* compact one- and two-particle RDMs contracted with the h2_rdm engine
  through targeted spin-sector plans, never forming a many-body density
  matrix as a production object
* spin-resolved entanglement observables for the ground state and for the
  thermal mixture of the fixed (1,1), M_S=0 sector.

Conventions
-----------
Particle RDMs are unnormalized: Tr rho1_sigma = 1, Tr rho2_ab = 1 and the
same-spin two-body blocks vanish structurally (Tr rho2_aa = Tr rho2_bb = 0).
Entropies are in bits (log base 2). Spin orbitals are interleaved
(alpha -> 2p, beta -> 2p+1). For (N_alpha, N_beta) = (1, 1) the alpha-beta 
two-body RDM in the ordered product basis is the complete fixed-sector
density matrix
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.special import logsumexp
from pyscf import ao2mo, fci, gto, scf
from pyscf.fci import cistring

from h2_rdm import (
    FixedSpinBasis,
    PySCFInterleavedAdapter,
    RDMContractionPlan,
    alpha_beta_wedge_to_product,
)

PRIMARY_BASIS = "cc-pvdz"
DEFAULT_BETA_HARTREE_INVERSE = 1000.0
SCAN_GRID_ANGSTROM = tuple(np.round(np.linspace(0.4, 6.0, 57), 10))

def compute_grid() -> tuple[float, ...]:
    """Sorted union of the plotting grid and every exact anchor geometry"""
    values = {round(float(r), 10) for r in SCAN_GRID_ANGSTROM}
    return tuple(sorted(values))


# ---------------------------------------------------------------------------
# Electronic structure: complete fixed-sector diagonalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverConfig:
    """Electronic-structure tolerances and validation gates"""
    rhf_energy_tolerance: float = 1.0e-12
    rhf_gradient_tolerance: float = 1.0e-8
    rhf_max_cycle: int = 200
    hermiticity_tolerance: float = 1.0e-12
    dense_residual_validation: float = 1.0e-10
    orthonormality_tolerance: float = 5.0e-8
    spin_rounding_tolerance: float = 1.0e-2
    spin_defect_tolerance: float = 1.0e-8
    spin_cluster_tolerance_hartree: float = 1.0e-9
    oracle_tolerance: float = 2.0e-8
    triplet_oracle_tolerance: float = 1.0e-7
    identity_tolerance: float = 1.0e-9
    trace_tolerance: float = 2.0e-8


DEFAULT_CONFIG = SolverConfig()


@dataclass
class RootState:
    """One validated M_S=0 eigenstate of the complete sector spectrum"""
    index: int
    energy_total_hartree: float
    ci: np.ndarray = field(repr=False)
    s2: float = 0.0
    spin: int = 0
    residual_norm: float = 0.0


@dataclass
class GeometryRecord:
    """Electronic structure of one exact H2 geometry (complete spectrum)"""
    r_angstrom: float
    basis_name: str
    norb: int
    nelec: tuple[int, int]
    nuclear_repulsion_hartree: float
    rhf_total_hartree: float
    fci_singlet_total_hartree: float
    primary_ci: np.ndarray = field(repr=False)
    roots: list[RootState] = field(repr=False)
    n_singlets: int = 0
    n_triplets: int = 0
    max_residual_norm: float = 0.0
    max_spin_defect: float = 0.0
    orthonormality_error: float = 0.0
    h1_mo: np.ndarray = field(default=None, repr=False)
    eri_mo: np.ndarray = field(default=None, repr=False)

    @property
    def ci_shape(self) -> tuple[int, int]:
        return self.primary_ci.shape

    @property
    def determinant_count(self) -> int:
        return int(self.ci_shape[0] * self.ci_shape[1])

    @property
    def singlet_triplet_gap_hartree(self) -> float:
        singlets = [r.energy_total_hartree for r in self.roots if r.spin == 0]
        triplets = [r.energy_total_hartree for r in self.roots if r.spin == 1]
        if not singlets or not triplets:
            return math.nan
        return float(min(triplets) - min(singlets))

    @property
    def next_gap_hartree(self) -> float:
        energies = sorted(r.energy_total_hartree for r in self.roots)
        if len(energies) < 3:
            return math.nan
        return float(energies[2] - energies[0])


def _canonicalize_phase(vector: np.ndarray) -> np.ndarray:
    out = np.asarray(vector).copy()
    flat = out.ravel()
    pivot = int(np.argmax(np.abs(flat)))
    phase = flat[pivot] / abs(flat[pivot])
    out /= phase
    if np.max(np.abs(np.imag(out))) < 5.0e-14:
        out = np.asarray(np.real(out), dtype=np.float64)
    return out


def _spin_from_s2(s2: float, tolerance: float) -> int:
    raw = 0.5 * (-1.0 + math.sqrt(max(0.0, 1.0 + 4.0 * float(s2))))
    spin = int(round(raw))
    if abs(raw - spin) > tolerance:
        raise RuntimeError(f"eigenstate is not spin-pure: <S^2>={s2:.6g}")
    return spin


def _energy_clusters(energies: Sequence[float], tolerance: float) -> list[list[int]]:
    clusters: list[list[int]] = [[0]]
    for index in range(1, len(energies)):
        if abs(float(energies[index] - energies[index - 1])) <= tolerance:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    return clusters


def sector_hamiltonian(
    h1_mo: np.ndarray,
    eri_mo: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Dense electronic Hamiltonian of one fixed-(nalpha,nbeta) sector.

    Assembled column by column from PySCF's direct Hamiltonian action
    """
    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    dim = na * nb
    h2e = fci.direct_spin1.absorb_h1e(h1_mo, eri_mo, norb, nelec, 0.5)
    ham = np.empty((dim, dim))
    unit = np.zeros(dim)
    for column in range(dim):
        unit[column] = 1.0
        ham[:, column] = fci.direct_spin1.contract_2e(
            h2e, unit.reshape(na, nb), norb, nelec
        ).ravel()
        unit[column] = 0.0
    return ham, (na, nb)


def _purify_spin_clusters(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    ci_shape: tuple[int, int],
    ham: np.ndarray,
    cluster_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simultaneously diagonalize H and S^2 inside degenerate clusters.

    eigh returns arbitrary rotations inside degenerate energy
    clusters, which mix singlet and triplet components; since [H, S^2] = 0
    the in-cluster S^2 matrix can be diagonalized exactly.  Diagonalizing
    S^2 alone is not sufficient: a cluster may contain two states of the
    same spin whose real sub-tolerance splitting an arbitrary S^2-noise
    rotation would erase.  Each equal-spin subspace is therefore
    re-diagonalized in the projected Hamiltonian, and the full root list is
    re-sorted by the revalidated energies before it is returned.
    """
    values = eigenvalues.copy()
    vectors = eigenvectors.copy()
    for cluster in _energy_clusters(values, cluster_tolerance):
        if len(cluster) < 2:
            continue
        block = vectors[:, cluster]
        s2_action = np.column_stack(
            [
                fci.spin_op.contract_ss(
                    block[:, k].reshape(ci_shape), norb, nelec
                ).ravel()
                for k in range(len(cluster))
            ]
        )
        s2_matrix = block.T @ s2_action
        s2_values, rotation = np.linalg.eigh(0.5 * (s2_matrix + s2_matrix.T))
        rotated = block @ rotation
        # Group the S^2 eigenvectors by integer spin and re-diagonalize the
        # projected Hamiltonian inside every equal-spin subspace.
        spins = np.array([
            round(0.5 * (-1.0 + math.sqrt(max(0.0, 1.0 + 4.0 * float(s2)))))
            for s2 in s2_values
        ])
        cluster_values = np.empty(len(cluster))
        for spin in np.unique(spins):
            members = np.flatnonzero(spins == spin)
            sub = rotated[:, members]
            h_sub = sub.T @ (ham @ sub)
            sub_values, sub_rotation = np.linalg.eigh(0.5 * (h_sub + h_sub.T))
            rotated[:, members] = sub @ sub_rotation
            cluster_values[members] = sub_values
        vectors[:, cluster] = rotated
        values[cluster] = cluster_values
    order = np.argsort(values, kind="stable")
    return values[order], vectors[:, order]


def solve_h2_geometry(
    r_angstrom: float,
    *,
    basis_name: str = PRIMARY_BASIS,
    config: SolverConfig = DEFAULT_CONFIG,
) -> GeometryRecord:
    """
    Solve neutral H2 at one exact geometry: complete M_S=0 eigensystem.

    Validation is fail-closed: RHF convergence (with a fallback),
    Hamiltonian hermiticity, per-root residuals, orthonormality, spin purity
    after in-cluster S^2 rotation, an independent spin-adapted singlet oracle
    (energy and state overlap), and an independent dense (2,0)-sector solve
    certifying the lowest triplet.
    """
    r_angstrom = float(r_angstrom)
    mol = gto.M(
        atom=(("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r_angstrom))),
        unit="Angstrom",
        basis=basis_name,
        charge=0,
        spin=0,
        cart=False,
        symmetry=False,
        verbose=0,
    )

    mf = scf.RHF(mol)
    mf.conv_tol = config.rhf_energy_tolerance
    mf.conv_tol_grad = config.rhf_gradient_tolerance
    mf.max_cycle = config.rhf_max_cycle
    rhf_total = float(mf.kernel())
    if not mf.converged:
        mf = mf.newton()
        mf.conv_tol = config.rhf_energy_tolerance
        mf.conv_tol_grad = config.rhf_gradient_tolerance
        rhf_total = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge at R={r_angstrom:.4f} A")

    norb = int(mol.nao_nr())
    mo_coeff = np.asarray(mf.mo_coeff)
    h1_mo = np.asarray(mo_coeff.T @ mf.get_hcore() @ mo_coeff, order="C")
    eri_mo = np.asarray(ao2mo.kernel(mol, mo_coeff, aosym="s4"), order="C")
    enuc = float(mol.energy_nuc())
    nelec = tuple(map(int, mol.nelec))

    ham, ci_shape = sector_hamiltonian(h1_mo, eri_mo, norb, nelec)
    hermiticity = float(np.max(np.abs(ham - ham.T)))
    if hermiticity > config.hermiticity_tolerance:
        raise RuntimeError(
            f"Sector Hamiltonian is not Hermitian at R={r_angstrom:.4f} A: "
            f"{hermiticity:.3e}"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (ham + ham.T))
    eigenvalues, eigenvectors = _purify_spin_clusters(
        eigenvalues, eigenvectors, norb, nelec, ci_shape, ham,
        config.spin_cluster_tolerance_hartree,
    )

    gram = eigenvectors.T @ eigenvectors
    orthonormality_error = float(np.linalg.norm(gram - np.eye(gram.shape[0])))
    if orthonormality_error > config.orthonormality_tolerance:
        raise RuntimeError(f"Eigenvector set not orthonormal at R={r_angstrom:.4f} A")

    roots: list[RootState] = []
    max_residual = 0.0
    max_spin_defect = 0.0
    for index in range(len(eigenvalues)):
        vector = eigenvectors[:, index]
        residual = float(np.linalg.norm(ham @ vector - eigenvalues[index] * vector))
        max_residual = max(max_residual, residual)
        if residual > config.dense_residual_validation:
            raise RuntimeError(
                f"Root {index} residual {residual:.3e} at R={r_angstrom:.4f} A"
            )
        ci = _canonicalize_phase(vector.reshape(ci_shape))
        s2, _ = fci.spin_op.spin_square0(ci, norb, nelec)
        spin = _spin_from_s2(float(s2), config.spin_rounding_tolerance)
        if spin not in (0, 1):
            raise RuntimeError(f"Impossible two-electron spin {spin} at root {index}")
        spin_defect = abs(float(s2) - spin * (spin + 1))
        max_spin_defect = max(max_spin_defect, spin_defect)
        if spin_defect > config.spin_defect_tolerance:
            raise RuntimeError(
                f"Root {index} spin defect {spin_defect:.3e} at R={r_angstrom:.4f} A"
            )
        roots.append(
            RootState(index, float(eigenvalues[index] + enuc), ci, float(s2), spin, residual)
        )

    singlets = [r for r in roots if r.spin == 0]
    triplets = [r for r in roots if r.spin == 1]
    # Exact multiplicity counts of the two-electron M_S=0 sector: K(K+1)/2
    # singlets and K(K-1)/2 triplets for K spatial orbitals (55/45 in cc-pVDZ).
    expected_singlets = norb * (norb + 1) // 2
    expected_triplets = norb * (norb - 1) // 2
    if len(singlets) != expected_singlets or len(triplets) != expected_triplets:
        raise RuntimeError(
            f"Multiplicity count {len(singlets)}/{len(triplets)} differs from the "
            f"exact {expected_singlets}/{expected_triplets} at R={r_angstrom:.4f} A"
        )
    lowest_singlet = min(singlets, key=lambda r: r.energy_total_hartree)

    # Independent spin-adapted singlet oracle: energy and state agreement.
    spin0 = fci.direct_spin0.FCI(mol)
    spin0.conv_tol = 1.0e-12
    spin0.max_cycle = 500
    e0_ref, ci0 = spin0.kernel(h1_mo, eri_mo, norb, nelec, ecore=enuc, nroots=1)
    if not bool(spin0.converged):
        raise RuntimeError(f"Spin-adapted oracle failed at R={r_angstrom:.4f} A")
    if abs(lowest_singlet.energy_total_hartree - float(e0_ref)) > config.oracle_tolerance:
        raise RuntimeError("Dense and spin-adapted singlet energies disagree")
    overlap = abs(float(np.vdot(np.asarray(ci0).ravel(), lowest_singlet.ci.ravel())))
    if overlap < 1.0 - 1.0e-8:
        raise RuntimeError(
            f"Dense and spin-adapted singlet states disagree: |<a|b>|={overlap!r}"
        )

    # Independent dense (2,0)-sector solve: its ground state is the lowest
    # triplet, certifying the M_S=0 triplet identification.
    ham_t, _ = sector_hamiltonian(h1_mo, eri_mo, norb, (2, 0))
    e_t = float(np.linalg.eigvalsh(0.5 * (ham_t + ham_t.T))[0] + enuc)
    lowest_triplet = min(t.energy_total_hartree for t in triplets)
    if abs(lowest_triplet - e_t) > config.triplet_oracle_tolerance:
        raise RuntimeError(
            f"(2,0) oracle disagrees with the M_S=0 triplet at R={r_angstrom:.4f} A: "
            f"{lowest_triplet!r} vs {e_t!r}"
        )

    return GeometryRecord(
        r_angstrom=r_angstrom,
        basis_name=basis_name,
        norb=norb,
        nelec=nelec,
        nuclear_repulsion_hartree=enuc,
        rhf_total_hartree=rhf_total,
        fci_singlet_total_hartree=lowest_singlet.energy_total_hartree,
        primary_ci=lowest_singlet.ci,
        roots=roots,
        n_singlets=len(singlets),
        n_triplets=len(triplets),
        max_residual_norm=max_residual,
        max_spin_defect=max_spin_defect,
        orthonormality_error=orthonormality_error,
        h1_mo=h1_mo,
        eri_mo=eri_mo,
    )


# ---------------------------------------------------------------------------
# Compact RDM backend 
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class H2RDMBackend:
    """Targeted contraction plans for the fixed-spin sector"""
    basis: FixedSpinBasis
    adapter: PySCFInterleavedAdapter
    plan_1a: RDMContractionPlan
    plan_1b: RDMContractionPlan
    plan_2ab: RDMContractionPlan
    plan_2aa: RDMContractionPlan
    plan_2bb: RDMContractionPlan


def build_rdm_backend(norb: int, nalpha: int, nbeta: int) -> H2RDMBackend:
    basis = FixedSpinBasis(norb, nalpha, nbeta)
    adapter = PySCFInterleavedAdapter(basis)
    return H2RDMBackend(
        basis=basis,
        adapter=adapter,
        plan_1a=RDMContractionPlan(basis, 1, sector="alpha"),
        plan_1b=RDMContractionPlan(basis, 1, sector="beta"),
        plan_2ab=RDMContractionPlan(basis, 2, sector="ab"),
        plan_2aa=RDMContractionPlan(basis, 2, sector="aa"),
        plan_2bb=RDMContractionPlan(basis, 2, sector="bb"),
    )


def expected_traces(nalpha: int, nbeta: int) -> dict[str, float]:
    """Unnormalized-RDM block traces derived from the electron counts"""
    return {
        "rho1_alpha": float(nalpha),
        "rho1_beta": float(nbeta),
        "rho2_aa": float(math.comb(nalpha, 2)),
        "rho2_ab": float(nalpha * nbeta),
        "rho2_bb": float(math.comb(nbeta, 2)),
    }


# ---------------------------------------------------------------------------
# Entanglement measures
# ---------------------------------------------------------------------------

def hermitian_spectrum(matrix: np.ndarray, *, psd_tolerance: float = 2.0e-9) -> np.ndarray:
    """Eigenvalues of a Hermitian matrix"""
    herm_error = float(np.linalg.norm(matrix - matrix.conj().T))
    if herm_error > 5.0e-9:
        raise RuntimeError(f"Matrix is not Hermitian: {herm_error:.3e}")
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
    if float(eigenvalues[0]) < -psd_tolerance:
        raise RuntimeError(f"RDM is not PSD: lambda_min={eigenvalues[0]:.3e}")
    eigenvalues[eigenvalues < 0.0] = 0.0
    return eigenvalues


def entropy_unnormalized(matrix_or_eigenvalues: np.ndarray, *, cutoff: float = 1.0e-13) -> float:
    """Entropy -Tr X log2 X without trace normalization."""
    values = np.asarray(matrix_or_eigenvalues)
    if values.ndim == 2:
        values = hermitian_spectrum(values)
    values = np.real(values)
    values = values[values > cutoff]
    return float(-np.sum(values * np.log2(values)))


def entropy_probabilities(probabilities: np.ndarray, *, cutoff: float = 1.0e-15) -> float:
    """Shannon entropy in bits of a normalized probability vector."""
    probabilities = np.asarray(probabilities, dtype=float)
    if abs(float(probabilities.sum()) - 1.0) > 2.0e-9:
        raise ValueError("Probability vector is not normalized")
    probabilities = probabilities[probabilities > cutoff]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def partial_transpose_product(matrix: np.ndarray, dim_a: int, dim_b: int) -> np.ndarray:
    """Partial transpose on the second factor of an A x B product basis."""
    return (
        matrix.reshape(dim_a, dim_b, dim_a, dim_b)
        .transpose(0, 3, 2, 1)
        .reshape(dim_a * dim_b, dim_a * dim_b)
    )


def opposite_spin_negativity(product_rdm: np.ndarray, norb: int, n_pairs: float) -> float:
    """
    Two-body up-down negativity of the product-basis block

    Evaluated as ``-sum lambda_i^(-)`` of the partial transpose
    """
    pt = partial_transpose_product(product_rdm, norb, norb)
    eigenvalues = np.linalg.eigvalsh(0.5 * (pt + pt.conj().T))
    negativity = float(-np.sum(eigenvalues[eigenvalues < -1.0e-12]))
    trace_norm_formula = 0.5 * (float(np.sum(np.abs(eigenvalues))) - float(n_pairs))
    return 0.5 * negativity


def same_spin_negativity(block: np.ndarray, norb: int) -> float:
    """
    Same-spin two-body fermionic negativity

    Structurally zero for one electron per spin. Evaluated anyway so the
    vacuous sector is demonstrated by the notebook.
    """
    if np.max(np.abs(np.imag(block))) > 2.0e-11:
        raise ValueError("The same-spin witness requires a real representation")
    from itertools import combinations

    spatial_pairs = list(combinations(range(norb), 2))
    extended = np.zeros((norb, norb, norb, norb))
    for a, (i, j) in enumerate(spatial_pairs):
        for b, (k, l) in enumerate(spatial_pairs):
            value = float(np.real(block[a, b]))
            extended[i, j, k, l] = value
            extended[j, i, k, l] = -value
            extended[i, j, l, k] = -value
            extended[j, i, l, k] = value
    tp4 = extended.transpose(0, 3, 2, 1) - extended.transpose(0, 3, 1, 2)
    scaled = 0.5 * tp4.reshape(norb * norb, norb * norb)
    eigenvalues = np.linalg.eigvalsh(0.5 * (scaled + scaled.T))
    return 0.5 * float(-np.sum(eigenvalues[eigenvalues < -1.0e-12]))


# ---------------------------------------------------------------------------
# Ground-state and thermal analysis
# ---------------------------------------------------------------------------

def _contract_blocks(backend: H2RDMBackend, psi: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "rho1_alpha": backend.plan_1a.contract(psi),
        "rho1_beta": backend.plan_1b.contract(psi),
        "rho2_ab": backend.plan_2ab.contract(psi),
        "rho2_aa": backend.plan_2aa.contract(psi),
        "rho2_bb": backend.plan_2bb.contract(psi),
    }


def _contract_mixture_blocks(
    backend: H2RDMBackend, states: Sequence[np.ndarray], weights: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "rho1_alpha": backend.plan_1a.contract_mixture(states, weights),
        "rho1_beta": backend.plan_1b.contract_mixture(states, weights),
        "rho2_ab": backend.plan_2ab.contract_mixture(states, weights),
        "rho2_aa": backend.plan_2aa.contract_mixture(states, weights),
        "rho2_bb": backend.plan_2bb.contract_mixture(states, weights),
    }


def _block_observables(
    blocks: dict[str, np.ndarray],
    backend: H2RDMBackend,
    nelec: tuple[int, int],
    config: SolverConfig,
) -> tuple[dict[str, float], np.ndarray]:
    nalpha, nbeta = nelec
    norb = backend.basis.norb
    product = alpha_beta_wedge_to_product(
        blocks["rho2_ab"], backend.plan_2ab.system_masks, norb
    )
    traces = {
        "rho1_alpha": float(np.trace(blocks["rho1_alpha"]).real),
        "rho1_beta": float(np.trace(blocks["rho1_beta"]).real),
        "rho2_aa": float(np.trace(blocks["rho2_aa"]).real),
        "rho2_ab": float(np.trace(product).real),
        "rho2_bb": float(np.trace(blocks["rho2_bb"]).real),
    }
    expected = expected_traces(nalpha, nbeta)
    trace_error = max(abs(traces[key] - expected[key]) for key in expected)
    if trace_error > config.trace_tolerance:
        raise RuntimeError(f"Particle-RDM trace validation failed: {trace_error:.3e}")
    if _same_spin_max_abs(blocks) > 0.0:
        raise RuntimeError("Same-spin blocks are not structurally zero")

    s1a = entropy_unnormalized(blocks["rho1_alpha"])
    s1b = entropy_unnormalized(blocks["rho1_beta"])
    s2ab = entropy_unnormalized(product)
    observables = {
        "s_rho1_alpha": s1a,
        "s_rho1_beta": s1b,
        "s_rho2_ab": s2ab,
        "i2_ab": float(nbeta * s1a + nalpha * s1b - s2ab),
        "n2_ab": opposite_spin_negativity(product, norb, float(nalpha * nbeta)),
        "n2_aa": same_spin_negativity(blocks["rho2_aa"], norb),
        "n2_bb": same_spin_negativity(blocks["rho2_bb"], norb),
        "rho2_aa_frobenius": float(np.linalg.norm(blocks["rho2_aa"])),
        "max_trace_error": float(trace_error),
    }
    return observables, product


def _same_spin_max_abs(blocks: dict[str, np.ndarray]) -> float:
    return float(
        max(np.max(np.abs(blocks["rho2_aa"])), np.max(np.abs(blocks["rho2_bb"])))
    )


def analyze_ground_state(
    record: GeometryRecord,
    backend: H2RDMBackend,
    *,
    config: SolverConfig = DEFAULT_CONFIG,
) -> dict[str, float]:
    """
    Ground-state observables computation
    """

    # Reduced two-body route.
    psi = backend.adapter.to_mbody(record.primary_ci)
    blocks = _contract_blocks(backend, psi)
    reduced, _ = _block_observables(
        blocks, backend, record.nelec, config
    )

    # Global spin-partition route.
    c_matrix = backend.adapter.to_spin_product_matrix(record.primary_ci)
    schmidt = np.linalg.svd(c_matrix, compute_uv=False)

    probabilities = schmidt**2
    probabilities /= probabilities.sum()

    spin_entropy = entropy_probabilities(probabilities)
    trace_norm_excess = 0.5 * (float(np.sum(schmidt)) ** 2 - 1.0)
    spin_negativity = float(0.5 * trace_norm_excess)

    return {
        "e_fci": record.fci_singlet_total_hartree,
        "gap_st": record.singlet_triplet_gap_hartree,
        "e_spin": spin_entropy,
        "n_spin": spin_negativity,
        "i_ab": 2.0 * spin_entropy,
        "i2_ab": reduced["i2_ab"],
        "n2_ab": reduced["n2_ab"],
    }


def thermal_weights(roots: Sequence[RootState], beta_hartree_inverse: float) -> np.ndarray:
    """
    Raw normalized Boltzmann weights over the complete spectrum

    The eigenvalues come from a dense solve.
    """
    energies = np.array([r.energy_total_hartree for r in roots], dtype=float)
    log_weights = -beta_hartree_inverse * (energies - energies.min())
    return np.exp(log_weights - logsumexp(log_weights))


def analyze_thermal_state(
    record: GeometryRecord,
    backend: H2RDMBackend,
    *,
    beta_hartree_inverse: float = DEFAULT_BETA_HARTREE_INVERSE,
    config: SolverConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """
    Observables of the Gibbs mixture over the complete M_S=0 sector
    """
    weights = thermal_weights(record.roots, beta_hartree_inverse)
    states = [backend.adapter.to_mbody(root.ci) for root in record.roots]
    blocks = _contract_mixture_blocks(backend, states, weights)
    observables, product = _block_observables(blocks, backend, record.nelec, config)

    rho_ci = np.zeros_like(product)
    for weight, root in zip(weights, record.roots):
        c_matrix = backend.adapter.to_spin_product_matrix(root.ci)
        rho_ci += weight * np.outer(c_matrix.ravel(), np.conj(c_matrix.ravel())).real
    ci_engine_dev = float(np.max(np.abs(product - rho_ci)))

    spectrum = hermitian_spectrum(product)
    weight_spectrum_dev = float(
        np.max(np.abs(np.sort(spectrum)[::-1] - np.sort(weights)[::-1]))
    )
    s_gamma = entropy_unnormalized(spectrum)
    s_mixture = entropy_probabilities(weights)

    marg_a = np.einsum("pqrq->pr", product.reshape((record.norb,) * 4))
    marg_b = np.einsum("pqps->qs", product.reshape((record.norb,) * 4))
    marginal_dev = float(
        max(
            np.max(np.abs(marg_a - blocks["rho1_alpha"])),
            np.max(np.abs(marg_b - blocks["rho1_beta"])),
        )
    )
    n_from_ci = opposite_spin_negativity(
        rho_ci, record.norb, float(record.nelec[0] * record.nelec[1])
    )

    identity_devs = {
        "ci_vs_engine_density": ci_engine_dev,
        "gibbs_spectrum": weight_spectrum_dev,
        "entropy_vs_shannon": abs(s_gamma - s_mixture),
        "marginals": marginal_dev,
        "n2_vs_ci_global": observables["n2_ab"] - n_from_ci,
    }
    worst = max(abs(value) for value in identity_devs.values())
    if worst > config.identity_tolerance:
        raise RuntimeError(
            f"Thermal identity gate failed at R={record.r_angstrom:.4f} A: "
            f"{identity_devs}"
        )

    spins = np.array([r.spin for r in record.roots])
    observables.update(
        {
            "r_angstrom": record.r_angstrom,
            "beta_hartree_inverse": beta_hartree_inverse,
            "weights": weights,
            "root_energies": np.array([r.energy_total_hartree for r in record.roots]),
            "root_spins": spins,
            "singlet_weight": float(weights[spins == 0].sum()),
            "triplet_weight": float(weights[spins == 1].sum()),
            "s_mixture": s_mixture,
            # For (1,1) the reduced two-body MI *is* the global spin MI; both
            # sides are computed and gated above.
            "i_ab": float(
                observables["s_rho1_alpha"] + observables["s_rho1_beta"] - s_gamma
            ),
            "n_global": n_from_ci,
            "tail_fraction_bound": 0.0,
            "identity_devs": identity_devs,
        }
    )
    return observables


# ---------------------------------------------------------------------------
# Scan driver
# ---------------------------------------------------------------------------

def run_scan(
    grid: Sequence[float] = None,
    *,
    basis_name: str = PRIMARY_BASIS,
    beta_hartree_inverse: float = DEFAULT_BETA_HARTREE_INVERSE,
    config: SolverConfig = DEFAULT_CONFIG,
    keep_records: bool = False,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Solve every geometry once; return per-point ground and thermal data.

    Each element is {"r": R, "gs": {...}, "thermal": {...}}; with
    keep_records=True the GeometryRecord is attached under "record".
    """
    if grid is None:
        grid = compute_grid()
    iterator: Any = grid
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(grid, desc=f"H2/{basis_name} scan")
        except ImportError:
            pass

    backend: H2RDMBackend | None = None
    points: list[dict[str, Any]] = []
    for r in iterator:
        record = solve_h2_geometry(r, basis_name=basis_name, config=config)
        if backend is None or backend.basis.norb != record.norb:
            backend = build_rdm_backend(record.norb, *record.nelec)
        point = {
            "r": float(r),
            "gs": analyze_ground_state(record, backend, config=config),
            "thermal": analyze_thermal_state(
                record,
                backend,
                beta_hartree_inverse=beta_hartree_inverse,
                config=config,
            ),
        }
        if keep_records:
            point["record"] = record
        points.append(point)
    return points


def curve(points: Sequence[dict[str, Any]], state: str, key: str) -> np.ndarray:
    return np.array([point[state][key] for point in points])


def point_at(points: Sequence[dict[str, Any]], r_angstrom: float) -> dict[str, Any]:
    for point in points:
        if abs(point["r"] - float(r_angstrom)) < 1.0e-9:
            return point
    raise KeyError(f"No scan point at R={r_angstrom!r} A")


# ---------------------------------------------------------------------------
# H2-specific analyses 
# ---------------------------------------------------------------------------

def analytic_measures_from_occupations(probabilities: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, None)
    p = p / p.sum()
    e_ud = entropy_probabilities(p)
    return {
        "e_ud": e_ud,
        "e1": 2.0 * e_ud,
        "i_ud": 2.0 * e_ud,
        "n_ud": float(0.5 * 0.5 * (float(np.sqrt(p).sum()) ** 2 - 1.0)),
    }


def lowdin_shull_oracle(
    r_angstrom: float, basis_name: str = PRIMARY_BASIS
) -> dict[str, Any]:
    """
    Cross-solver natural-occupation route to the two-electron measures.

    A separate PySCF RHF + FCI solve with occupations from
    CISolver.make_rdm1 (spectrum of the spatial 1-RDM). Relative to the
    production pipeline this exercises a different eigensolver (Davidson vs
    dense eigh) and a different RDM code path (PySCF's four-index
    generator vs the compact contraction engine), but both routes share
    PySCF's integrals and Hamiltonian-action kernels
    """
    mol = gto.M(
        atom=(("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, float(r_angstrom)))),
        unit="Angstrom",
        basis=basis_name,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"Oracle RHF not converged at R={r_angstrom} ({basis_name})")
    ci_solver = fci.FCI(mf)
    e_fci, vector = ci_solver.kernel()
    if not bool(ci_solver.converged):
        raise RuntimeError(f"Oracle FCI not converged at R={r_angstrom} ({basis_name})")
    norb = int(mf.mo_coeff.shape[1])
    s2, _ = ci_solver.spin_square(vector, norb, mol.nelec)
    if abs(float(s2)) > 1.0e-8:
        raise RuntimeError(f"Oracle state is not a singlet at R={r_angstrom}")
    rdm1 = ci_solver.make_rdm1(vector, norb, mol.nelec)
    occupations = np.clip(np.linalg.eigvalsh(rdm1), 0.0, None) / 2.0
    if abs(float(occupations.sum()) - 1.0) > 1.0e-9:
        raise RuntimeError("Oracle occupations do not sum to one")
    measures = analytic_measures_from_occupations(occupations)
    measures.update(
        {
            "r_angstrom": float(r_angstrom),
            "basis_name": basis_name,
            "e_fci": float(e_fci),
            "occupations": np.sort(occupations)[::-1],
        }
    )
    return measures