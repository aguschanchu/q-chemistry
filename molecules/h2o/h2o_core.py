"""
H2O/STO-3G reproduction core (arXiv 2604.07633)

Setup (manuscript Sec. III): H2O at fixed H-O-H angle 104.5 deg, O-H distance
R in [0.4, 4.0] Angstrom, STO-3G basis (7 spatial orbitals), N = 10 electrons,
N_up = N_down = 5. The ground state and the thermal state of Eq. (32) both live in the 
441-dimensional fixed-M_S = 0 sector.

Method: RHF orbitals/integrals from PySCF. The complete dense spectrum is computed at every R,
degenerate clusters are simultaneously diagonalized in H and S^2, and each root carries
verified <S^2>, Var(S^2), <S_z>. The thermal state is
evaluated from this spectral decomposition.
"""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp
import matplotlib.pyplot as plt

import openfermion as of

import h2o_rdm as rdm

MODULE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = MODULE_DIR / "figures"

THETA_DEG = 104.5
BASIS_NAME = "sto-3g"
BETA_HARTREE_INVERSE = 1000.0
R_GRID = tuple(np.round(np.arange(0.4, 4.0 + 1e-9, 0.05), 10))
FIG_XLIM = (0.4, 4.0)

# Exact multiplet counts of the (10e, 7orb) M_S=0 sector
EXPECTED_MULTIPLET_COUNTS = (196, 210, 35)

FIGURE_STEMS = {
    2: "fig02_low_lying_energies",
    3: "fig03_ground_rdm_spectra",
    4: "fig04_ground_rdm_entropies",
    5: "fig05_negativities",
    6: "fig06_mutual_information",
    7: "fig07_thermal_weight_entropies",
    8: "fig08_thermal_rdm_spectra",
}


# --------------------------------------------------------------------------- #
# Configuration and result records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SolverConfig:
    scf_conv_tol: float = 1e-12
    scf_max_cycle: int = 200
    imaginary_tol: float = 1e-10
    hermiticity_tol: float = 1e-10
    residual_tol: float = 1e-8
    orthonormality_tol: float = 1e-10
    spin_cluster_tol_hartree: float = 1e-5
    spin_rounding_tol: float = 1e-2
    s2_variance_tol: float = 1e-8
    sz_tol: float = 1e-10
    trace_tol: float = 1e-8
    psd_tol: float = 5e-9
    thermal_weight_cut: float = 1e-16
    tail_bound: float = 1e-10
    band_window_hartree: float = 1e-3
    spin_block_asymmetry_tol: float = 1e-7
    pure_negativity_tol: float = 1e-9


DEFAULT_CONFIG = SolverConfig()


@dataclass(frozen=True)
class H2OSession:
    sector: rdm.SectorBasis
    split_maps: rdm.SpinSplitMaps
    pair_maps: rdm.PairBlockMaps
    generators: rdm.RDMGenerators
    s2_matrix: np.ndarray
    sz_matrix: np.ndarray


@dataclass
class GeometryRecord:
    """Complete solved M_S=0 spectrum of one geometry"""

    r_angstrom: float
    e_scf: float
    e_nuc: float
    mo_source: str
    energies: np.ndarray        
    eigenvectors: np.ndarray    
    spins: np.ndarray         
    s2: np.ndarray
    s2_variance: np.ndarray
    sz: np.ndarray
    residuals: np.ndarray
    gram_error: float
    multiplet_counts: tuple[int, int, int]


@dataclass
class StateAnalysis:
    """All plotted observables of one state (ground or thermal)."""

    kind: str
    eigs_rho_up: np.ndarray        
    eigs_rho1_up: np.ndarray        
    eigs_rho2_updown: np.ndarray    
    eigs_rho2_upup: np.ndarray      
    s_rho_up: float
    s_rho1_up: float
    s_rho2_updown: float
    s_rho2_upup: float
    n_updown: float                 # Eq. (24)
    n2_updown: float                # Eq. (26)
    n2_upup: float                  # Eq. (31)
    i_updown: float                 # Eq. (9)
    i2_updown: float                # Eq. (17b)
    trace_error: float
    spin_block_asymmetry: float
    pure_negativity_dev: float = 0.0
    m_retained: int = 1
    tail_fraction: float = 0.0
    s_p: float = 0.0
    s_q: float = 0.0


@dataclass
class GeometryResult:
    r_angstrom: float
    e_scf: float
    energies: np.ndarray
    spins: np.ndarray
    band_size: int
    band_spread: float
    band_gap: float
    gs: StateAnalysis
    thermal: StateAnalysis
    residual_max: float
    gram_error: float
    s2_variance_max: float
    sz_max: float


# --------------------------------------------------------------------------- #
# Session construction
# --------------------------------------------------------------------------- #

def build_session() -> H2OSession:
    sector = rdm.build_sector_basis()
    split_maps = rdm.build_spin_split_maps(sector)
    pair_maps = rdm.build_pair_block_maps()
    with contextlib.redirect_stderr(io.StringIO()): 
        generators = rdm.build_generators(sector)
    s2_matrix = _real_dense(
        sector.basis.get_operator_matrix(
            of.hamiltonians.s_squared_operator(rdm.N_SPATIAL)), label="S^2")
    sz_matrix = _real_dense(
        sector.basis.get_operator_matrix(
            of.hamiltonians.sz_operator(rdm.N_SPATIAL)), label="S_z")
    if np.abs(sz_matrix).max() > 1e-12:
        raise AssertionError("S_z does not vanish on the M_S=0 sector")
    return H2OSession(sector, split_maps, pair_maps, generators,
                      s2_matrix, sz_matrix)


def _real_dense(matrix, *, label: str, imaginary_tol: float = 1e-10) -> np.ndarray:
    dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    if np.iscomplexobj(dense):
        imag_norm = float(np.abs(dense.imag).max())
        if imag_norm > imaginary_tol:
            raise AssertionError(f"{label} has imaginary norm {imag_norm:.3e}")
        dense = dense.real
    return 0.5 * (dense + dense.T)


# --------------------------------------------------------------------------- #
# Electronic structure and sector Hamiltonian
# --------------------------------------------------------------------------- #

def h2o_geometry(r_angstrom: float) -> list[tuple[str, tuple[float, float, float]]]:
    """Geometry: O at the origin, both O-H bonds of length R at 104.5 deg."""
    half_angle = math.radians(THETA_DEG) / 2.0
    x = r_angstrom * math.sin(half_angle)
    z = r_angstrom * math.cos(half_angle)
    return [("O", (0.0, 0.0, 0.0)), ("H", (x, 0.0, z)), ("H", (-x, 0.0, z))]


def _spinorb_tensors_untruncated(one_body_integrals: np.ndarray,
                                 two_body_integrals: np.ndarray):
    """
    Interleaved spin-orbital tensors, untruncated.
    """
    n_spin = 2 * one_body_integrals.shape[0]
    one_body = np.zeros((n_spin, n_spin))
    one_body[0::2, 0::2] = one_body_integrals
    one_body[1::2, 1::2] = one_body_integrals
    two_body = np.zeros((n_spin,) * 4)
    two_body[0::2, 1::2, 1::2, 0::2] = two_body_integrals
    two_body[1::2, 0::2, 0::2, 1::2] = two_body_integrals
    two_body[0::2, 0::2, 0::2, 0::2] = two_body_integrals
    two_body[1::2, 1::2, 1::2, 1::2] = two_body_integrals
    return one_body, two_body


def fermion_operator_from_mo_integrals(
        constant: float, h1_mo: np.ndarray,
        eri_mo_chemist: np.ndarray) -> of.FermionOperator:
    """OpenFermion Hamiltonian from spatial MO integrals."""
    one_body, two_body = _spinorb_tensors_untruncated(
        np.asarray(h1_mo), np.asarray(eri_mo_chemist).transpose(0, 2, 3, 1))
    interaction = of.InteractionOperator(float(constant), one_body, 0.5 * two_body)
    return of.get_fermion_operator(interaction)


def run_electronic_structure(r_angstrom: float, *,
                             config: SolverConfig = DEFAULT_CONFIG):
    """RHF integrals only, with a deterministic same-geometry retry ladder"""

    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom=h2o_geometry(r_angstrom), unit="Angstrom",
                basis=BASIS_NAME, charge=0, spin=0, symmetry=False, verbose=0)
    mean_field = scf.RHF(mol)
    mean_field.conv_tol = config.scf_conv_tol
    mean_field.max_cycle = config.scf_max_cycle
    try:
        mean_field.kernel()
    except np.linalg.LinAlgError:
        pass
    mo_source = "pyscf_rhf"
    if not mean_field.converged and mean_field.mo_coeff is not None:
        mean_field = mean_field.newton()
        mean_field.kernel()
        mo_source = "pyscf_newton"
    if not mean_field.converged:
        shifted = scf.RHF(mol)
        shifted.conv_tol = config.scf_conv_tol
        shifted.max_cycle = 300
        shifted.level_shift = 0.5
        shifted.kernel()
        mean_field = scf.newton(scf.RHF(mol))
        mean_field.conv_tol = config.scf_conv_tol
        mean_field.kernel(dm0=shifted.make_rdm1())
        mo_source = "pyscf_newton_shifted"
    if not mean_field.converged:
        raise RuntimeError(
            f"SCF failed at R={r_angstrom} Angstrom "
            f"(RHF, Newton and level-shifted retries)")
    orbitals = mean_field.mo_coeff
    h1_mo = orbitals.T @ mean_field.get_hcore() @ orbitals
    eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, orbitals), orbitals.shape[1])
    fermion_op = fermion_operator_from_mo_integrals(
        mol.energy_nuc(), h1_mo, eri_mo)
    return fermion_op, {"e_scf": float(mean_field.e_tot),
                        "e_nuc": float(mol.energy_nuc()),
                        "mo_source": mo_source}


def sector_hamiltonian(fermion_op: of.FermionOperator, session: H2OSession,
                       config: SolverConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Dense real symmetric 441 x 441 Hamiltonian on the M_S=0 sector."""
    matrix = session.sector.basis.get_operator_matrix(fermion_op)
    dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    if np.iscomplexobj(dense):
        imag_norm = float(np.abs(dense.imag).max())
        if imag_norm > config.imaginary_tol:
            raise AssertionError(f"Hamiltonian imaginary norm {imag_norm:.3e}")
        dense = dense.real
    asymmetry = float(np.abs(dense - dense.T).max())
    if asymmetry > config.hermiticity_tol:
        raise AssertionError(f"Hamiltonian asymmetry {asymmetry:.3e}")
    return 0.5 * (dense + dense.T)


# --------------------------------------------------------------------------- #
# Spin purification and the per-geometry solve
# --------------------------------------------------------------------------- #

def _spin_from_s2(s2_value: float, tolerance: float) -> int:
    raw = 0.5 * (-1.0 + math.sqrt(max(0.0, 1.0 + 4.0 * float(s2_value))))
    spin = int(round(raw))
    if abs(raw - spin) > tolerance:
        raise RuntimeError(f"eigenstate is not spin-pure: <S^2>={s2_value:.6g}")
    return spin


def _energy_clusters(energies: Sequence[float], tolerance: float) -> list[list[int]]:
    clusters: list[list[int]] = [[0]]
    for index in range(1, len(energies)):
        if abs(float(energies[index] - energies[index - 1])) <= tolerance:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    return clusters


def _purify_spin_clusters(values: np.ndarray, vectors: np.ndarray,
                          s2_matrix: np.ndarray, hamiltonian: np.ndarray,
                          cluster_tolerance: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Simultaneously diagonalize H and S^2 inside degenerate energy clusters.

    Projected S^2 is diagonalized within each cluster, then H is
    re-diagonalized inside every equal-spin subspace.
    """
    values = values.copy()
    vectors = vectors.copy()
    for cluster in _energy_clusters(values, cluster_tolerance):
        if len(cluster) < 2:
            continue
        block = vectors[:, cluster]
        projected_s2 = block.T @ (s2_matrix @ block)
        s2_values, rotation = np.linalg.eigh(
            0.5 * (projected_s2 + projected_s2.T))
        rotated = block @ rotation
        spins = np.array([
            round(0.5 * (-1.0 + math.sqrt(max(0.0, 1.0 + 4.0 * float(s2)))))
            for s2 in s2_values
        ])
        cluster_values = np.empty(len(cluster))
        for spin in np.unique(spins):
            members = np.flatnonzero(spins == spin)
            subspace = rotated[:, members]
            projected_h = subspace.T @ (hamiltonian @ subspace)
            sub_values, sub_rotation = np.linalg.eigh(
                0.5 * (projected_h + projected_h.T))
            rotated[:, members] = subspace @ sub_rotation
            cluster_values[members] = sub_values
        vectors[:, cluster] = rotated
        values[cluster] = cluster_values
    order = np.argsort(values, kind="stable")
    return values[order], vectors[:, order]


def solve_geometry(r_angstrom: float, session: H2OSession, *,
                   config: SolverConfig = DEFAULT_CONFIG) -> GeometryRecord:
    """Complete M_S=0 eigensystem at one geometry."""
    r_angstrom = float(r_angstrom)
    fermion_op, meta = run_electronic_structure(r_angstrom, config=config)
    hamiltonian = sector_hamiltonian(fermion_op, session, config)

    values, vectors = np.linalg.eigh(hamiltonian)
    values, vectors = _purify_spin_clusters(
        values, vectors, session.s2_matrix, hamiltonian,
        config.spin_cluster_tol_hartree)

    residuals = np.linalg.norm(hamiltonian @ vectors - vectors * values, axis=0)
    gram_error = float(np.abs(vectors.T @ vectors - np.eye(len(values))).max())
    s2_action = session.s2_matrix @ vectors
    s2_values = np.einsum("ik,ik->k", vectors, s2_action)
    s2_variance = np.einsum("ik,ik->k", s2_action, s2_action) - s2_values ** 2
    sz_values = np.einsum("ik,ik->k", vectors, session.sz_matrix @ vectors)

    if residuals.max() > config.residual_tol:
        raise AssertionError(
            f"R={r_angstrom}: residual {residuals.max():.3e} above tolerance, please re-run")
    if gram_error > config.orthonormality_tol:
        raise AssertionError(f"R={r_angstrom}: Gram error {gram_error:.3e}")
    if np.abs(s2_variance).max() > config.s2_variance_tol:
        raise AssertionError(
            f"R={r_angstrom}: Var(S^2) {np.abs(s2_variance).max():.3e}")
    if np.abs(sz_values).max() > config.sz_tol:
        raise AssertionError(f"R={r_angstrom}: <S_z> deviates from zero")

    spins = np.array([
        _spin_from_s2(s2, config.spin_rounding_tol) for s2 in s2_values
    ], dtype=np.int8)
    counts = tuple(int((spins == spin).sum()) for spin in (0, 1, 2))
    if counts != EXPECTED_MULTIPLET_COUNTS:
        raise AssertionError(
            f"R={r_angstrom}: multiplet counts {counts} != "
            f"{EXPECTED_MULTIPLET_COUNTS}")
    if spins[0] != 0:
        raise AssertionError(f"R={r_angstrom}: ground state is not a singlet")

    return GeometryRecord(
        r_angstrom=r_angstrom, e_scf=meta["e_scf"], e_nuc=meta["e_nuc"],
        mo_source=meta["mo_source"], energies=values, eigenvectors=vectors,
        spins=spins, s2=s2_values, s2_variance=s2_variance, sz=sz_values,
        residuals=residuals, gram_error=gram_error, multiplet_counts=counts)


# --------------------------------------------------------------------------- #
# Thermal weights and state analysis
# --------------------------------------------------------------------------- #

@dataclass
class ThermalWeights:
    p_full: np.ndarray
    retained_indices: np.ndarray
    retained_p: np.ndarray
    tail_fraction: float    
    s_p: float
    s_q: float


def thermal_weights(energies: np.ndarray, *,
                    beta: float = BETA_HARTREE_INVERSE,
                    config: SolverConfig = DEFAULT_CONFIG) -> ThermalWeights:
    """Eq. (32) weights over the complete sector spectrum."""
    log_q = -beta * (np.asarray(energies) - float(energies[0]))
    log_partition = float(logsumexp(log_q))
    p_full = np.exp(log_q - log_partition)
    retained = np.flatnonzero(log_q > math.log(config.thermal_weight_cut))
    dropped = np.setdiff1d(np.arange(len(log_q)), retained)
    tail_fraction = float(np.exp(logsumexp(log_q[dropped]) - log_partition)) \
        if len(dropped) else 0.0
    if tail_fraction > config.tail_bound:
        raise AssertionError(f"thermal tail {tail_fraction:.3e} above bound")
    retained_p = p_full[retained] / p_full[retained].sum()
    return ThermalWeights(
        p_full=p_full, retained_indices=retained,
        retained_p=retained_p, tail_fraction=tail_fraction,
        s_p=rdm.shannon_bits(p_full),
        s_q=rdm.weight_entropy_bits(np.exp(log_q)))


def analyze_state(states: np.ndarray, weights: np.ndarray,
                  session: H2OSession, *, kind: str,
                  config: SolverConfig = DEFAULT_CONFIG,
                  thermal: Optional[ThermalWeights] = None) -> StateAnalysis:
    """All plotted observables of a pure state or spectral mixture."""

    weights = np.asarray(weights, dtype=float)
    blocks = rdm.reduced_blocks(states, weights, session.generators,
                                session.pair_maps)
    rho_up = rdm.rho_up_from_states(states, weights, session.split_maps)
    rho_down = rdm.rho_down_from_states(states, weights, session.split_maps)

    expected = rdm.expected_traces()
    trace_error = max(
        abs(np.trace(blocks[name]).real - expected[name])
        for name in ("rho1_full", "rho1_up", "rho1_down", "rho2_full",
                     "rho2_upup", "rho2_dndn", "rho2_updown"))
    trace_error = max(trace_error, abs(np.trace(rho_up).real - 1.0),
                      abs(np.trace(rho_down).real - 1.0))
    if trace_error > config.trace_tol:
        raise AssertionError(f"{kind}: RDM trace error {trace_error:.3e}")

    eigs_rho_up = rdm.hermitian_spectrum(rho_up, psd_tolerance=config.psd_tol,
                                         label="rho_up")
    eigs_rho1_up = rdm.hermitian_spectrum(blocks["rho1_up"],
                                          psd_tolerance=config.psd_tol,
                                          label="rho1_up")
    eigs_rho2_updown = rdm.hermitian_spectrum(blocks["rho2_updown"],
                                              psd_tolerance=config.psd_tol,
                                              label="rho2_updown")
    eigs_rho2_upup = rdm.hermitian_spectrum(blocks["rho2_upup"],
                                            psd_tolerance=config.psd_tol,
                                            label="rho2_upup")

    entropy_up = rdm.spectral_entropy_bits(eigs_rho_up)
    entropy_down = rdm.spectral_entropy_bits(
        rdm.hermitian_spectrum(rho_down, psd_tolerance=config.psd_tol,
                               label="rho_down"))
    entropy_rho1_up = rdm.spectral_entropy_bits(eigs_rho1_up)
    entropy_rho1_down = rdm.spectral_entropy_bits(
        rdm.hermitian_spectrum(blocks["rho1_down"],
                               psd_tolerance=config.psd_tol, label="rho1_down"))
    entropy_rho2_updown = rdm.spectral_entropy_bits(eigs_rho2_updown)
    entropy_rho2_upup = rdm.spectral_entropy_bits(eigs_rho2_upup)

    n_updown = rdm.total_negativity(states, weights, session.split_maps)
    n2_updown = rdm.opposite_spin_negativity(blocks["rho2_updown"])
    n2_upup = rdm.same_spin_negativity(blocks["rho2_upup"],
                                       session.pair_maps.upup_pairs)

    global_entropy = rdm.shannon_bits(weights) if kind == "thermal" else 0.0
    i_updown = entropy_up + entropy_down - global_entropy
    i2_updown = (rdm.N_DOWN * entropy_rho1_up + rdm.N_UP * entropy_rho1_down
                 - entropy_rho2_updown)

    spin_block_asymmetry = max(
        float(np.abs(blocks["rho1_up"] - blocks["rho1_down"]).max()),
        float(np.abs(blocks["rho2_upup"] - blocks["rho2_dndn"]).max()))
    if spin_block_asymmetry > config.spin_block_asymmetry_tol:
        raise AssertionError(
            f"{kind}: up/down block asymmetry {spin_block_asymmetry:.3e}")

    pure_negativity_dev = 0.0
    if kind == "gs":
        pure_negativity_dev = abs(
            n_updown - rdm.pure_total_negativity(
                np.atleast_2d(states)[0], session.split_maps))
        if pure_negativity_dev > config.pure_negativity_tol:
            raise AssertionError(
                f"gs: Eq. (24) vs Eq. (25) deviation {pure_negativity_dev:.3e}")

    return StateAnalysis(
        kind=kind,
        eigs_rho_up=eigs_rho_up, eigs_rho1_up=eigs_rho1_up,
        eigs_rho2_updown=eigs_rho2_updown, eigs_rho2_upup=eigs_rho2_upup,
        s_rho_up=entropy_up, s_rho1_up=entropy_rho1_up,
        s_rho2_updown=entropy_rho2_updown, s_rho2_upup=entropy_rho2_upup,
        n_updown=n_updown, n2_updown=n2_updown, n2_upup=n2_upup,
        i_updown=i_updown, i2_updown=i2_updown,
        trace_error=trace_error, spin_block_asymmetry=spin_block_asymmetry,
        pure_negativity_dev=pure_negativity_dev,
        m_retained=len(weights),
        tail_fraction=thermal.tail_fraction if thermal is not None else 0.0,
        s_p=thermal.s_p if thermal is not None else 0.0,
        s_q=thermal.s_q if thermal is not None else 0.0)


def analyze_geometry(record: GeometryRecord, session: H2OSession, *,
                     beta: float = BETA_HARTREE_INVERSE,
                     config: SolverConfig = DEFAULT_CONFIG) -> GeometryResult:
    """Ground-state and thermal analyses sharing one spectral solve."""
    ground = analyze_state(record.eigenvectors[:, 0], np.array([1.0]),
                           session, kind="gs", config=config)
    weights = thermal_weights(record.energies, beta=beta, config=config)
    thermal = analyze_state(
        record.eigenvectors[:, weights.retained_indices].T,
        weights.retained_p, session, kind="thermal", config=config,
        thermal=weights)

    in_band = record.energies <= record.energies[0] + config.band_window_hartree
    band_size = int(in_band.sum())
    band_spread = float(record.energies[band_size - 1] - record.energies[0])
    band_gap = float(record.energies[band_size] - record.energies[band_size - 1]) \
        if band_size < len(record.energies) else float("nan")

    return GeometryResult(
        r_angstrom=record.r_angstrom, e_scf=record.e_scf,
        energies=record.energies, spins=record.spins,
        band_size=band_size, band_spread=band_spread, band_gap=band_gap,
        gs=ground, thermal=thermal,
        residual_max=float(record.residuals.max()),
        gram_error=record.gram_error,
        s2_variance_max=float(np.abs(record.s2_variance).max()),
        sz_max=float(np.abs(record.sz).max()))


def run_scan(grid: Sequence[float] = R_GRID, *,
             session: Optional[H2OSession] = None,
             beta: float = BETA_HARTREE_INVERSE,
             config: SolverConfig = DEFAULT_CONFIG,
             progress: bool = True) -> list[GeometryResult]:
    """One deterministic pass over the R grid."""
    if session is None:
        session = build_session()
    iterator: Sequence[float] = list(grid)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="R scan", ncols=80, mininterval=5.0)
    results = []
    for r_angstrom in iterator:
        record = solve_geometry(r_angstrom, session, config=config)
        results.append(analyze_geometry(record, session, beta=beta,
                                        config=config))
    return results


def scan_arrays(results: Sequence[GeometryResult]) -> dict[str, np.ndarray]:
    """R-major arrays holding everything the figures need."""
    def per_state(name: str, attribute: str) -> np.ndarray:
        return np.array([getattr(getattr(point, name), attribute)
                         for point in results])

    arrays: dict[str, np.ndarray] = {
        "r": np.array([point.r_angstrom for point in results]),
        "energies": np.array([point.energies for point in results]),
        "spins": np.array([point.spins for point in results], dtype=np.int8),
    }
    for prefix in ("gs", "thermal"):
        for attribute in ("eigs_rho_up", "eigs_rho1_up", "eigs_rho2_updown",
                          "eigs_rho2_upup", "s_rho_up", "s_rho1_up",
                          "s_rho2_updown", "s_rho2_upup", "n_updown",
                          "n2_updown", "n2_upup", "i_updown", "i2_updown"):
            arrays[f"{prefix}_{attribute}"] = per_state(prefix, attribute)
    for attribute in ("m_retained", "tail_fraction", "s_p", "s_q"):
        arrays[f"thermal_{attribute}"] = per_state("thermal", attribute)
    return arrays


# --------------------------------------------------------------------------- #
# Figures: one compute + one plot function per manuscript figure
# --------------------------------------------------------------------------- #

_SPECTRUM_STYLE = [
    ("eigs_rho_up", "tab:blue", r"$\Gamma^2_\nu$"),
    ("eigs_rho1_up", "tab:orange", r"$\lambda[\rho^{(1)}_\uparrow]$"),
    ("eigs_rho2_updown", "tab:green",
     r"$\lambda[\rho^{(2)}_{\uparrow\!\downarrow}]$"),
    ("eigs_rho2_upup", "tab:red",
     r"$\lambda[\rho^{(2)}_{\uparrow\!\uparrow}]$"),
]
_ENTROPY_STYLE = [
    ("s_rho_up", "tab:blue", r"$\rho_\uparrow$"),
    ("s_rho1_up", "tab:orange", r"$\rho^{(1)}_\uparrow$"),
    ("s_rho2_updown", "tab:green", r"$\rho^{(2)}_{\uparrow\!\downarrow}$"),
    ("s_rho2_upup", "tab:red", r"$\rho^{(2)}_{\uparrow\!\uparrow}$"),
]
ENTROPY_SATURATION = {
    "s_rho_up": math.log2(21),
    "s_rho1_up": 5 * math.log2(7 / 5),
    "s_rho2_updown": 25 * math.log2(49 / 25),
    "s_rho2_upup": 10 * math.log2(21 / 10),
}


def save_figure(figure, stem: str, figures_dir: Path = FIGURES_DIR) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(figures_dir / f"{stem}.png", bbox_inches="tight", dpi=300)


def compute_figure2_data(arrays: dict[str, np.ndarray]) -> dict:
    """Per spin S, the k-th lowest total energy at each R."""
    data = {"r": arrays["r"]}
    for spin, count in zip((0, 1, 2), EXPECTED_MULTIPLET_COUNTS):
        levels = np.empty((len(arrays["r"]), count))
        for row, (energies, spins) in enumerate(
                zip(arrays["energies"], arrays["spins"])):
            levels[row] = np.sort(energies[spins == spin])
        data[f"levels_s{spin}"] = levels
    return data


def plot_figure2(data: dict, figures_dir: Path = FIGURES_DIR):
    styles = {0: ("black", "-", r"$S=0$"), 1: ("tab:blue", "--", r"$S=1$"),
              2: ("tab:red", "-.", r"$S=2$")}
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for spin in (0, 1, 2):
        color, linestyle, label = styles[spin]
        axis.plot(data["r"], data[f"levels_s{spin}"], color=color,
                  linestyle=linestyle, linewidth=0.8)
        axis.plot([], [], color=color, linestyle=linestyle, label=label)
    axis.set_xlim(*FIG_XLIM)
    axis.set_ylim(-75.05, -74.55)
    axis.set_xlabel(r"$R(\mathrm{\AA})$")
    axis.set_ylabel(r"Energy $(E_h)$")
    axis.legend(loc="upper right", framealpha=1.0)
    save_figure(figure, FIGURE_STEMS[2], figures_dir)
    return figure


def _spectra_data(arrays: dict[str, np.ndarray], prefix: str) -> dict:
    data = {"r": arrays["r"]}
    for name, _, _ in _SPECTRUM_STYLE:
        data[name] = arrays[f"{prefix}_{name}"]
    return data


def compute_figure3_data(arrays: dict[str, np.ndarray]) -> dict:
    return _spectra_data(arrays, "gs")


def plot_figure3(data: dict, figures_dir: Path = FIGURES_DIR):
    figure, axes = plt.subplots(4, 1, figsize=(5.4, 8.6), sharex=True)

    for axis, (name, color, label) in zip(axes, _SPECTRUM_STYLE):
        axis.plot(data["r"], data[name], color=color, linewidth=0.9)
        axis.set_ylabel(label)
        axis.set_ylim(0.0, 1.05)
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.set_xlim(*FIG_XLIM)
    axes[-1].set_xlabel(r"$R(\mathrm{\AA})$")
    figure.suptitle("Ground state", y=0.92)
    save_figure(figure, FIGURE_STEMS[3], figures_dir)
    return figure


def compute_figure4_data(arrays: dict[str, np.ndarray]) -> dict:
    data = {"r": arrays["r"]}
    for name, _, _ in _ENTROPY_STYLE:
        data[name] = arrays[f"gs_{name}"]
    return data


def plot_figure4(data: dict, figures_dir: Path = FIGURES_DIR):
    figure, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(5.6, 7.2), sharex=True)
    for name, color, label in _ENTROPY_STYLE:
        axis_top.plot(data["r"], data[name], color=color, label=label)
        axis_bottom.plot(data["r"], data[name] / ENTROPY_SATURATION[name],
                         color=color, label=label)
    axis_top.set_ylabel(r"$S(\rho)$")
    axis_top.set_ylim(0, 16)
    axis_top.legend(loc="upper left", framealpha=1.0)
    axis_bottom.set_ylabel(r"$S(\rho)/S_{\max}$")
    axis_bottom.set_ylim(0, 0.9)
    axis_bottom.legend(loc="upper left", framealpha=1.0)
    axis_bottom.set_xlabel(r"$R(\mathrm{\AA})$")
    axis_bottom.set_xlim(*FIG_XLIM)
    save_figure(figure, FIGURE_STEMS[4], figures_dir)
    return figure


def compute_figure5_data(arrays: dict[str, np.ndarray]) -> dict:
    """Stored values are on the Eq. (24)/(26)/(31) scale"""
    
    data = {"r": arrays["r"],
            "display_factor_updown2": np.array(0.5)}
    for prefix in ("gs", "thermal"):
        for name in ("n_updown", "n2_updown", "n2_upup"):
            data[f"{prefix}_{name}"] = arrays[f"{prefix}_{name}"]
    return data


def plot_figure5(data: dict, figures_dir: Path = FIGURES_DIR):
    figure, axes = plt.subplots(2, 1, figsize=(5.6, 7.6))
    display_factor = float(data["display_factor_updown2"])
    for axis, prefix, title in zip(axes, ("gs", "thermal"),
                                   ("Ground state", "Thermal state")):
        axis.plot(data["r"], data[f"{prefix}_n_updown"], color="purple",
                  label=r"$\mathcal{N}_{\uparrow\!\downarrow}$")
        axis.plot(data["r"], display_factor * data[f"{prefix}_n2_updown"],
                  color="tab:green",
                  label=r"$\mathcal{N}^{(2)}_{\uparrow\!\downarrow}$")
        axis.plot(data["r"], data[f"{prefix}_n2_upup"], color="tab:red",
                  label=r"$\mathcal{N}^{(2)}_{\uparrow\!\uparrow}$")
        twin = axis.twinx()
        twin.plot(data["r"], data[f"{prefix}_n2_upup"], color="tab:red",
                  linestyle="--")
        twin.set_ylim(0.0, 0.035)
        twin.set_yticks(np.arange(0.0, 0.0301, 0.005))
        twin.set_ylabel(r"$\mathcal{N}^{(2)}_{\uparrow\!\uparrow}$")
        axis.set_xlim(*FIG_XLIM)
        axis.set_ylim(0.0, 2.3)
        axis.set_ylabel(r"$\mathcal{N}$")
        axis.set_title(title)
        axis.set_zorder(twin.get_zorder() + 1)
        axis.patch.set_visible(False)
        axis.legend(loc="center right" if prefix == "gs" else "upper left",
                    framealpha=1.0)
    axes[1].set_xlabel(r"$R(\mathrm{\AA})$")
    figure.tight_layout()
    save_figure(figure, FIGURE_STEMS[5], figures_dir)
    return figure


def compute_figure6_data(arrays: dict[str, np.ndarray]) -> dict:
    data = {"r": arrays["r"]}
    for prefix in ("gs", "thermal"):
        for name in ("i_updown", "i2_updown"):
            data[f"{prefix}_{name}"] = arrays[f"{prefix}_{name}"]
    return data


def plot_figure6(data: dict, figures_dir: Path = FIGURES_DIR):
    figure, axes = plt.subplots(2, 1, figsize=(5.6, 7.2))
    for axis, prefix, title in zip(axes, ("gs", "thermal"),
                                   ("Ground state", "Thermal state")):
        axis.plot(data["r"], data[f"{prefix}_i_updown"], color="black",
                  label=r"$I_{\uparrow\!\downarrow}$")
        axis.plot(data["r"], data[f"{prefix}_i2_updown"], color="gray",
                  linestyle="--", label=r"$I^{(2)}_{\uparrow\!\downarrow}$")
        axis.set_xlim(*FIG_XLIM)
        axis.set_ylim(0.0, 4.8)
        axis.set_ylabel(r"$I$")
        axis.set_title(title)
        axis.legend(loc="upper left", framealpha=1.0)
    axes[1].set_xlabel(r"$R(\mathrm{\AA})$")
    figure.tight_layout()
    save_figure(figure, FIGURE_STEMS[6], figures_dir)
    return figure


def compute_figure7_data(arrays: dict[str, np.ndarray]) -> dict:
    return {"r": arrays["r"], "s_p": arrays["thermal_s_p"],
            "s_q": arrays["thermal_s_q"]}


def plot_figure7(data: dict, figures_dir: Path = FIGURES_DIR):
    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    axis.plot(data["r"], data["s_p"], color="tab:blue",
              label=r"$S(\mathbf{p})$")
    axis.plot(data["r"], data["s_q"], color="tab:orange", linestyle="--",
              label=r"$S(\mathbf{q})$")
    axis.axhline(math.log2(12), color="gray", linestyle=":", linewidth=0.8)
    axis.set_xlim(*FIG_XLIM)
    axis.set_ylim(0.0, 5.6)
    axis.set_xlabel(r"$R(\mathrm{\AA})$")
    axis.set_ylabel(r"$S$")
    axis.legend(loc="upper left", framealpha=1.0)
    save_figure(figure, FIGURE_STEMS[7], figures_dir)
    return figure


def compute_figure8_data(arrays: dict[str, np.ndarray]) -> dict:
    return _spectra_data(arrays, "thermal")


def plot_figure8(data: dict, figures_dir: Path = FIGURES_DIR):
    figure, axes = plt.subplots(4, 1, figsize=(5.4, 8.6), sharex=True)
    labels = [r"$\lambda[\rho_\uparrow]$"] + \
        [label for _, _, label in _SPECTRUM_STYLE[1:]]
    for axis, (name, color, _), label in zip(axes, _SPECTRUM_STYLE, labels):
        axis.plot(data["r"], data[name], color=color, linewidth=0.9)
        axis.set_ylabel(label)
        axis.set_ylim(0.0, 1.05)
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xlim(*FIG_XLIM)
    axes[-1].set_xlabel(r"$R(\mathrm{\AA})$")
    figure.suptitle("Thermal state", y=0.92)
    save_figure(figure, FIGURE_STEMS[8], figures_dir)
    return figure
