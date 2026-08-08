"""
H2O in the 6-31G basis: two-body negativities for manuscript Fig. 10.

This module solves the same problem with PySCF's iterative FCI machinery:

* RHF orbitals, MO integrals via ao2mo.
* Ground state: fci.direct_spin0 with a spin-penalty ladder selecting S=0.
* Thermal state (Eq. 32, beta = 1000 Eh^-1): lowest nroots M_S=0 roots
  from fci.direct_spin1, Boltzmann weights closed over degenerate
  clusters. 
* Spin-resolved two-body density blocks from fci.direct_spin1.make_rdm12s.

    n2_updown  Eq. (26): minus the sum of negative eigenvalues of the
               partial transpose of the 169x169 up-down product block.
    n2_upup    Eq. (31): minus the sum of negative eigenvalues of
               (1/2) rho^(2)tp built from the antisymmetrized up-up block.

Usage (from molecules/h2o):

    python h2o_631g.py --scan [--workers N] [--smoke]   # 6-31G scan
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"

# Shared physical conventions 
THETA_DEG = 104.5
BETA_HARTREE_INVERSE = 1000.0
FIG_XLIM = (0.4, 4.0)

BASIS_631G = "6-31g"
R_GRID_631G = tuple(np.round(np.arange(0.4, 4.0 + 1e-9, 0.1), 10))
R_GRID_STO3G_CHECK = tuple(np.round(np.arange(0.4, 4.0 + 1e-9, 0.05), 10))
NROOTS_THERMAL = 16
NROOTS_INNER = 4
R_INNER_MAX = 2.0
BAND_R_MIN = 3.4
DEGENERACY_TOL = 1.0e-7
SPIN_PURE_TOL = 1.0e-5

FIGURE_STEM_10 = "fig10_negativities"


# --------------------------------------------------------------------------- #
# Negativities from spin-resolved two-body blocks (equation scale)
# --------------------------------------------------------------------------- #

def hermitize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    return 0.5 * (matrix + matrix.conj().T)


def partial_transpose_beta(product_matrix: np.ndarray,
                           dim_alpha: int, dim_beta: int) -> np.ndarray:
    expected = dim_alpha * dim_beta
    matrix = np.asarray(product_matrix)
    if matrix.shape != (expected, expected):
        raise ValueError("Wrong product-basis matrix shape")
    return (matrix.reshape(dim_alpha, dim_beta, dim_alpha, dim_beta)
                  .transpose(0, 3, 2, 1)
                  .reshape(expected, expected))


def opposite_spin_negativity(gamma_updown: np.ndarray, norb: int,
                             baseline: float) -> float:
    """Eq. (26) on the equation scale: minus the negative-eigenvalue sum."""
    pt = partial_transpose_beta(gamma_updown, norb, norb)
    hermiticity = float(np.linalg.norm(pt - pt.conj().T))
    if hermiticity > 5.0e-7:
        raise AssertionError(f"up-down PT not Hermitian: {hermiticity:.3e}")
    eigenvalues = np.linalg.eigvalsh(hermitize(pt))
    negative_sum = float(-eigenvalues[eigenvalues < -1.0e-12].sum())
    trace_norm_formula = 0.5 * (float(np.abs(eigenvalues).sum()) - baseline)
    if abs(negative_sum - trace_norm_formula) > 5.0e-7:
        raise AssertionError("Opposite-spin negativity formulas disagree")
    return negative_sum


def extend_same_spin(compact_block: np.ndarray, norb: int) -> np.ndarray:
    """Wedge (i<j) block -> antisymmetrized norb^2 x norb^2 tensor."""
    pairs = list(combinations(range(norb), 2))
    block = np.asarray(compact_block)
    if block.shape != (len(pairs), len(pairs)):
        raise ValueError("Wrong compact same-spin block shape")
    extended = np.zeros((norb, norb, norb, norb), dtype=block.dtype)
    idx = np.array(pairs)
    i, j = idx[:, 0], idx[:, 1]
    for a, (p, q) in enumerate(pairs):
        row = block[a]
        extended[p, q][i, j] = row
        extended[q, p][i, j] = -row
        extended[p, q][j, i] = -row
        extended[q, p][j, i] = row
    return extended


def same_spin_negativity(compact_block: np.ndarray, norb: int) -> float:
    """Eq. (28)-(31): minus the negative-eigenvalue sum of rho^(2)tp."""
    if norb < 2 or np.asarray(compact_block).size == 0:
        return 0.0
    block = np.asarray(compact_block)
    if np.max(np.abs(np.imag(block))) > 2.0e-11:
        raise AssertionError("Same-spin witness requires a real representation")
    extended = np.real(extend_same_spin(block, norb))
    tp4 = (extended.transpose(0, 3, 2, 1)
           - extended.transpose(0, 3, 1, 2))
    scaled = 0.5 * tp4.reshape(norb * norb, norb * norb)
    symmetry = float(np.linalg.norm(scaled - scaled.T))
    if symmetry > 5.0e-7:
        raise AssertionError(f"same-spin PT not symmetric: {symmetry:.3e}")
    eigenvalues = np.linalg.eigvalsh(0.5 * (scaled + scaled.T))
    negative_sum = float(-eigenvalues[eigenvalues < -1.0e-12].sum())
    baseline = float(np.trace(block).real)
    trace_norm_formula = 0.5 * (float(np.abs(eigenvalues).sum()) - baseline)
    if abs(negative_sum - trace_norm_formula) > 5.0e-7:
        raise AssertionError("Same-spin negativity formulas disagree")
    return negative_sum


def pyscf_spin_blocks(ci: np.ndarray, norb: int, nelec) -> dict:
    """Wedge up-up and product up-down blocks from a CI vector."""
    from pyscf import fci
    _, (dm2aa, dm2ab, _) = fci.direct_spin1.make_rdm12s(
        np.asarray(ci), norb, nelec)
    pairs = list(combinations(range(norb), 2))
    idx = np.array(pairs)
    i, j = idx[:, 0], idx[:, 1]
    # gamma[(i,j),(k,l)] = dm2[k, i, l, j]; T[i, j, k, l] = dm2[k, i, l, j]
    t_aa = dm2aa.transpose(1, 3, 0, 2)
    gamma_upup = t_aa[i, j][:, i, j]
    gamma_updown = (dm2ab.transpose(1, 3, 0, 2)
                         .reshape(norb * norb, norb * norb))
    return {"gamma_upup": gamma_upup, "gamma_updown": gamma_updown}


def weighted_blocks(blocks: list[dict], weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=float)
    if len(blocks) != len(weights) or abs(weights.sum() - 1.0) > 2.0e-12:
        raise ValueError("Invalid RDM mixture")
    names = tuple(blocks[0])
    return {
        name: sum(float(weight) * block[name]
                  for weight, block in zip(weights, blocks))
        for name in names
    }


def clustered_weights(energies: np.ndarray, beta: float = BETA_HARTREE_INVERSE,
                      tol: float = DEGENERACY_TOL) -> np.ndarray:
    """Boltzmann weights with degenerate clusters closed (mean energy)."""
    energies = np.asarray(energies, dtype=float)
    clustered = energies.copy()
    start = 0
    while start < len(energies):
        stop = start + 1
        while (stop < len(energies)
               and abs(energies[stop] - energies[stop - 1]) <= tol):
            stop += 1
        clustered[start:stop] = np.mean(energies[start:stop])
        start = stop
    logw = -beta * (clustered - clustered[0])
    weights = np.exp(logw - logsumexp(logw))
    return weights


def negativities_from_blocks(blocks: dict, nelec, norb: int) -> dict:
    na, nb = nelec
    trace_updown = float(np.trace(blocks["gamma_updown"]).real)
    if abs(trace_updown - na * nb) > 1.0e-6:
        raise AssertionError(
            f"Tr gamma_updown = {trace_updown}, expected {na * nb}")
    trace_upup = float(np.trace(blocks["gamma_upup"]).real)
    if abs(trace_upup - na * (na - 1) / 2) > 1.0e-6:
        raise AssertionError(
            f"Tr gamma_upup = {trace_upup}, expected {na * (na - 1) / 2}")
    return {
        "n2_updown": opposite_spin_negativity(
            blocks["gamma_updown"], norb, na * nb),
        "n2_upup": same_spin_negativity(blocks["gamma_upup"], norb),
    }


# --------------------------------------------------------------------------- #
# Electronic structure (FCI)
# --------------------------------------------------------------------------- #

def h2o_geometry(r_angstrom: float) -> list:
    """Geometry: O at the origin, both O-H bonds of length R at 104.5 deg."""
    half_angle = math.radians(THETA_DEG) / 2.0
    x = r_angstrom * math.sin(half_angle)
    z = r_angstrom * math.cos(half_angle)
    return [("O", (0.0, 0.0, 0.0)), ("H", (x, 0.0, z)), ("H", (-x, 0.0, z))]


def rhf_orbitals(mol):
    """RHF MOs with a Newton fallback."""
    from pyscf import scf
    mean_field = scf.RHF(mol)
    mean_field.conv_tol = 1.0e-11
    mean_field.max_cycle = 200
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = scf.RHF(mol).newton()
        mean_field.conv_tol = 1.0e-11
        mean_field.max_cycle = 200
        mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("SCF failed (RHF and Newton retries)")
    return np.asarray(mean_field.mo_coeff), float(mean_field.e_tot)


def _fix_spin_ladder(mol, h1_mo, eri_mo, norb, nelec, enuc, ci0=None):
    """Singlet ground state via direct_spin0 with a spin-penalty ladder."""
    from pyscf import fci
    energy = math.inf
    s2 = float("nan")
    ci = None
    for shift in (0.0, 0.4, 1.5):
        solver = fci.direct_spin0.FCI(mol)
        if shift:
            solver = fci.addons.fix_spin_(solver, shift=shift, ss=0.0)
        solver.conv_tol = 1.0e-10
        solver.max_cycle = 400
        solver.lindep = 1.0e-14
        solver.pspace_size = 1500
        try:
            energy, ci = solver.kernel(h1_mo, eri_mo, norb, nelec,
                                       ecore=enuc, nroots=1, ci0=ci0)
        except ValueError:
            energy, ci = math.inf, None
            continue
        ci = np.asarray(ci)
        s2 = float(fci.spin_op.spin_square0(ci, norb, nelec)[0])
        if np.all(np.atleast_1d(solver.converged)) and abs(s2) < SPIN_PURE_TOL:
            return float(energy), ci, s2, True
    if ci is None:
        return math.inf, None, float("nan"), False
    return float(energy), ci, s2, False


def solve_point(r_angstrom: float, *, basis: str = BASIS_631G,
                nroots: int = NROOTS_THERMAL) -> dict:
    """Ground-state and thermal negativities at one geometry."""
    from pyscf import ao2mo, fci, gto

    max_memory = int(os.environ.get("QCHEM_MAX_MEMORY_MB", 4500))
    mol = gto.M(atom=h2o_geometry(r_angstrom), unit="Angstrom", basis=basis,
                charge=0, spin=0, symmetry=False, verbose=0,
                max_memory=max_memory)
    nelec = mol.nelec
    na, nb = nelec
    mo_coeff, e_scf = rhf_orbitals(mol)
    norb = mo_coeff.shape[1]
    h1_mo = np.asarray(mo_coeff.T @ mol.intor("int1e_kin") @ mo_coeff
                       + mo_coeff.T @ mol.intor("int1e_nuc") @ mo_coeff,
                       order="C")
    eri_mo = np.asarray(ao2mo.kernel(mol, mo_coeff, aosym="s4"), order="C")
    enuc = float(mol.energy_nuc())

    gs_energy, gs_ci, gs_s2, gs_converged = _fix_spin_ladder(
        mol, h1_mo, eri_mo, norb, nelec, enuc)

    # Thermal roots: lowest M_S=0 states of any S from direct_spin1
    if r_angstrom < R_INNER_MAX:
        nroots = min(nroots, NROOTS_INNER)
    expected_band = 12 if r_angstrom >= BAND_R_MIN else 1
    band_incomplete = False
    for attempt_nroots, pspace in ((nroots, 1500), (nroots + 4, 4000)):
        solver = fci.direct_spin1.FCI(mol)
        solver.conv_tol = 1.0e-8
        solver.max_cycle = 400
        solver.max_space = max(16, 2 * attempt_nroots + 8)
        solver.lindep = 1.0e-14
        solver.pspace_size = pspace
        solver.nroots = attempt_nroots
        energies, cis = solver.kernel(h1_mo, eri_mo, norb, nelec,
                                      ecore=enuc, nroots=attempt_nroots)
        converged = np.atleast_1d(solver.converged)
        energies = np.atleast_1d(np.asarray(energies, dtype=float))
        cis = list(cis) if isinstance(cis, (list, tuple)) else [cis]
        n_band = int(np.count_nonzero(energies - energies.min() < 1.0e-3))
        if n_band >= expected_band:
            band_incomplete = False
            break
        band_incomplete = True
    order = np.argsort(energies)
    energies = energies[order]
    cis = [np.asarray(cis[i]) for i in order]
    s2_list = [float(fci.spin_op.spin_square0(ci, norb, nelec)[0])
               for ci in cis]

    vectors = np.stack([ci.ravel() for ci in cis], axis=1)
    gram_error = float(np.linalg.norm(
        vectors.conj().T @ vectors - np.eye(len(cis))))
    if gram_error > 1.0e-6:
        raise AssertionError(f"Thermal roots not orthonormal: {gram_error:.2e}")

    # Variational guard: the final GS is the lowest spin-pure state
    gs_guard = {
        "source": "fix_spin",
        "fix_spin_energy": (float(gs_energy) if math.isfinite(gs_energy)
                            else None),
        "fix_spin_s2": float(gs_s2) if math.isfinite(gs_s2) else None,
        "fix_spin_converged": bool(gs_converged),
        "gs_minus_root0": (float(gs_energy - energies[0])
                           if math.isfinite(gs_energy) else None),
    }
    gs_blocks = None
    candidates = [j for j in range(len(cis))
                  if s2_list[j] < SPIN_PURE_TOL
                  and energies[j] < gs_energy - 1.0e-9]
    if candidates:
        j = candidates[0]
        gs_energy = float(energies[j])
        gs_s2 = s2_list[j]
        gs_ci = cis[j]
        gs_guard["source"] = f"thermal_root_{j}"
    elif not gs_converged or energies[0] < gs_energy - 1.0e-7:
        ci0 = cis[0] + cis[0].T
        ci0 /= np.linalg.norm(ci0)
        redo_energy, redo_ci, redo_s2, redo_converged = _fix_spin_ladder(
            mol, h1_mo, eri_mo, norb, nelec, enuc, ci0=ci0)
        if not (redo_converged and redo_energy <= energies[0] + 1.0e-7):
            raise RuntimeError(
                f"GS variational guard unresolved at R={r_angstrom}: "
                f"fix_spin {gs_energy}, root0 {energies[0]} (S^2 "
                f"{s2_list[0]}), retry {redo_energy} (S^2 {redo_s2})")
        gs_energy, gs_ci, gs_s2 = float(redo_energy), redo_ci, redo_s2
        gs_guard["source"] = "fix_spin_ci0_retry"

    root_blocks = [pyscf_spin_blocks(ci, norb, nelec) for ci in cis]
    if gs_blocks is None:
        if gs_guard["source"].startswith("thermal_root_"):
            gs_blocks = root_blocks[int(gs_guard["source"].rsplit("_", 1)[1])]
        else:
            gs_blocks = pyscf_spin_blocks(gs_ci, norb, nelec)

    weights = clustered_weights(energies)
    mix_blocks = weighted_blocks(root_blocks, weights)

    return {
        "basis": basis,
        "r_angstrom": float(r_angstrom),
        "norb": int(norb),
        "nelec": [int(na), int(nb)],
        "e_scf": float(e_scf),
        "gs_energy": float(gs_energy),
        "gs_s2": float(gs_s2),
        "gs_guard": gs_guard,
        "gs": negativities_from_blocks(gs_blocks, nelec, norb),
        "thermal": negativities_from_blocks(mix_blocks, nelec, norb),
        "thermal_meta": {
            "beta": BETA_HARTREE_INVERSE,
            "nroots": int(len(cis)),
            "band_size_1mEh": n_band,
            "band_incomplete": bool(band_incomplete),
            "energies": [float(e) for e in energies],
            "weights": [float(w) for w in weights],
            "root_s2": s2_list,
            "all_converged": bool(np.all(converged)),
            "gram_error": gram_error,
            "last_root_weight": float(weights[-1]),
        },
    }


# --------------------------------------------------------------------------- #
# Resumable scan driver
# --------------------------------------------------------------------------- #

def run_task(task) -> dict:
    r_angstrom, basis, nroots = task
    started = time.perf_counter()
    try:
        result = solve_point(r_angstrom, basis=basis, nroots=nroots)
    except Exception:
        result = {
            "basis": basis, "r_angstrom": float(r_angstrom),
            "error": traceback.format_exc(limit=4),
        }
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def jsonl_path(stem: str) -> Path:
    return DATA_DIR / f"{stem}.jsonl"


def merge_jsonl(stem: str) -> Path:
    """JSONL (append log) -> JSON (sorted, non-error records preferred)."""
    by_r: dict = {}
    with open(jsonl_path(stem)) as handle:
        for line in handle:
            record = json.loads(line)
            key = round(record["r_angstrom"], 6)
            if key not in by_r or "error" in by_r[key]:
                by_r[key] = record
    records = [by_r[key] for key in sorted(by_r)]
    out = DATA_DIR / f"{stem}.json"
    with open(out, "w") as handle:
        json.dump(records, handle, indent=1)
    n_err = sum("error" in record for record in records)
    print(f"wrote {out} ({len(records)} points, {n_err} errors)", flush=True)
    return out


def run_scan(grid, basis: str, stem: str, *, nroots: int = NROOTS_THERMAL,
             workers: int = 2) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    done = set()
    if jsonl_path(stem).exists():
        with open(jsonl_path(stem)) as handle:
            for line in handle:
                record = json.loads(line)
                if "error" not in record:
                    done.add(round(record["r_angstrom"], 6))
    tasks = [(r, basis, nroots) for r in grid if round(r, 6) not in done]
    tasks.sort(key=lambda task: -task[0])  # heavy (large R) first
    print(f"{stem}: {len(tasks)} tasks ({len(done)} already done), "
          f"{workers} workers", flush=True)
    finished = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            with open(jsonl_path(stem), "a") as handle:
                handle.write(json.dumps(record) + "\n")
            finished += 1
            tag = "ERROR" if "error" in record else "ok"
            print(f"[{finished}/{len(tasks)}] {record['basis']} "
                  f"R={record['r_angstrom']:.2f} {tag} "
                  f"({record['elapsed_seconds']:.1f}s, "
                  f"wall {time.perf_counter() - started:.0f}s)", flush=True)
            if tag == "ERROR":
                print(record["error"], flush=True)
    return merge_jsonl(stem)


# --------------------------------------------------------------------------- #
# STO-3G validation against the exact dense pipeline
# --------------------------------------------------------------------------- #

EXACT_NPZ = "h2o_sto3g_exact.npz"
EXACT_KEYS = ("gs_n2_updown", "gs_n2_upup",
              "thermal_n2_updown", "thermal_n2_upup")


def exact_sto3g_arrays() -> dict:
    """Release dense-pipeline scan (cached to data/h2o_sto3g_exact.npz)."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / EXACT_NPZ
    if cache.exists():
        return dict(np.load(cache))
    sys.path.insert(0, str(MODULE_DIR))
    import h2o_core
    results = h2o_core.run_scan(progress=True)
    arrays = h2o_core.scan_arrays(results)
    payload = {"r": arrays["r"], "e0": arrays["energies"][:, 0]}
    payload.update({key: arrays[key] for key in EXACT_KEYS})
    np.savez(cache, **payload)
    return payload


def sto3g_check(workers: int) -> int:
    """Hard validation gates: engine STO-3G scan vs exact dense pipeline."""
    engine_json = run_scan(R_GRID_STO3G_CHECK, "sto-3g", "h2o_sto3g_engine",
                           workers=workers)
    records = [record for record in json.load(open(engine_json))
               if "error" not in record]
    exact = exact_sto3g_arrays()

    failures = []
    r_engine = np.array([record["r_angstrom"] for record in records])
    if len(records) != len(exact["r"]) or not np.allclose(r_engine, exact["r"]):
        failures.append(f"grid mismatch: engine {len(records)} points, "
                        f"exact {len(exact['r'])}")
    else:
        window = (r_engine >= 2.4) & (r_engine <= 3.85)
        for state, name in (("gs", "n2_updown"), ("gs", "n2_upup"),
                            ("thermal", "n2_updown"), ("thermal", "n2_upup")):
            engine_curve = np.array([record[state][name]
                                     for record in records])
            delta = np.abs(engine_curve - exact[f"{state}_{name}"])
            print(f"{state}_{name}: max|delta| = {delta.max():.3e} "
                  f"(window R in [2.4, 3.85]: {delta[window].max():.3e})")
            if delta.max() > 2.0e-4:
                failures.append(f"{state}_{name} deviates {delta.max():.3e}")
        delta_e = np.abs(np.array([record["gs_energy"]
                                   for record in records]) - exact["e0"])
        print(f"gs_energy: max|delta| = {delta_e.max():.3e}")
        if delta_e.max() > 1.0e-7:
            failures.append(f"gs_energy deviates {delta_e.max():.3e}")

    for record in records:
        meta = record["thermal_meta"]
        if record["gs_energy"] > meta["energies"][0] + 1.0e-7:
            failures.append(f"non-variational GS at R={record['r_angstrom']}")
        if meta["last_root_weight"] > 1.0e-10:
            failures.append(f"thermal truncation at R={record['r_angstrom']}: "
                            f"last root weight {meta['last_root_weight']:.2e}")
        if not meta["all_converged"]:
            failures.append(f"unconverged roots at R={record['r_angstrom']}")
    guarded = [(record["r_angstrom"], record["gs_guard"]["source"])
               for record in records
               if record["gs_guard"]["source"] != "fix_spin"]
    if guarded:
        print(f"variational guard engaged at {len(guarded)} points: "
              f"{[round(r, 2) for r, _ in guarded]}")
    if failures:
        print("FAILED gates:")
        for failure in failures:
            print("  -", failure)
        return 1
    print("sto3g-check: all gates passed")
    return 0


# --------------------------------------------------------------------------- #
# Figure 10: compute + plot pair 
# --------------------------------------------------------------------------- #

def compute_figure10_data(sto3g_arrays: dict, records_631g: list) -> dict:
    records = sorted((record for record in records_631g
                      if "error" not in record),
                     key=lambda record: record["r_angstrom"])
    data = {"r_sto3g": np.asarray(sto3g_arrays["r"]),
            "r_631g": np.array([record["r_angstrom"] for record in records])}
    for prefix in ("gs", "thermal"):
        for name in ("n2_updown", "n2_upup"):
            data[f"sto3g_{prefix}_{name}"] = np.asarray(
                sto3g_arrays[f"{prefix}_{name}"])
            data[f"g631_{prefix}_{name}"] = np.array(
                [record[prefix][name] for record in records])
    return data


def plot_figure10(data: dict, figures_dir=None):
    import matplotlib.pyplot as plt
    import h2o_core

    figure, axes = plt.subplots(2, 1, figsize=(5.6, 7.6), sharex=True)
    for axis, prefix, title in zip(axes, ("gs", "thermal"),
                                   ("Ground state", "Thermal state")):
        axis.plot(data["r_631g"], data[f"g631_{prefix}_n2_upup"],
                  color="tab:red", linestyle="-",
                  label=r"$\mathcal{N}^{(2)}_{\uparrow\!\uparrow}$")
        axis.plot(data["r_631g"], data[f"g631_{prefix}_n2_updown"],
                  color="tab:green", linestyle="-",
                  label=r"$\mathcal{N}^{(2)}_{\uparrow\!\downarrow}$")
        axis.plot(data["r_sto3g"], data[f"sto3g_{prefix}_n2_upup"],
                  color="tab:red", linestyle="-.")
        axis.plot(data["r_sto3g"], data[f"sto3g_{prefix}_n2_updown"],
                  color="tab:green", linestyle="-.")
        axis.set_xlim(*FIG_XLIM)
        axis.set_ylim(0.0, 1.2)
        axis.set_yticks(np.arange(0.0, 1.201, 0.2))
        axis.set_ylabel(r"$\mathcal{N}$")
        axis.set_title(title)
        axis.legend(loc="upper left" if prefix == "gs" else "upper right",
                    framealpha=1.0)
    axes[1].set_xlabel(r"$R(\mathrm{\AA})$")
    figure.tight_layout()
    h2o_core.save_figure(figure, FIGURE_STEM_10,
                         figures_dir or h2o_core.FIGURES_DIR)
    return figure


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scan", action="store_true",
                        help="run the 6-31G scan (resumable)")
    parser.add_argument("--sto3g-check", action="store_true",
                        help="validate the engine against the dense pipeline")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--figure", action="store_true",
                        help="build Fig. 10 from cached data")
    parser.add_argument("--smoke", action="store_true",
                        help="two-point scan (R=1.0, 3.0) instead of the grid")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=None,                                                                                                                                     
                        help="OpenMP threads per worker (workers x threads "                                                                                                                     
                             "<= cores); default: keep OMP_NUM_THREADS (1)") 
    args = parser.parse_args()
    if args.threads:                                                 
        os.environ["OMP_NUM_THREADS"] = str(args.threads)    

    if args.merge_only:
        for stem in ("h2o_631g_scan", "h2o_sto3g_engine"):
            if jsonl_path(stem).exists():
                merge_jsonl(stem)
        return 0
    if args.sto3g_check:
        return sto3g_check(args.workers)
    if args.figure:
        records = json.load(open(DATA_DIR / "h2o_631g_scan.json"))
        data = compute_figure10_data(exact_sto3g_arrays(), records)
        plot_figure10(data)
        print("figure written to", MODULE_DIR / "figures" / FIGURE_STEM_10)
        return 0
    grid = (1.0, 3.0) if args.smoke else R_GRID_631G
    stem = "h2o_631g_smoke" if args.smoke else "h2o_631g_scan"
    out = run_scan(grid, BASIS_631G, stem, workers=args.workers)
    records = json.load(open(out))
    n_err = sum("error" in record for record in records)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
