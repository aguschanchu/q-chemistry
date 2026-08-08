# q-chemistry

Companion code for

> J. Garcia, J. A. Cianciulli, and R. Rossignoli,
> *Fermionic entanglement and quantum correlation measures in molecules*,
> [arXiv:2604.07633](https://arxiv.org/abs/2604.07633) (2026).

The repository computes spin-resolved entanglement and quantum-correlation
measures, von Neumann entropies of particle reduced density matrices (RDMs),
global and two-body mutual informations, and fermionic negativities, along
molecular dissociation curves, for exact finite-basis (full-CI) ground states
and for low-temperature Gibbs states projected onto a fixed-$M_S$ sector.

Particle RDMs are normalized such that their traces equal to particle-pair 
counts and all entropies are reported in bits (base-2 logarithms). Each 
molecule ships as a self-contained three-file release: a narrative notebook, 
a molecule-specific physics core, and a compact RDM engine. 

## Repository map

| Path | Contents |
|---|---|
| [`molecules/h2o/`](molecules/h2o/) | H₂O/STO-3G — the manuscript's system. Reproduces Figures 2–8 of the paper from a single scan of the complete 441-dimensional $M_S=0$ sector. A second notebook (`main_631g.ipynb` + `h2o_631g.py`) reproduces the Fig. 10 basis-set comparison from an iterative-FCI 6-31G scan.|
| [`molecules/h2/`](molecules/h2/) | H₂/cc-pVDZ — the same measures applied to the analytically transparent two-electron case. |

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run the releases

Each notebook is a single clean `Run All`: one deterministic scan, then the
figures. Both notebooks can equally be opened and run interactively
(`jupyter notebook main.ipynb`).

$H_2O$/STO-3G (~4–5 min, writes `figures/fig02…fig08`).

$H_2$/cc-pVDZ (~1 min, writes `figures/01…06`).

$H_2O$/6-31G Fig. 10 (`molecules/h2o/main_631g.ipynb`) replots from the
committed scan cache in minutes. Regenerating the cache itself is a many-core job 
(the $M_S=0$ sector holds $\binom{13}{5}^2$ determinants and each
of the 37 geometries takes minutes of Davidson iterations)