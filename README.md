# M-body Entanglement and Entropy Analysis of H2O Dissociation

## Overview

This project provides a computational example of applying M-body entanglement and entropy measures to the dissociation of the water molecule (H2O). It utilizes Full Configuration Interaction (FCI) calculations within the STO-3G basis to track how electronic correlation and entanglement evolve as the O-H bonds are symmetrically stretched to the dissociation limit.

The analysis relies on the `openfermion` ecosystem for quantum chemistry calculations and the [`fermionic-mbody`](https://github.com/aguschanchu/fermionic-mbody) library for efficient Reduced Density Matrix (RDM) computations.

## Key Features and Methodology

1.  **FCI Dissociation Curve:** Calculation of the ground state potential energy curve restricted to the $S_z=0$ subspace.
2.  **M-body RDM Generation:** Efficient generation of 1-RDM and 2-RDM tensors, including specific spin blocks (UPUP, UPDOWN) in both antisymmetrized and ordered bases.
3.  **M-body Entropy:** Calculation of the von Neumann entropy of normalized M-RDMs to quantify correlation.
4.  **Fermionic Negativity:** Quantification of entanglement using the Sum of Negative Eigenvalues (SNE) of the partially transposed 2-RDM blocks, handling fermionic statistics.
5.  **Mutual Information:** Analysis of the mutual information between spin-UP and spin-DOWN subsystems.
6.  **Dissociation Limit Analysis:** Detailed investigation of the degenerate ground state manifold at the dissociation limit, identifying the subspace spanned by Slater Determinants versus correlated states by minimizing the 1-RDM purity defect ($N - Tr(\rho_1^2)$).
7.  **Basis Sparsity Optimization (Demonstration):** Example of optimizing the single-particle basis to maximize the sparsity of the wavefunction coefficient matrix.

## Usage

The analysis is provided as Jupyter Notebook
