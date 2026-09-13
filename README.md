# Neural Quantum States (NQS) & VMC for Beryllium Atom

Variational Monte Carlo (VMC) implementation using JAX/Flax to investigate ground-state properties and electronic correlation in the Beryllium atom ($Be$, $Z=4$).

## Key Features
* **Framework:** High-performance vectorized sampling and automatic differentiation (AD) via JAX/Flax.
* **Optimization:** Stochastic Reconfiguration (SR) / Natural Gradient optimization using the Fisher Information Matrix.
* **Wavefunction Ansatz:** Shared-determinant architecture paired with a Neural Jastrow factor.
* **Cusp Conditions:** Exact analytical enforcement of Kato cusp conditions using the non-linear coordinate transformation $t(r) = \frac{r^2}{1+r}$.
* **Static Correlation:** Multi-determinant Complete Active Space (CAS) extension.

## Repository Structure
* `Be_nqs_f.py`: Main NQS/VMC training and optimization pipeline.
* `Be_comparison.py`: Comparative benchmarks between standard analytical Jastrow, neural Jastrow, and CAS models.
* `*.png` / `*.pdf`: Plots showing local energy convergence, Jastrow profiles, and nodal surface analysis.
* `*.npz`: Raw numerical outputs and energy trajectories.

## Requirements
```bash
pip install jax jaxlib flax numpy matplotlib scipy