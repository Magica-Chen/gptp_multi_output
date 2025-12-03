# gptp_multi_output

![GitHub](https://img.shields.io/github/license/Magica-Chen/gptp_multi_output?style=plastic)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Magica-Chen/gptp_multi_output?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/Magica-Chen/gptp_multi_output?style=plastic)
![GPML](https://img.shields.io/badge/Matlab-GPML--v3.6-orange)
[![DOI](https://zenodo.org/badge/92859352.svg)](https://zenodo.org/badge/latestdoi/92859352)
![GitHub Repo stars](https://img.shields.io/github/stars/Magica-Chen/gptp_multi_output?style=social)
![Twitter](https://img.shields.io/twitter/follow/MagicaChen?style=social)

MATLAB toolbox for Gaussian Process (GP), Student-t Process (TP), multivariate GP (MV-GP) and multivariate TP (MV-TP) regression built on top of GPML 3.6.

This code is proof-of-concept, not optimized for speed.

## Requirements
- MATLAB R2016b or maybe newer
- GPML MATLAB Code v3.6 (run `startup.m` from GPML before using this repo)
- Add this repo to the MATLAB path via `add_path.m`

## Quick start
1. Clone GPML 3.6 and run its `startup.m`.
2. Run `add_path.m` in this repo.
3. Call `gptp_general` with your data:
   ```matlab
   % GP regression with defaults (SE kernel, sn = 0.1)
   GP = gptp_general(xtr, ytr, xte);

   % MV-TP regression with custom kernel and inits
   covfunc = @covSEard;
   para_init = @SE_init;
   sn = 0.05;
   mvTP = gptp_general(xtr, ytr, xte, sn, covfunc, para_init, "TPVS");
   ```

## Key entry points
- `gptp_general.m` — high-level driver for GP, TP, MV-GP, MV-TP and their independent variants.
- `solver/` — model-specific solvers that wrap GPML primitives:
  - `gp_solve_gpml.m`, `tp_solve_gpml.m`, `mvgp_solve_gpml.m`, `mvtp_solve_gpml.m`
- `init/` — hyperparameter initialisation helpers. Adapt these to your data for best results:
  - `SE_init.m` (kernel hyperparameters)
  - `Omega_init.m` (output correlation) — provides a starting point only; production use should supply a task-specific initialiser.
  - `nv_init.m` (degrees of freedom for TP/MV-TP)
- `util/` and `cov/` — supporting math utilities and default SE covariance functions from GPML.
- `sample/` and `example/` — small scripts for sampling and simulation demos.

## Notes on initialisation
Performance is sensitive to initial hyperparameters. The default `SE_init.m`, `Omega_init.m`, and `nv_init.m` are generic and meant as placeholders. For serious use, design problem-specific initialisation based on your training data and kernel choice (e.g., scale parameters, sensible output correlations, or informative priors on degrees of freedom).


## Repository layout
- `gptp_general.m` — main API
- `solver/` — optimisation and prediction routines
- `init/` — hyperparameter initialisers
- `util/`, `cov/` — utilities and covariance functions
- `sample/`, `example/` — sampling and usage examples

## References
1. Chen, Zexun, and Bo Wang. "How priors of initial hyperparameters affect Gaussian process regression models." *Neurocomputing* 275 (2018): 1702–1710.
2. Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate Gaussian and Student-t Process Regression for Multi-output Prediction." *Neural Computing and Applications* 32.8 (2020): 3005–3028.
3. Chen, Zexun, Jun Fan, and Kuo Wang. "Remarks on multivariate Gaussian Process." *Neural Computing and Applications* (published version of arXiv:2010.09830), 2023.

