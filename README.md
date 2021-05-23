# gptp_multi_output

![GitHub](https://img.shields.io/github/license/Magica-Chen/gptp_multi_output?style=plastic)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Magica-Chen/gptp_multi_output?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/Magica-Chen/gptp_multi_output?style=plastic)
![GPML](https://img.shields.io/badge/Matlab-GPML--v3.6-orange)
[![DOI](https://zenodo.org/badge/92859352.svg)](https://zenodo.org/badge/latestdoi/92859352)
![GitHub Repo stars](https://img.shields.io/github/stars/Magica-Chen/gptp_multi_output?style=social)
![Twitter](https://img.shields.io/twitter/follow/MagicaChen?style=social)

This toolkit is used to implement multivariate Gaussian process regression (MV-GPR) and multivariate Student-t process regression (MV-TPR). 

## Setup

This toolkit is based on GPML MATLAB Code http://www.gaussianprocess.org/gpml/code/matlab/doc/, version 3.6.

You have to run `startup.m` in GPML 3.6 first, and then run `add_path.m` in this toolkit.

## Code Structure

The main function is` gptp_general.m`, which relies on four functions in `solver` folder.

### solver

These four functions are used to solve GPR, TPR, MV-GPR and MV-TPR, respectively.

* `gp_solve_gpml.m`
* `tp_solve_gpml.m`
* `mvgp_solve_gpml.m`
* `mvtp_solve_gpml.m`

### init

These functions are used to generate initial hyperparameters for the corresponding covariance function.

BE CAREFUL TO these initialisation function: 

**`Omega_init.m`, `SE_init.m`, and `nv_init.m` (in TPR and MV-TPR)**. 

These functions play an important role in the final results, if you would like to obtain considerable results, **PLEASE USE YOUR OWN FUNCTIONS** according to **YOUR OWN EXPERT OPINIONS using training data**.

**Do not forget to replace `SE_init.m` by the corresponding kernel initialisation function if you write a new for yourself.**

### util and cov

`covSEiso.m`, `covSEard.m` and `sq_dist.m` are collected from GPML MATLAB Code. 

`covSEiso.m` and `covSEard.m` are used as default covariance function. More covariance functions can be selected in the GPML Code toolbox. 

`MultiGamma.m` and `vec2mat_diag.m` are two small functions, which are used in the `mvgp_solve_gpml.m` and `mvtp_solve_gpml.m`.

### sample

`gptp_sample.m ` is to generate a sample from GP or TP with specified row and column covariance and zero mean function.

`mv_gptp_sample.m` is to generate a sample from MV-GP or MV-TP with specified row and column covariance and zero mean function.

### example

`SimulatedExample.m` is a simple example for multi-output prediction (MV-GP and MV-TP) compared with independent prediction using GP and TP .

`SimulatedParameter.m` is a simple example for multi-output prediction (MV-GP and MV-TP) parameter estimation .

You need to modify `Omega_init.m`, `SE_init.m`, and `nv_init.m` to obtain the best results for parameter simulation and prediction.

## Note

This code is proof-of-concept, not optimized for speed.

## Reference 

[1] Chen, Zexun, and Bo Wang. "How priors of initial hyperparameters affect Gaussian process regression models." Neurocomputing 275 (2018): 1702-1710.

[2] Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate Gaussian and Student $-t $ Process Regression for Multi-output Prediction." Neural Computing and Applications 32.8 (2020): 3005-3028.

[3] Chen, Zexun, Jun Fan, and Kuo Wang. "Remarks on multivariate Gaussian Process." arXiv preprint arXiv:2010.09830 (2020).

