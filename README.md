# gptp_multi_output

![GitHub](https://img.shields.io/github/license/Magica-Chen/gptp_multi_output?style=plastic)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Magica-Chen/gptp_multi_output?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/Magica-Chen/gptp_multi_output?style=plastic)
![GPML](https://img.shields.io/badge/Matlab-GPML--v3.6-orange)

This toolkit is used to implement multivariate Gaussian process regression (MV-GPR) and multivariate Student-t process regression (MV-TPR). 

## Code structure

### Main functions
The main function is gptp_general.m. It can return GPR, TPR, MV-GPR, MV-TPR and their comparisons.
There are four useful sub-functions:
* gp_solve_gpml.m 
* tp_solve_gpml.m，
* mvgp_solve_gpml.m
* mvtp_solve_gpml.m

These four functions are used to solve GPR, TPR, MV-GPR and MV-TPR, respectively.

### Initial hyperparameter functions
* BE CAREFUL TO these initialisation funciton: **nv_init.m, Omega_init.m, SE_init.m, and nu (in TPR and MV-TPR)**. These functions are used to initialise parameters. These functions play an important role in the final results, if you would like to obtain considerable results, **PLEASE USE YOUR OWN FUNCTIONS** according to **YOUR OWN EXPERT OPINIONS using training data**.

### Utility functions
* MultiGamma.m and vec2mat_diag.m are two small functions, which are used in the mvgp_solve_gpml.m and mvtp_solve_gpml.m.
* Omega(in MV-GPR and MV-TPR), and hyperparameters in the specific kernel. 

Becasue the initial hyperparameters play an important role in the performance of regression. 
If possible, you should write your own initialisation function.  

**Do not forget to replace SE_init.m by the corresponding kernel initialation function if you write a new for yourself.**

### Cov and utility function
This tookkit is based on GPML Matlab Code http://www.gaussianprocess.org/gpml/code/matlab/doc/, version 3.6.

covSEiso.m, covSEard.m and sq_dist.m are collected from GPML Matlab Code. 

covSEiso and covSEard are used as default covariance function. More covariance functions can be selected in the GPML Code toolbox. 


## History
### ----2018/01/19-----

File added: SimulatedExample.m

Add a simple example for multi-output prediction (MV-GP and MV-TP) compared with independent prediction using GP and TP

Update optimset setting for Matlab 2017 and later in gptp_general.m 

### ----2017/12/04-----

Add gptp_sample.m file
This file is to generate a sample from GP or TP with specificed row and column covariance and zero mean function.

Add mv_gptp_sample.m file
This file is to generate a sample from MV-GP or MV-TP with specificed row and column covariance and zero mean function.

## Note
This code is proof-of-concept, not optimized for speed.

## Reference 

[1] Chen, Zexun, and Bo Wang. "How priors of initial hyperparameters affect Gaussian process regression models." Neurocomputing 275 (2018): 1702-1710.

[2] Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate Gaussian and Student $-t $ Process Regression for Multi-output Prediction." arXiv preprint arXiv:1703.04455 (2017).

