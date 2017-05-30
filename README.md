# gptp_multi_output
This package is used to implement multivariate Gaussian process regression (MV-GPR) and multivariate Student-t process regression (MV-TPR). NOTE: This code is proof-of-concept, not optimized for speed.

This package is based on GPML Matlab Code http://www.gaussianprocess.org/gpml/code/matlab/doc/

The main function is gptp_general.m.
There are four useful sub-functions, gp_solve_gpml.m, tp_solve_gpml.m, mvgp_solve_gpml.m, mvtp_solve_gpml.m
These four functions are used to solve GPR, TPR, MV-GPR and MV-TPR, respectively.

covSEiso.m, covSEard.m and sq_dist.m are collected from GPML Matlab Code. 

nu_init.m, Omega_init.m, SE_init.m are used to initialise parameter nu (in TPR and MV-TPR), Omega(in MV-GPR and MV-TPR), and hyperparameters in the specific kernel. Actually, the initial hyperparameters play an important role in the performance of regression. If possible, you can write your own initialisation function.  

covSEiso and covSEard are used as default covariance function. More covariance functions can be selected in the GPML Code toolbox. 

Notice: if the covariance function is changed, do not forget to replace SE_init.m by the corresponding kernel initialation function.

MultiGamma.m and vec2mat_diag.m are two small functions, which are used in the mvgp_solve_gpml.m and mvtp_solve_gpml.m.


Reference: 

[1] Chen, Zexun, and Bo Wang. "How priors of initial hyperparameters affect Gaussian process regression models." arXiv preprint arXiv:1605.07906 (2016).

[2] Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate Gaussian and Student $-t $ Process Regression for Multi-output Prediction." arXiv preprint arXiv:1703.04455 (2017).

