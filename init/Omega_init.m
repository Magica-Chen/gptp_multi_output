function [diag_Omega,non_diag_Omega] = Omega_init(xtr, ytr)
% Omega_init -- Produce the initial matrix Oemga in a specific method.
%
% Syntax:
%   [diag_Omega,non_diag_Omega] = Omega(d_output)
%
% In: 
%   xtr    - training inputs
%   ytr    - training outputs
%
% Out;
%   diag_Omega      - diagonal element of Omega
%   non_diag_Omega  - non-diagonal element of Omega
%
% Remark: it is important to initialise Omega in an effective way. This is
% an example. If possible, you can make your own Omega_init according to
% the details of problem.
%
% Copyright:  Magica Chen 2018/11/16
%     email:  sxtpy2010@gmail.com
%
% Reference :
%    [1] Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate
%        Gaussian and Student $-t $ Process Regression for Multi-output
%        Prediction." arXiv preprint arXiv:1703.04455 (2017).
%%

d_target = size(ytr,2);
cov_target = corr(ytr);

value_diag = abs(unifrnd(-0.1, 0.1, d_target,1) + diag(cov_target));
diag_Omega = exp(value_diag);             % diagonal part

non_diag = [];
for i = 1:d_target
    temp = [non_diag; diag(cov_target,-i)];
    non_diag = temp;
end

value_non_diag = randn(d_target*(d_target-1)/2,1) + non_diag;
non_diag_Omega = exp(value_non_diag);

% For parameter estimation experiment only
% value_diag = unifrnd(-0.01, 0.01, d_target,1) + 1*ones(d_target,1);
% diag_Omega = exp(value_diag);             % diagonal part
% 
% value_non_diag = randn(d_target*(d_target-1)/2,1) + log(0.8*ones(...
%     d_target*(d_target-1)/2,1));
% non_diag_Omega = exp(value_non_diag);
%
% diag_Omega = 1 + 0.02* rand(2,1);
% non_diag_Omega = 0.8 + 0.01 * rand();
end

