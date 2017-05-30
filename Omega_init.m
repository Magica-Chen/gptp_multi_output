function [diag_Omega,non_diag_Omega] = Omega_init(cov_target)
% Omega_init -- Produce the initial matrix Oemga in a specific method.
%
% Syntax:
%   [diag_Omega,non_diag_Omega] = Omega(d_output)
%
% In: 
%   cov_target    - covariance of targets
%
% Out;
%   diag_Omega      - diagonal element of Omega
%   non_diag_Omega  - non-diagonal element of Omega
%
% Remark: it is important to initialise Omega in an effective way. This is
% an example. If possible, you can make your own Omega_init according to
% the details of problem.
%
% Copyright:  Magica Chen 2017/05/19
%     email:  sxtpy2010@gmail.com
%
% Reference :
%    [1] Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate
%        Gaussian and Student $-t $ Process Regression for Multi-output
%        Prediction." arXiv preprint arXiv:1703.04455 (2017).
%%

d_target = size(cov_target,1);

value_diag = rand(d_target,1).*diag(cov_target);
diag_Omega = exp(value_diag);             % diagonal part

% value_non_diag = unifrnd(-1,1,d_target*(d_target-1)/2,1);   % non-diagonal part
non_diag = [];
for i = 1:d_target
    temp = [non_diag; diag(cov_target,-i)];
    non_diag = temp;
end

value_non_diag = randn(d_target*(d_target-1)/2,1).*non_diag;
non_diag_Omega = exp(value_non_diag);

end

