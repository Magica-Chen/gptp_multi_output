function [varargout] = mvtp_solve_gpml(w,x,y,k,xt)
% mvtp_solve_gpml - Solve TP regression for multi-output prediction
% problem using covariance functions in the gpml toolbox
%
% Syntax:
%   [...] = mvtp_solve_gpml(w,x,y,k,xt)
%
% In:
%   w     - Log-parameters (nv, sigma2, theta, Omega)
%   x     - Training inputs
%   y     - Training outputs
%   k     - Covariance function handle as @(x,theta) ...
%   xt/dk - Test inputs / Covariance function derivatives (optional)
%
% Out (if xt is empty or not supplied):
%
%   e     - Negative log marginal likelihood
%   eg    - ... and its gradient
%
% Out (if xt is not empty):
%
%   Eft   - Predicted mean
%   Varft - Predicted marginal variance
%   Covft - Predicted joint covariance matrix
%   lb    - 95% confidence lower bound
%   ub    - 95% confidence upper bound
%
% Description:
%   Consider the following MV-TP regression [1] problem:
%
%       f ~ MV-TP(0,k(x,x'),nv),
%     y_k = f(x_k),  k=1,2,...,n,
%
%   where k(x,x') = k_theta(x,x') + sigma2*delta(x,x'). The covariance
%   function handle is used by GPML Matlab.
%
% References:
%
%   [1] A. Shah, A. G. Wilson, and Z. Ghahramani. Student-t processes
%       as alternatives to Gaussian processes. In Proceedings of the
%       17th International Conference on Artificial Intelligence and
%       Statistics (AISTATS), volume 33 of JMLR W&CP, pages 877-885, 2014.
%
% Copyright:
%   2014-2015   Arno Solin
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  2017/05/30 Modified by Magica Chen, sxtpy2010@gmail.com
%
% Reference :
%    [3] Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate
%        Gaussian and Student $-t $ Process Regression for Multi-output
%        Prediction." arXiv preprint arXiv:1703.04455 (2017).
%---------------------------------------------------------------------
%% Extract values, some values should do log transformation
n      = size(x,1);   d = size(y,2);
% 1st parameter is nv
para_nu = exp(w(1));

% 2nd parameter is sigma2
para_sigma2 = exp(w(2));

% Last (d-1)*d/2 parameters is the parameters in non-diagonal Omega
para_non_diagonal_Omega = w(end - (d-1)*d/2 +1:end);

% After last (d-1)*d/2 parameters, there are d parameters for diagonal Omega
para_diagonal_Omega = exp(w(end -(d-1)*d/2-d + 1: end - (d-1)*d/2));

% parameter Omega contains diagonal and non-diagonal
para_Omega = [para_diagonal_Omega(:);para_non_diagonal_Omega(:)];

% The remaining parameters are allocated to kernel
para_kernel = exp(w(3 : end -(d-1)*d/2-d));

% Count the number of parameter in Omega and kernel
num_Omega = numel(para_Omega); num_kernel = numel(para_kernel);

% re-construct the correlation matrix Omega(d*d) using the parameter in w
[L_Omega,Omega] =  vec2mat_diag(w(end-num_Omega+1:end),d);
%% Compute kernel and Cholesky
% Evaluate covariance matrix
% r = abs(bsxfun(@minus,x(:),x(:)'));
% K11 = k(r,para_kernel)+para_sigma2*eye(n);
K11 = k(w(3 : end -(d-1)*d/2-d),x,x)+eye(n)*para_sigma2;

% Solve Cholesky factor
[L_kernel,p] = chol(K11,'lower');

% Try again with jitter, if numerical problems
if p>0
    % Add jitter and try again
    jitter = 1e-9*diag(rand(n,1));
    [L_kernel,p] = chol(K11+jitter,'lower');
    
    % Still no luck
    if p>0, varargout={nan,nan*w}; return; end
end

% Evaluate quantities
vv = L_kernel\y;            vv_Omega = L_Omega\y';
alpha = L_kernel'\vv;       alpha_Omega = L_Omega'\vv_Omega;

U = K11 + vv_Omega'*vv_Omega;
tau = para_nu + n -1;

% Cholesky of U
[L_U,q] = chol(U,'lower');

% Try again with jitter, if numerical problems
if q>0
    % Add jitter and try again
    jitter = 1e-9*diag(rand(n,1));
    [L_U,q] = chol(U+jitter,'lower');
    
    % Still no luck
    if q>0, varargout={nan,nan*w}; return; end
end

%% Do prediction
if nargin>4 && (~isempty(xt) && ~iscell(xt))
    
    % Make additional covariance matrices
    %     K22 = k(bsxfun(@minus,xt(:),xt(:)'),para_kernel);
    %     K21 = k(bsxfun(@minus,xt(:),x(:)'),para_kernel);
    K22 = k(w(3 : end -(d-1)*d/2-d),xt,xt);
    K21 = k(w(3 : end -(d-1)*d/2-d),xt,x);
    
    % Solve the mean
    Eft = K21*alpha;
    
    % Solve the variance
    v = L_kernel\K21';
    %----------------------------------------------------------------------
    Varft_total = kron(diag(K22) - sum(v.^2,1)',diag(Omega + vv'*vv))./...
        (para_nu + n -2);
    Varft = reshape(Varft_total,d,size(xt,1))';
    %----------------------------------------------------------------------
    
    % Return
    varargout = {Eft,Varft};
    
    % Solve the full covariance matrix
    if nargout>2
        % Conditional variance is different
        Covft = kron(K22 - v'*v,Omega + vv'*vv)./(para_nu+n-2);
        %Covft = (K22 - v'*v)*(nv+beta1-2)/(nv+n-2);
        varargout = {Eft,Varft,Covft};
    end
    
    % Also return the 95% interval
    if nargout>3
        lb = Eft + tinv(0.025,para_nu+n).*sqrt(Varft);
        ub = Eft + tinv(0.975,para_nu+n).*sqrt(Varft);
        varargout = {Eft,Varft,Covft,lb,ub};
    end
    % Evaluate marginal likelihood
else
    
    % Evaluate negative log marginal likelihood
    e = (tau + d)*sum(log(diag(L_U)))- tau*sum(log(diag(...
        L_kernel))) + n*sum(log(diag(L_Omega))) + MultiGamma(0.5*tau,n,'Log')...
        - MultiGamma(0.5*(tau +d),n,'Log')+ d*n/2*log(pi);
    
    %         e = 1/2*(para_nu+n+d-1)*log(det(U))+ d/2*sum(log(diag(...
    %         L_kernel))) + n/2*sum(log(diag(L_Omega))) + MultiGamma(1/2*(para_nu...
    %         + n -1),n,'Log') - MultiGamma(1/2*(para_nu +d+n-1),n,'Log')+...
    %         d*n/2*log(pi);
    
    % Calculate gradients
    if nargout > 1 && nargin > 4
        
        % Catch the derivatives
        dk = xt;
        
        % Allocate space
        eg = zeros(2+num_Omega+num_kernel,1);
        
        % Inverse K,U,Omega
        invK = L_kernel'\(L_kernel\eye(n));
        invU = L_U'\(L_U\eye(n));
        invOmega = L_Omega'\(L_Omega\eye(d));
        
        % Derivative w.r.t. nv
        eg(1) = sum(log(diag(L_U)))- sum(log(diag(L_kernel))) + ...
            1/2*MultiGamma(1/2*tau,n,'DerLog')- 1/2*MultiGamma(1/2*...
            (tau+d),n,'DerLog');
        
        % Derivative w.r.t. sigma2
        eg(2) = (tau+d)/2*trace(invU) - tau/2*trace(invK);
        % eg(2) = d/2*trace(invK) - 0.5*(para_nu +n+d-1)*trace(H*invU*invK);
        % eg(2) = d/2*trace(invK) - 0.5*(para_nu +n+d-1)*trace(H*(U\invK));
        
        % Derivative w.r.t. theta in the kernel
        for j= 1:num_kernel
%             dK = dk{j}(bsxfun(@minus,x(:),x(:)'),para_kernel);
            dK = dk{j}(w(3 : end -(d-1)*d/2-d),x,x);
            %------------------------------------------------------------------
            eg(j+2) = (tau+d)/2*trace(invU*dK) - tau/2*trace(invK*dK);
        end
        
        % Return derivatives, Account for log-scale
        eg = eg.*exp(w);
        %%
        % The parameters in the Omega(diagonal and non-diagonal)
        dOmega = zeros(d,d);
        for i = 1:d
            for j = 1:i
                E = zeros(d,d);
                if i==j
                    E(i,j) = para_diagonal_Omega(i);
                else
                    E(i,j) = 1;
                end
                %-----------------------------------------------------------
                dOmega(i,j) = -(tau + d)*trace(invU*alpha_Omega'*(E*L_Omega'+L_Omega*E)*...
                    alpha_Omega) + n/2*trace(invOmega*(E*L_Omega'+L_Omega*E));
            end
        end
        
        % reshape dOmega in a vector,eg_Omega
        eg_Omega = cell(d,1);
        for l= 1:d
            eg_Omega{l} = diag(dOmega,1-l);
        end
        
        % cell to matrix
        eg_Omega = cell2mat(eg_Omega);
        
        % throw all the parameter in eg_Omega into eg
        for i = 1 :num_Omega
            eg(i+2+num_kernel) = eg_Omega(i);
        end
        
        %%
        % Return
        varargout = {e,eg};
    else
        varargout = {e};
    end
    
end
