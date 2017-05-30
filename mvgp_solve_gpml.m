function [varargout] = mvgp_solve_gpml(w,x,y,k,xt)
% mvgp_solve_gpml - Solve GP regression for multi-output prediction
% problem using covariance functions in the gpml toolbox
%
% Syntax:
%   [...] = mvgp_solve_gpml(w,x,y,k,xt)
%
% In:
%   w     - Log-parameters (sigma2, theta, Omega)
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
%   Consider the following MV-GP regression [1] problem:
%
%       f ~ MV-GP(0,k(x,x')),
%     y_k = f(x_k),  k=1,2,...,n,
%
%   where k(x,x') = k_theta(x,x') + sigma2*delta(x,x'). The covariance
%   function handle is used by GPML Matlab.
%
% References:
%
%   [1] C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for
%       Machine Learning. MIT Press, 2006.
%   [2] 2014-2015   Arno Solin
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  2017/05/30 Modified by Magica Chen, sxtpy2010@gmail.com
%
% Reference :
%    [3] Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate
%        Gaussian and Student $-t $ Process Regression for Multi-output
%        Prediction." arXiv preprint arXiv:1703.04455 (2017).
% -------------------------------------------------------------------------
%% Extract values, some values should do log transformation
n      = size(x,1);   d = size(y,2);

% 1st parameter is sigma2
para_sigma2 = exp(w(1));

% Last (d-1)*d/2 parameters is the parameters in non-diagonal Omega
para_non_diagonal_Omega = w(end - (d-1)*d/2 +1:end);

% After last (d-1)*d/2 parameters, there are d parameters for diagonal Omega
para_diagonal_Omega = exp(w(end -(d-1)*d/2-d + 1: end - (d-1)*d/2));

% parameter Omega contains diagonal and non-diagonal
para_Omega = [para_diagonal_Omega(:);para_non_diagonal_Omega(:)];

% The remaining parameters are allocated to kernel
para_kernel = exp(w(2 : end -(d-1)*d/2-d));

% Count the number of parameter in Omega and kernel
num_Omega = numel(para_Omega); num_kernel = numel(para_kernel);

% re-construct the correlation matrix Omega(d*d) using the parameter in w
[L_Omega,Omega] =  vec2mat_diag(w(end-num_Omega+1:end),d);
%--------------------------------------------------------------------------
%%
% Evaluate covariance matrix
% K11 = k(bsxfun(@minus,x(:),x(:)'),para_kernel)+eye(n)*para_sigma2;
K11 = k(w(2 : end -(d-1)*d/2-d),x,x)+eye(n)*para_sigma2;

% Solve Cholesky factor
[L_kernel,p] = chol(K11,'lower');

% Not psd
if p>0
    
    % Add jitter and try again
    jitter = 1e-9*diag(rand(n,1));
    [L_kernel,p] = chol(K11+jitter,'lower');
    
    % Still no luck
    if p>0, varargout={nan,nan*w}; return; end
    
end

%-------------------------------------------------------------------------
% Evaluate quantities
vv = L_kernel\y;            vv_Omega = L_Omega\y';
alpha = L_kernel'\vv;       alpha_Omega = L_Omega'\vv_Omega;

% Do prediction
if nargin>4 && (~isempty(xt) && ~iscell(xt))
    
    % Make additional covariance matrices
    %     K22 = k(bsxfun(@minus,xt(:),xt(:)'),para_kernel);
    %     K21 = k(bsxfun(@minus,xt(:),x(:)'),para_kernel);
    K22 = k(w(2 : end -(d-1)*d/2-d),xt,xt);
    K21 = k(w(2 : end -(d-1)*d/2-d),xt,x);

    % Solve the mean
    Eft = K21*alpha;
    
    % Solve the variance
    v = L_kernel\K21';
    %----------------------------------------------------------------------
    Varft_total = kron(diag(K22) - sum(v.^2,1)',diag(Omega));
    Varft = reshape(Varft_total,d,size(xt,1))';
    %----------------------------------------------------------------------
    
    % Return
    varargout = {Eft,Varft};
    
    % Solve the full covariance matrix
    %----------------------------------------------------------------------
    if nargout>2
        Covft = kron(K22 - v'*v,Omega);
        varargout = {Eft,Varft,Covft};
    end
    %----------------------------------------------------------------------
    
    % Also return the 95% interval
    if nargout>3
        lb = Eft - 1.96.*sqrt(Varft);
        ub = Eft + 1.96.*sqrt(Varft);
        varargout = {Eft,Varft,Covft,lb,ub};
    end
    
    % Evaluate marginal likelihood
else
    %----------------------------------------------------------------------
    % No need to have beta
    % Solve beta
    % beta1 = vv'*vv;
    
    %---------The negative log marginal likelihood---------------------
    % Matrix process is different from this
    e = d*n/2*log(2*pi)+ d*sum(log(diag(L_kernel)))+ n*sum(log(diag(L_Omega)))...
        + trace(0.5*alpha*alpha_Omega);
    
    % Calculate gradients
    if nargout > 1 && nargin > 4
        
        % Catch the derivatives
        dk = xt;
        
        % Allocate space
        eg = zeros(1+num_Omega+num_kernel,1);
        
        %------------------------------------------------------------------
        % Derivative w.r.t. sigma2
        invK = L_kernel'\(L_kernel\eye(n));
        invOmega = L_Omega'\(L_Omega\eye(d));
        eg(1) = d*0.5*trace(invK) -  0.5*trace(alpha*invOmega*alpha');
        
        % The params in the kernel
        for j= 1:num_kernel
%             dK = dk{j}(bsxfun(@minus,x(:),x(:)'),para_kernel);
            dK = dk{j}(w(2 : end -(d-1)*d/2-d),x,x);            
            
            % eg(j+1) = d*0.5*sum(invK(:).*dK(:)) - 0.5*(alpha_Omega'*alpha_Omega)...
            %    *(invK*dK*invK);
            % or
            eg(j+1) = d*0.5*trace(invK*dK) - 0.5*trace(alpha*invOmega*alpha'*dK);
        end
        
        % Return derivatives
        eg = eg.*exp(w);
        
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
                dOmega(i,j) = 0.5*n*trace(invOmega*(E*L_Omega'+L_Omega*E)) - 0.5*trace(...
                    alpha_Omega*invK*alpha_Omega'*(E*L_Omega'+L_Omega*E));
            end
        end
        
        % reshape dOmega in a vector,eg_Omega
        eg_Omega = cell(d,1);
        for l= 1:d
            eg_Omega{l} = diag(dOmega,1-l);
        end
        
        eg_Omega = cell2mat(eg_Omega);
        
        % throw all the parameter in eg_Omega into eg
        for i = 1 :num_Omega
            eg(i+1+num_kernel) = eg_Omega(i);
        end
        % Return
        varargout = {e,eg};
        
    else
        varargout = {e};
    end
    
end
