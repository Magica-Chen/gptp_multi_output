function [varargout] = tp_solve_gpml(w,x,y,k,xt)
% TP_SOLVE - Solve TP regression problem using the covariance
% functions in the gpml toolbox.
%
% All the code based on orginal code gp_solve.
%
% Syntax:
%   [...] = tp_solve_gpml(w,x,y,k,xt)
%
% In:
%   w     - Log-parameters (nu, sigma2, theta)
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
%   Consider the following TP regression [1] problem:
%
%       f ~ TP(0,k(x,x'),nu),
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
% Modified by Magica Chen, 2017/05/30, sxtpy2010@gmail.com
%
%%

  % Log transformed parameters
  param = exp(w);

  % Extract values
  n      = size(x,1);
  nu     = param(1);
  sigma2 = param(2);

  % Evaluate covariance matrix
%   r = abs(bsxfun(@minus,x(:),x(:)'));
%   K11 = k(r,param(3:end))+sigma2*eye(n);
  K11 = k(w(3:end),x,x)+eye(n)*sigma2;
  
  % Solve Cholesky factor
  [L,p] = chol(K11,'lower');
  
  % Try again with jitter, if numerical problems
  if p>0
      
      % Add jitter and try again
      jitter = 1e-9*diag(rand(n,1));
      [L,p] = chol(K11+jitter,'lower');
      
      % Still no luck
      if p>0, varargout={nan,nan*w}; return; end
      
  end
  
  % Evaluate quantities
  vv = L\y(:);
  alpha = L'\vv;
  
  % Solve beta
  beta1 = vv'*vv;
  
  % Do prediction
  if nargin>4 && (~isempty(xt) && ~iscell(xt))
      
      % Make additional covariance matrices
%       K22 = k(bsxfun(@minus,xt(:),xt(:)'),param(3:end));
%       K21 = k(bsxfun(@minus,xt(:),x(:)'),param(3:end));
      K22 = k(w(3:end),xt,xt);
      K21 = k(w(3:end),xt,x);
      
      % Solve the mean
      Eft = K21*alpha;
      
      % Solve the variance
      v = L\K21';
      Varft = (diag(K22) - sum(v.^2,1)')*(nu+beta1-2)/(nu+n-2);
      
      % Return
      varargout = {Eft,Varft};
      
      % Solve the full covariance matrix
      if nargout>2
          Covft = (K22 - v'*v)*(nu+beta1-2)/(nu+n-2);
          varargout = {Eft,Varft,Covft};
      end
      
      % Also return the 95% interval
      if nargout>3
          lb = Eft + tinv(0.025,nu+n)*sqrt(Varft);
          ub = Eft + tinv(0.975,nu+n)*sqrt(Varft);
          varargout = {Eft,Varft,Covft,lb,ub};
      end
      
  % Evaluate marginal likelihood
  else
      
      % Evaluate negative log marginal likelihood
      e = n/2*log((nu-2)*pi) + sum(log(diag(L))) ...
          - gammaln((nu+n)/2) + gammaln(nu/2) ...
          + (nu+n)/2*log(1+beta1/(nu-2));
      
      % Calculate gradients
      if nargout > 1 && nargin > 4
          
          % Catch the derivatives
          dk = xt;
          
          % Allocate space
          eg = zeros(1,numel(param));
          
          % Derivative w.r.t. nu
          eg(1) = -n/(2*(nu-2)) + psi((nu+n)/2)/2 - psi(nu/2)/2 ... 
              - 1/2*log(1 + beta1/(nu-2)) ...
              + beta1*(nu+n)/2/(nu-2)/(beta1+nu-2);
          eg(1) = -eg(1);
          
          % Derivative w.r.t. sigma2
          invK = L'\(L\eye(n));
          eg(2) = 0.5*trace(invK) - 0.5*(nu+n)/(nu-2+beta1)*(alpha'*alpha);
          
          % The rest of the params
          for j=1:(numel(param)-2)
%               dK = dk{j}(r,param(3:end));
              dK = dk{j}(w(3:end),x,x);
              eg(j+2) = 0.5*sum(invK(:).*dK(:)) - 0.5*(nu+n)/(nu-2+beta1)*(alpha'*(dK*alpha));
          end
          
          % Account for log-scale
          eg = eg.*exp(w);
          
          % Return
          varargout = {e,eg};
          
      else
          varargout = {e};
      end
      
  end
  