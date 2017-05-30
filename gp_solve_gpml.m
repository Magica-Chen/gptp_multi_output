function [varargout] = gp_solve_gpml(w,x,y,k,xt)
% GP_SOLVE_GPML - Solve GP regression problem using the covariance
% functions in the gpml toolbox.
%
% All the code based on orginal code gp_solve.
%
% Syntax:
%   [...] = gp_solve_gpml(w,x,y,k,xt)
%
% In:
%   w     - Log-parameters (sigma2, theta)
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
%   Consider the following GP regression [1] problem:
%
%       f ~ GP(0,k(x,x')),
%     y_k = f(x_k),  k=1,2,...,n,
%
%   where k(x,x') = k_theta(x,x') + sigma2*delta(x,x'). The covariance
%   function handle is used by GPML Matlab.
% 
% References:
%
%   [1] C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for 
%       Machine Learning. MIT Press, 2006.
%
% Copyright:
%   2014-2015   Arno Solin
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.
%
% Modified by Magica Chen, 2017/05/30, sxtpy2010@gmail.com

%%

  % Log transformed parameters
  param = exp(w);

  % Extract values
  n      = size(x,1);
  sigma2 = param(1);
  
  % Evaluate covariance matrix
  % K11 = k(bsxfun(@minus,x(:),x(:)'),param(2:end))+eye(n)*sigma2;
  K11 = k(w(2:end),x,x)+eye(n)*sigma2;
  
  % Solve Cholesky factor
  [L,p] = chol(K11,'lower');
  
  % Not psd
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
  
  % Do prediction
  if nargin>4 && (~isempty(xt) && ~iscell(xt))
      
      % Make additional covariance matrices
%       K22 = k(bsxfun(@minus,xt(:),xt(:)'),param(2:end));
%       K21 = k(bsxfun(@minus,xt(:),x(:)'),param(2:end));
      K22 = k(w(2:end),xt,xt);
      K21 = k(w(2:end),xt,x);

      % Solve the mean
      Eft = K21*alpha;
      
      % Solve the variance
      v = L\K21';
      Varft = diag(K22) - sum(v.^2,1)';
      
      % Return
      varargout = {Eft,Varft};
      
      % Solve the full covariance matrix
      if nargout>2
          Covft = K22 - v'*v;
          varargout = {Eft,Varft,Covft};
      end
      
      % Also return the 95% interval
      if nargout>3
          lb = Eft - 1.96*sqrt(Varft);
          ub = Eft + 1.96*sqrt(Varft);
          varargout = {Eft,Varft,Covft,lb,ub};
      end
      
  % Evaluate marginal likelihood
  else
      
      % Solve beta
      beta1 = vv'*vv;
      
      % The negative log marginal likelihood
      e = n/2*log(2*pi) + sum(log(diag(L))) + 1/2*beta1;
      
      % Calculate gradients
      if nargout > 1 && nargin > 4
          
          % Catch the derivatives
          dk = xt;
          
          % Allocate space
          eg = zeros(1,numel(param));
          
          % Derivative w.r.t. sigma2
          invK = L'\(L\eye(n)); 
          eg(1) = 0.5*trace(invK) - 0.5*(alpha'*alpha);
          
          % The rest of the params
          for j=1:(numel(param)-1)
          %    dK = dk{j}(bsxfun(@minus,x(:),x(:)'),param(2:end));
              dK = dk{j}(w(2:end),x,x);
              eg(j+1) = 0.5*sum(invK(:).*dK(:)) - 0.5*alpha'*(dK*alpha);              
          end
          
          % Return derivatives
          eg = eg.*exp(w);
          
          % Return
          varargout = {e,eg};
          
      else
          varargout = {e};
      end
      
  end
  