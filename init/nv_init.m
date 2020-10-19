function nv = nv_init(xtr, ytr)
% nu_init -- Produce the initial nu for TP related models
%
% Syntax:
%   [...] = nu_init(n_input,~)
%
% In
%    xtr   - training inputs
%    ytr   - training outputs
% Out
%        nv    - initial nv
%
% Remark:  the degree of freedom controls the tails' fatness of
% distribution. It may have important influence on the performance of TP
% related models.
%
%
% Copyright:  Magica Chen 2018/11/16
%     email:  sxtpy2010@gmail.com
%
%%
if nargin >2 || nargin < 1
    error('The number of input must be 1')
end

% This is the default method, if you would like to obtain the better
% results, please try your own idea considering the training data.
n_input = size(xtr, 2);
sigma = min(std(ytr, 0, 1));
nv = 2 + unifrnd(0, sigma)/n_input;  % initial nu is between 2 and the number of data

% For parameter estimiation experiements only
% nv = 3 + unifrnd(-0.8,0.8);  % initial nu is between 2 and the number of data
end
