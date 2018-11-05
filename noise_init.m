function noise = noise_init(ytr)
% noise_init -- Produce the initial noise level for GP/TP related models
%
% Syntax:
%   [...] = noise_init(n_input,~)
%
% In
%    ytr   - target training
%
% Out
%        noise    - initial noise
%
%
%
% Copyright:  Magica Chen 2018/11/04
%     email:  sxtpy2010@gmail.com
%
%%
if nargin >2 || nargin < 1
    error('The number of input must be 1')
end

noise = rand() * min(min(abs(ytr)));  %noise level can be chosen by user
% noise = 1;
end
