function nu = nu_init(n_input)
% nu_init -- Produce the initial nu for TP related models
%
% Syntax:
%   [...] = nu_init(n_input,~)
%
% In
%    n_input   - length of training inputs
%
% Out
%        nu    - initial nu
%
% Remark:  the degree of freedom controls the tails' fatness of
% distribution. It may have important influence on the performance of TP
% related models.
%
%
% Copyright:  Magica Chen 2017/05/19
%     email:  sxtpy2010@gmail.com
%
%%
if nargin >2 || nargin < 1
    error('The number of input must be 1')
end

nu = 2+rand()./n_input;  % initial nu is between 2 and the number of data

end
