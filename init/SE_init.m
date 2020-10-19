function hyp = SE_init(xtr,ytr, ~)
% SE_init -- Produce the initial hyperparameters for SE kernel in the most   
%            commonly-used method.
% 
% Syntax:
%   [...] = SE_init(d_input,~)
%
% In
%    xtr   -  training input
%    ytr   -  training target
%
% Out
%      hyp     - If there are two inputs, hyp means the number of hypers.
%                If there is only one input, hyp means the initial hypers 
%
% Remark:  the choice of prior distribution on the initial hyperparameters 
% may play a vital rule in the parameter estimation and the performance of 
% GPR model. You can design a proper initialisation approach for a specific
% kernel based on the details of problem.
%
% Copyright:  Magica Chen 2018/11/15
%     email:  sxtpy2010@gmail.com
%
% Reference :
%    [1] Chen, Zexun, and Bo Wang. "How priors of initial hyperparameters 
%        affect Gaussian process regression models." arXiv preprint 
%        arXiv:1605.07906 (2016).
%%
if nargin >4 || nargin < 2
    error('The number of input must be 2 or 3')
end

d_input = size(xtr,2);
if nargin==3
    hyp = 1 + d_input;        % report the number of hyp
    return 
end              % report number of parameters

% you can produce your own initial parameters using the information of xtr
% and ytr. This is the default method, maybe not good in many cases.

n_parameter = 1 + d_input*1;    % The number of parameter in the kernel
hyp = rand(n_parameter,1);      % This is the most commonly-used method

% For parameter estimiation experiements only
% n_parameter = 1 + d_input*1;    % The number of parameter in the kernel
% hyp = rand(n_parameter,1);      % This is the most commonly-used method
% hyp(2) = 0.38;
% 
% hyp = [0.5 ; 2.5] + [0.1*rand(); rand()];
end

