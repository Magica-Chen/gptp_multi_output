function [varargout] = gptp_general(xtr,ytr,xte,sn,covfunc,para_init,option)

% GPTP_GENERAL -- Solve GP/TP regression for both single output and
% multiple a output problem.
%
% Syntax:
%   [...] = gptp_general(xtr,ytr,xte,cov,option)
%
% In :
%   xtr     - Training inputs
%   ytr     - Training outputs
%   xte     - Test inputs
%   cov     - Covariance function (Using the covariance functions in gpml toolbox)
% para_init - initialisation of hyperparameters for the specific kernel
%   option  - 'GP','TP','GPVS','TPVS',and 'All'
%   sn      - noise level
% Out :
%
%   GPpredictor    - GP predictor, including predicted mean,
%   variance, covariance, lower bound, upper bound, nagetive log marginal
%   likelihood, and hyperparameters in the kernel.
%
%   TPpredictor    - TP predictor, including predicted mean,
%   variance, covariance, lower bound, upper bound, nagetive log marginal
%   likelihood, the degree of freedom, and hyperparameters in the kernel.
%
%   indGPpredictor - (independent) GP predictor, including predicted mean,
%   variance, covariance, lower bound, upper bound, nagetive log marginal
%   likelihood, the degree of freedom, and hyperparameters in the kernel.
%
%   indTPpredictor - (independent) TP predictor, including predicted mean,
%   variance, covariance, lower bound, upper bound, nagetive log marginal
%   likelihood, the degree of freedom, and hyperparameters in the kernel.
%
%   mvGPpredictor - MV-GP predictor, including predicted mean,
%   variance, covariance, lower bound, upper bound, nagetive log marginal
%   likelihood, the degree of freedom, and hyperparameters in the kernel.
%
%   mvTPpredictor - MV-TP predictor, including predicted mean,
%   variance, covariance, lower bound, upper bound, nagetive log marginal
%   likelihood, the degree of freedom, and hyperparameters in the kernel.
%
% Usage:
%       GPpredictor =gptp_general(xtr,ytr,xte,sn,cov,para_init,"GP");
%   or: TPpredictor =gptp_general(xtr,ytr,xte,sn,cov,para_init,"TP");
%   r: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte);
%   or: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte,sn);
%   or: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte,sn,cov);
%   or: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte,sn,cov,para_init);
% % or: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte,sn,cov,
%                                                    para_init,"VS",);')
%   or: [indGPpredictor, mvGPpredictor] =gptp_general(xtr,ytr,xte,sn,cov,
%                                                    para_init,"GPVS");
%   or: [mvGPpredictor] =gptp_general(xtr,ytr,xte,sb,cov,para_init,"GPVS");
%   or: [indTPpredictor, mvTPpredictor] =gptp_general(xtr,ytr,xte,sn,cov,
%                                                    para_init,"TPVS");
%   or: [mvTPpredictor] =gptp_general(xtr,ytr,xte,cov, sn, para_init,"TPVS");
%   or: [mvGPpredictor, mvTPpredictor, indGPpredictor,indTPpredictor] =
%           gptp_general(xtr,ytr,xte,sn,cov,para_init,"All");
%
% Copyright: Magica Chen 2018/10/28
%     Modified on 2019/07/16
%     email: sxtpy2010@gmail.com
%
% Reference :
%    [1] Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate
%        Gaussian and Student $-t $ Process Regression for Multi-output
%        Prediction." arXiv preprint arXiv:1703.04455 (2017).

%% Variables check
if nargin < 3 || nargin > 7
    disp('Usage: GPpredictor =gptp_general(xtr,ytr,xte,sn,cov,para_init,"GP");')
    disp('   or: TPpredictor =gptp_general(xtr,ytr,xte,sn,cov,para_init,"TP");')
    disp('   or: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte);')
    disp('   or: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte,sn);')
    disp('   or: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte,sn,cov);')
    disp(['   or: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte,',...
        'sn,cov,para_init);'])
    %    disp('   or: [GPpredictor, TPpredictor] =gptp_general(xtr,ytr,xte,cov,para_init,"VS");')
    disp(['   or: [indGPpredictor, mvGPpredictor] =gptp_general(xtr,',...
        'ytr,xte,sn,cov,para_init,"GPVS");'])
    disp(['   or: [indTPpredictor, mvTPpredictor] =gptp_general(xtr,',...
        'ytr,xte,sn,cov,para_init,"TPVS");'])
    disp(['   or: [mvGPpredictor, mvTPpredictor, indGPpredictor,',...
        'indTPpredictor] =gptp_general(xtr,ytr,xte,sn,cov,para_init,"All");'])
    return
end

[n_input, d_input] = size(xtr); % length and dimension of training input
d_target =size(ytr,2); % dimension of target

if n_input < 4
    error('Training points are too less')
end

if d_input~= size(xte,2)
    error('The dimension of inputs (training ans test) must be the same')
end

%% default input
if nargin ==3         % if there is no specific kernel, we use SE as default
    sn = 0.1;         % default noise level
    if d_input ==1
        covfunc = @covSEiso;  % SEiso for 1-d training input
    else
        covfunc = @covSEard;  % SEard for multi-d training input
    end
    para_init = @SE_init; % default initialisation method
end

if nargin ==4
    if d_input ==1
        covfunc = @covSEiso;  % SEiso for 1-d training input
    else
        covfunc = @covSEard;  % SEard for multi-d training input
    end
    para_init = @SE_init; % SEard for multi-d training input
end

if nargin ==5
    para_init = @SE_init; % SEard for multi-d training input
end

n_parameter = para_init(xtr, ytr, 1);
dk = cell(n_parameter,1);
k = @(hyp,x,z) covfunc(hyp,x,z);       % Re-define the form of
for i = 1:n_parameter
    dk{i} = @(hyp,x,z) covfunc(hyp,x,z,i);
end

%-------------------------Matlab version is < 2017-------------------------
optim_new = optimset('GradObj','on','display','off','MaxIter',500);
%-------------------------Matlab version >=2017----------------------------
% optim = optimset('GradObj','On','Algorithm','trust-region','display',...
%     'off');
% optim_new = optimset('GradObj','On','Algorithm','trust-region','display',...
%     'off','MaxIter',250);

%% Default function includes both GP and TP (or MV-GP and MV-TP)
if nargin < 7
    [GPpredictor, TPpredictor] = gptp_part(xtr, ytr, xte,k,dk,...
        para_init, n_parameter,optim_new, sn);
    varargout = {GPpredictor, TPpredictor};
end
%% According to selected option
if nargin == 7
    switch option
        case 'GP'
            % only perform GPR
            GPpredictor = gp_part(xtr, ytr, xte,k,dk, para_init, ...
                n_parameter,optim_new, sn);
            varargout = {GPpredictor};
            
        case 'TP'
            % only perform TPR
            TPpredictor = tp_part(xtr, ytr, xte, k,dk,para_init, ...
                n_parameter,optim_new, sn);
            varargout = {TPpredictor};
            
            %         case 'VS'
            %             % Model comparison, if the dimension of output is one, this is a
            %             % comparison of GPR and TPR, otherwise this is a comparison of
            %             % MV-GPR and MV-TPR
            %             [GPpredictor, TPpredictor] = gptp_part(xtr, ytr, xte,n_input,...
            %                 d_input,d_target,k,dk,para_init,nv_init,n_parameter,optim_new, sn);
            %             varargout = {GPpredictor, TPpredictor};
            
        case 'GPVS'
            if d_target > 1
                mvGPpredictor = gp_part(xtr, ytr, xte, k,dk,para_init, ...
                    n_parameter,optim_new, sn);
                if nargout == 1
                    varargout = {mvGPpredictor};
                else
                    indGPpredictor = cell(1,d_target); % allocate space
                    
                    for i = 1:d_target
                        indGPpredictor{i} = gp_part(xtr, ytr(:,i),xte,...
                            d_input,1,k,dk,para_init, n_parameter,...
                            optim_new, sn);
                    end
                    varargout = {indGPpredictor, mvGPpredictor};
                end
            else
                error('The dimension of output must be more than 2')
            end
            
        case 'TPVS'
            if d_target > 1
                mvTPpredictor = tp_part(xtr, ytr, xte,k,dk,para_init,...
                    n_parameter,optim_new, sn);
                if nargout == 1
                    varargout = {mvTPpredictor};
                else
                    indTPpredictor = cell(1,d_target); % allocate space
                    
                    for i = 1:d_target
                        indTPpredictor{i} = tp_part(xtr,ytr(:,i),xte,...
                            n_input,d_input,1,k,dk,para_init,n_parameter,...
                            optim_new, sn);
                    end
                    varargout = {indTPpredictor, mvTPpredictor};
                end
            else
                error('The dimension of output must be more than 2')
            end
            
        case 'All'
            % Model comparison in details. if the dimension of output
            % one, this is the only comparison of GPR and TPR, otherwise
            % this is a comparison of independent GPR, indepedent TPR,
            % MV-GPR and MV-TPR.
            if d_target > 1
                [mvGPpredictor, mvTPpredictor] = gptp_part(xtr, ytr, xte, ...
                    k,dk,para_init,n_parameter, optim_new, sn);
                if nargout == 2
                    varargout = {mvGPpredictor, mvTPpredictor};
                else
                    indGPpredictor = cell(1,d_target); % allocate space
                    indTPpredictor = cell(1,d_target);
                    for i = 1:d_target
                        [indGPpredictor{i}, indTPpredictor{i}] = gptp_part(...
                            xtr,ytr(:,i),xte,k,dk, para_init,n_parameter,...
                            optim_new, sn);
                    end
                    varargout = {mvGPpredictor, mvTPpredictor,indGPpredictor,...
                        indTPpredictor};
                end
            else
                [GPpredictor, TPpredictor] = gptp_part(xtr, ytr, xte, ...
                    k,dk,para_init,n_parameter, optim_new, sn);
                varargout = {GPpredictor, TPpredictor};
            end
    end
end
end

%% implement GPR/MV-GPR only
function GPpredictor = gp_part(xtr, ytr, xte, k, dk, para_init, ...
    n_parameter,opts_new, sn)
%% Global part
% [n_input, d_input] = size(xtr); % length and dimension of training input
d_target =size(ytr,2); % dimension of target
opts = optimset('GradObj','on','display','off');
nlml_gp= Inf;
numinit = 10;

if d_target ==1                            % 1 output regression
    funcGP = @gp_solve_gpml;
    for j=1:numinit
        kernel_gp = para_init(xtr, ytr);
        param_gp = log([sn; kernel_gp]');

        % Optimization
        [~,nlml_gp_new] = fminunc(@(w) funcGP(w,xtr,ytr,k,dk), ...
            param_gp,opts);
        
        if (nlml_gp_new < nlml_gp)
            param_gp_final = param_gp ;
            nlml_gp = nlml_gp_new;
        end
    end
    
else
    funcGP = @mvgp_solve_gpml;
    for j=1:numinit                 % multiple output regression
        kernel_gp = para_init(xtr, ytr);
        %------------------------------------------------------------------
        [diag_Omega_gp,non_diag_Omega_gp] = Omega_init(xtr, ytr);
        %------------------------------------------------------------------
        % param_gp = log([sn; kernel_gp;diag_Omega_gp;...
        %    non_diag_Omega_gp]);
        
        % For parameter estimiation experiements only
        param_gp = log([sn*(1 + 0.1*rand); kernel_gp;diag_Omega_gp;...
            non_diag_Omega_gp]);
        %------------------------------------------------------------------
        % Optimization
        [~,nlml_gp_new] = fminunc(@(w) funcGP(w,xtr,ytr,k,dk), ...
            param_gp,opts);
        
        if (nlml_gp_new < nlml_gp)
            param_gp_final = param_gp;
            nlml_gp = nlml_gp_new;
        end
    end
end

[w_gp_final,nlml_gp_final] = fminunc(@(w) funcGP(w,xtr,ytr,k,dk),...
    param_gp_final,...
    opts_new);
[m_gp,s2_gp,cov_gp,lb_gp,ub_gp] = funcGP(w_gp_final,xtr,ytr,k,xte);

%build a struct class variable
GPpredictor.mean = m_gp;
GPpredictor.variance = s2_gp;
GPpredictor.covariance = cov_gp;
GPpredictor.lb = lb_gp;
GPpredictor.ub = ub_gp;
GPpredictor.nlml = nlml_gp_final;

GPpredictor.hyp.sn = exp(w_gp_final(1));
GPpredictor.hyp.ell = exp(w_gp_final(2:n_parameter));
GPpredictor.hyp.sf = exp(w_gp_final(n_parameter+1));

if d_target>1
    GPpredictor.hyp.diag_Omega = exp(w_gp_final(n_parameter...
        +1+1 : n_parameter+1+d_target));
    GPpredictor.hyp.non_diag_Omega = w_gp_final(...
        n_parameter+1 +d_target+1:end);
end
end

%% implement TPR/MV-TPR only
function TPpredictor = tp_part(xtr, ytr, xte, k,dk,para_init,...
    n_parameter,opts_new,sn)
%% Global part
% [n_input, d_input] = size(xtr); % length and dimension of training input
d_target =size(ytr,2); % dimension of target
opts = optimset('GradObj','on','display','off');
nlml_tp= Inf;
numinit = 10;
%%
if d_target ==1
    funcTP = @tp_solve_gpml;
    for j=1:numinit
        kernel_tp = para_init(xtr, ytr);
        nv = nv_init(xtr, ytr);
        param_tp = log([nv;sn;kernel_tp]');
        % Optimization
        [~,nlml_tp_new] = fminunc(@(w) funcTP(w,xtr,ytr,k,dk), ...
            param_tp,opts);
        
        if (nlml_tp_new < nlml_tp)
            param_tp_final = param_tp;
            nlml_tp = nlml_tp_new;
        end
    end
else
    
    funcTP = @mvtp_solve_gpml;
    for j=1:numinit
        kernel_tp = para_init(xtr, ytr);
        nv = nv_init(xtr, ytr);
        %------------------------------------------------------------------
        [diag_Omega_tp,non_diag_Omega_tp] = Omega_init(xtr, ytr);
        %------------------------------------------------------------------
%         param_tp = log([nv;sn; kernel_tp;diag_Omega_tp;...
%             non_diag_Omega_tp]);
                
        % For parameter estimiation experiements only
        param_tp = log([nv;
            sn*(1 + 0.1*rand);
            kernel_tp;
            diag_Omega_tp;
            non_diag_Omega_tp]);
        %------------------------------------------------------------------
        % Optimization
        [~,nlml_tp_new] = fminunc(@(w) funcTP(w,xtr,ytr,k,dk), ...
            param_tp,opts);
        
        if (nlml_tp_new < nlml_tp)
            param_tp_final = param_tp;
            nlml_tp = nlml_tp_new;
        end
    end
end

[w_tp_final,nlml_tp_final] = fminunc(@(w) funcTP(w,xtr,ytr,...
    k,dk),param_tp_final,opts_new);
[m_tp,s2_tp,cov_tp,lb_tp,ub_tp] = funcTP(w_tp_final,xtr,ytr,k,xte);

% build a struct class variable
TPpredictor.mean = m_tp;
TPpredictor.variance = s2_tp;
TPpredictor.covariance = cov_tp;
TPpredictor.lb = lb_tp;
TPpredictor.ub = ub_tp;
TPpredictor.nlml = nlml_tp_final;

TPpredictor.hyp.nv = exp(w_tp_final(1));
TPpredictor.hyp.sn = exp(w_tp_final(2));
TPpredictor.hyp.ell = exp(w_tp_final(3:n_parameter+1));
TPpredictor.hyp.sf = exp(w_tp_final(n_parameter+2));

if d_target>1
    TPpredictor.hyp.diag_Omega = exp(w_tp_final(n_parameter+2+1:...
        n_parameter+...
        2+d_target));
    TPpredictor.hyp.non_diag_Omega = w_tp_final(n_parameter+2 +...
        d_target+1:end);
end

end

%% implement both GPR/MV-GPR and TPR/MV-TPR
function [GPpredictor, TPpredictor] = gptp_part(xtr, ytr, xte, k, dk, ...
    para_init,n_parameter,opts_new,sn)
% [n_input, d_input] = size(xtr); % length and dimension of training input
d_target =size(ytr,2); % dimension of target
opts = optimset('GradObj','on','display','off');
nlml_gp = Inf; nlml_tp= Inf;
numinit = 10;

% Select the model solve function based on the dimension of targets
if d_target ==1                        % single output GP/TP regression
    funcGP = @gp_solve_gpml; funcTP = @tp_solve_gpml;
    for j=1:numinit
        %% GP part
        kernel_gp = para_init(xtr, ytr);
        param_gp = log([sn; kernel_gp]');
        
        % Optimization
        [~,nlml_gp_new] = fminunc(@(w) funcGP(w,xtr,ytr,k,dk), ...
            param_gp,opts);
        
        if (nlml_gp_new < nlml_gp)
            param_gp_final = param_gp ;
            nlml_gp = nlml_gp_new;
        end
        
        %% TP part
        kernel_tp = para_init(xtr, ytr);
        nv = nv_init(xtr, ytr);
        param_tp = log([nv;sn;kernel_tp]');
        % Optimization
        [~,nlml_tp_new] = fminunc(@(w) funcTP(w,xtr,ytr,k,dk), ...
            param_tp,opts);
        
        if (nlml_tp_new < nlml_tp)
            param_tp_final = param_tp;
            nlml_tp = nlml_tp_new;
        end
    end
    
else                                    % multiple output regression
    
    funcGP = @mvgp_solve_gpml; funcTP = @mvtp_solve_gpml;
    for j=1:numinit
        %% MV-GP part
        kernel_gp = para_init(xtr, ytr);
        %----------------------------------------------------------
        [diag_Omega_gp,non_diag_Omega_gp] = Omega_init(xtr, ytr);
        %----------------------------------------------------------
        param_gp = log([sn; kernel_gp;diag_Omega_gp;...
            non_diag_Omega_gp]);
        %----------------------------------------------------------
        % Optimization
        [~,nlml_gp_new] = fminunc(@(w) funcGP(w,xtr,ytr,k,dk), ...
            param_gp,opts);
        
        if (nlml_gp_new < nlml_gp)
            param_gp_final = param_gp;
            nlml_gp = nlml_gp_new;
        end
        
        %% MV-TP part
        kernel_tp = para_init(xtr, ytr);
        nv = nv_init(xtr, ytr);
        %------------------------------------------------------------------
        [diag_Omega_tp,non_diag_Omega_tp] = Omega_init(xtr, ytr);
        %------------------------------------------------------------------
        param_tp = log([nv;sn; kernel_tp;diag_Omega_tp;...
            non_diag_Omega_tp]);
        %------------------------------------------------------------------
        % Optimization
        [~,nlml_tp_new] = fminunc(@(w) funcTP(w,xtr,ytr,k,dk), ...
            param_tp,opts);
        
        if (nlml_tp_new < nlml_tp)
            param_tp_final = param_tp;
            nlml_tp = nlml_tp_new;
        end
    end
end

%% GP/MV-GP part
[w_gp_final,nlml_gp_final] = fminunc(@(w) funcGP(w,xtr,ytr,k,dk),...
    param_gp_final,...
    opts_new);
[m_gp,s2_gp,cov_gp,lb_gp,ub_gp] = funcGP(w_gp_final,xtr,ytr,k,xte);
% build a struct class variable
GPpredictor.mean = m_gp;
GPpredictor.variance = s2_gp;
GPpredictor.covariance = cov_gp;
GPpredictor.lb = lb_gp;
GPpredictor.ub = ub_gp;
GPpredictor.nlml = nlml_gp_final;

GPpredictor.hyp.sn = exp(w_gp_final(1));
GPpredictor.hyp.ell = exp(w_gp_final(2:n_parameter));
GPpredictor.hyp.sf = exp(w_gp_final(n_parameter+1));

if d_target>1
    GPpredictor.hyp.diag_Omega = exp(w_gp_final(n_parameter+1+1 : ...
        n_parameter +1+d_target));
    GPpredictor.hyp.non_diag_Omega = w_gp_final(n_parameter+1 +...
        d_target+1:end);
end

%% TP/MV-TP part
[w_tp_final,nlml_tp_final] = fminunc(@(w) funcTP(w,xtr,ytr,...
    k,dk),param_tp_final,opts_new);
[m_tp,s2_tp,cov_tp,lb_tp,ub_tp] = funcTP(w_tp_final,xtr,ytr,k,xte);

% build a struct class variable
TPpredictor.mean = m_tp;
TPpredictor.variance = s2_tp;
TPpredictor.covariance = cov_tp;
TPpredictor.lb = lb_tp;
TPpredictor.ub = ub_tp;
TPpredictor.nlml = nlml_tp_final;

TPpredictor.hyp.nv = exp(w_tp_final(1));
TPpredictor.hyp.sn = exp(w_tp_final(2));
TPpredictor.hyp.ell = exp(w_tp_final(3:n_parameter+1));
TPpredictor.hyp.sf = exp(w_tp_final(n_parameter+2));

if d_target>1
    TPpredictor.hyp.diag_Omega = exp(w_tp_final(n_parameter+2+1 : ...
        n_parameter+2+d_target));
    TPpredictor.hyp.non_diag_Omega = w_tp_final(n_parameter+2 +...
        d_target+1:end);
end

end

