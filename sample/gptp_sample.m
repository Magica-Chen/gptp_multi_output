function [varargout] = gptp_sample(cov_x,x,hyp,nv)
% This function is to generate a sample from GP(0,C) or genearte a sample
% from TP(nv,0,C), where C is the selected kernel and nv is the degree of
% freedom.
%   Formula
%     GP sample:
%     If gn \sim N(0,1), C is the kernel GP,which can be written as the SVD
%     decompostion, namely, C = u*s*v', thus, z_gp = u*sqrt(s)*gn is a
%     sample of GP(0,C), since E(z_gp)=0, Cov(z_gp,z_gp')=
%     (u*sqrt(s))(u*sqrt(s)' = C;
%
%     TP sample:
%     If tn \sim t(nv),which the mean of t distribution is 0 and the
%     variance is nv/(nv-2), C is the kernel GP,which can be written as the
%     SVD decompostion, namely, C = u*s*v', thus,
%     z_tp = (nv/(nv-2))^(-1/2)*u*sqrt(s)*tn is a sample of TP(nv,0,C),
%     since E(z_tp)=0; Cov(z_tp,z_tp') = (nv/(nv-2))^(-0.5))*(U*sqrt(s)*
%                                        (nv/(nv-2))*((nv/(nv-2))^(-0.5)*U*sqrt(s))'
%                                      = (U*sqrt(s))(U*sqrt(s))'=C
%   Usage:
%     disp('Usage: [gp_sample] = gptp_sample(cov_x,x,hyp);')
%     disp('   or: [tp_sample] = gptp_sample(cov_x,x,hyp,nv);')
%     disp('   or: [gp_sample tp_sample] = gptp_sample(cov,x,hyp,nv);')
%
%   Input:
%     cov_x: the kernel or covariance function of x
%     x    : the points which is used to evaluate the cov
%     hyp  : the enough hypers, which is used to evaluate the cov
%     nv   : the degree of freedom of student t distribution
%
%   Output:
%     gp_sample and tp_sample
%
%   Copyright Magica Chen, 2018/10/28
%      email: sxtpy2010@gmail.com

if nargin < 3 || nargin >4
    disp('Usage: [gp_sample] = gptp_sample(cov_x,x,hyp);')
    disp('   or: [tp_sample] = gptp_sample(cov_x,x,hyp,nv);')
    disp('   or: [gp_sample tp_sample] = gptp_sample(cov,x,hyp,nv);')
    return
end

n = size(x,1);
C = feval(cov_x, hyp, x);
[u,s,~] = svd(C);  %SVD decomposition, C=usv'

if nargin == 3
    gn = randn(n,1);  % Genearate a sample from standard normal distribution
    z_gp = u*sqrt(s)*gn;
    varargout = {z_gp};
end

if nargin == 4
    tn = trnd(nv,n,1); % Genearate a sample from student t distribution with
                       % degree of freedom is nv.
                       % The mean is 0 and the variance is nv/(nv-2)
    z_tp = (nv/(nv-2))^(-0.5).*u*sqrt(s)*tn;
    
    if nargout < 2
        varargout = {z_tp};
    else
        gn = randn(n,1);  % Genearate a sample from standard normal distribution
        z_gp = u*sqrt(s)*gn;
        varargout = {z_gp,z_tp};
    end    
end

