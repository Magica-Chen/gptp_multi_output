% Example code for multi-output prediction using some simulated data (It is
% not the code for the following paper, there are some sight differences)
%
% You need to modify Omega_init.m, SE_init.m, and nv_init.m to obtain the
% best results for parameter simulation
%
% Written by: Magica Chen 2018/10/28
%     email: sxtpy2010@gmail.com
%
% Reference :
%    [1] Chen, Zexun, Bo Wang, and Alexander N. Gorban. ?Multivariate
%        Gaussian and Student $-t $ Process Regression for Multi-output
%        Prediction.? arXiv preprint arXiv:1703.04455 (2017).
%%
clc
clear
close all
%% Global variable
seeds = 63;
rng(seeds)
% Sample points
N_sample = 100;
N_repeats = 50;
cov_col= @covSEiso;
x = linspace(0,5,N_sample)';
xtr = x;
xte = x;
% If you input is multi-dimensional, you have to choose covSEard, or any
% other ard kernels.
% If you want to obtain better results, please pay attention to the
% initialisation of hyperparameters, e,g, SE_init.m and nu_init(if you use TP)
kernel = @covSEiso; init_func = @SE_init;
%% Example for MV-GPR parameter estimation
phi11 = 1; phi22 = 1;
psi12 = 0.8;
sf = 2.5; ell = 0.5;
noise = 0.01; % noise level
nv = 3;

cov_row = [phi11 psi12;psi12 phi22];
hyp_init = log([ell,sf]);

mGPpredictor = cell(N_repeats,1);
mTPpredictor = cell(N_repeats,1);
yte_gp = cell(N_repeats,1);
yte_tp = cell(N_repeats,1);
% for i = 1: N_repeats
%     y_noise_gp = mv_gptp_sample(cov_col,cov_row,x, hyp_init);
%     % This is parameter estimation example
%     ytr_gp = y_noise_gp + normrnd(0, sqrt(noise), size(y_noise_gp));
%     
%     mGPpredictor{i} = gptp_general(xtr, ytr_gp, xte, noise, kernel, ...
%         init_func, "GPVS");
%     yte_gp{i} = y_noise_gp;
% end

for i = 1: N_repeats
    y_noise_tp = mv_gptp_sample(cov_col,cov_row,x, hyp_init, nv);
    % This is parameter estimation example
    ytr_tp = y_noise_tp + normrnd(0, sqrt(noise), size(y_noise_tp));

    mTPpredictor{i} = gptp_general(xtr, ytr_tp, xte, noise, kernel, ...
        init_func, 'TPVS');
    yte_tp{i} = y_noise_tp;
end

%%
% save("Run10_new", "mGPpredictor", "mTPpredictor", "sn", "nv", "phi11", ...
%     "phi22", "psi12", "ell", "sf")
%% parameter estimation quality

sn_gp = zeros(N_repeats,1);
sf_gp = zeros(N_repeats,1);
ell_gp = zeros(N_repeats,1);
psi12_gp = zeros(N_repeats,1);
phi11_gp = zeros(N_repeats,1);
phi22_gp = zeros(N_repeats,1);

sn_tp = zeros(N_repeats,1);
sf_tp = zeros(N_repeats,1);
ell_tp = zeros(N_repeats,1);
psi12_tp = zeros(N_repeats,1);
phi11_tp = zeros(N_repeats,1);
phi22_tp = zeros(N_repeats,1);
nv_tp = zeros(N_repeats,1);

for i = 1: N_repeats

%     sn_gp(i) = mGPpredictor{i}.hyp.sn;
%     sf_gp(i) = mGPpredictor{i}.hyp.sf;
%     ell_gp(i) = mGPpredictor{i}.hyp.ell;
%     psi12_gp(i) = mGPpredictor{i}.hyp.non_diag_Omega;
%     phi11_gp(i) = mGPpredictor{i}.hyp.diag_Omega(1);
%     phi22_gp(i) = mGPpredictor{i}.hyp.diag_Omega(2);      
    
    sn_tp(i) = mTPpredictor{i}.hyp.sn;
    sf_tp(i) = mTPpredictor{i}.hyp.sf;
    ell_tp(i) = mTPpredictor{i}.hyp.ell;
    psi12_tp(i) = mTPpredictor{i}.hyp.non_diag_Omega;
    phi11_tp(i) = mTPpredictor{i}.hyp.diag_Omega(1);
    phi22_tp(i) = mTPpredictor{i}.hyp.diag_Omega(2);
    nv_tp(i) = mTPpredictor{i}.hyp.nv;
    
end


% pM_GP = [ median(sn_gp), median(sf_gp), ...
%     median(ell_gp),median(phi11_gp), ...
%     median(phi22_gp),median(psi12_gp)]'
% 
% pMAE_GP = [ median(abs(noise - sn_gp))/noise, median(abs(sf - sf_gp))/sf, ...
%     median(abs(ell - ell_gp))/ell,median(abs(phi11 - phi11_gp))/phi11, ...
%     median(abs(phi22 - phi22_gp))/phi22, median(abs(psi12 -psi12_gp))/psi12]'


pM_TP = [ median(sn_tp), median(sf_tp), ...
    median(ell_tp),median(phi11_tp), ...
    median(phi22_tp),median(psi12_tp), ...
    median(nv_tp)]'

pMAE_TP = [ median(abs(noise - sn_tp))/noise, median(abs(sf - sf_tp))/sf, ...
    median(abs(ell - ell_tp))/ell,median(abs(phi11 - phi11_tp))/phi11, ...
    median(abs(phi22 - phi22_tp))/phi22, median(abs(psi12 -psi12_tp))/psi12, ...
    median(abs(nv - nv_tp))/nv]'

