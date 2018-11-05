% Example code for multi-output prediction using some simulated data (It is
% not the code for the following paper, there are some sight differences)
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
seeds = 74; %19;
rng(seeds)
% Sample points
N_sample = 100;
N_repeats = 20;
cov_col= @covSEiso;
x = linspace(0,5,N_sample)?;
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
sn = 1; %noise_level
nu = 3;

cov_row = [phi11 psi12;psi12 phi22];
hyp_init = log([ell,sf]);

mGPpredictor = cell(N_repeats,1);
mTPpredictor = cell(N_repeats,1);
yte_gp = cell(N_repeats,1);
yte_tp = cell(N_repeats,1);
for i = 1: N_repeats
    y_noise_gp = mv_gptp_sample(cov_col,cov_row,x, sn, hyp_init);
    y_noise_tp = mv_gptp_sample(cov_col,cov_row,x, sn, hyp_init, nu);
    % This is parameter estimation example
    ytr_gp = y_noise_gp;
    ytr_tp = y_noise_tp;
    
    mGPpredictor{i} = gptp_general(xtr, ytr_gp, xte, kernel, init_func,?GPVS?);
    mTPpredictor{i} = gptp_general(xtr, ytr_tp, xte, kernel, init_func,?TPVS?);
    yte_gp{i} = y_noise_gp;
    yte_tp{i} = y_noise_tp;
end

%%
save(?Run10_new?,?mGPpredictor?, ?mTPpredictor?, ?phi11?, ...
    ?phi22", ?psi12?, ?nu?, ?sf?, ?ell?,?sn?)
%% parameter estimation quality

pMAE_sn_gp = zeros(N_repeats,1);
pMAE_sf_gp = zeros(N_repeats,1);
pMAE_ell_gp = zeros(N_repeats,1);
pMAE_psi12_gp = zeros(N_repeats,1);
pMAE_phi11_gp = zeros(N_repeats,1);
pMAE_phi22_gp = zeros(N_repeats,1);
aMAE_mgp1 = zeros(N_repeats,1);
aMAE_mgp2 = zeros(N_repeats,1);

pMAE_sn_tp = zeros(N_repeats,1);
pMAE_sf_tp = zeros(N_repeats,1);
pMAE_ell_tp = zeros(N_repeats,1);
pMAE_psi12_tp = zeros(N_repeats,1);
pMAE_phi11_tp = zeros(N_repeats,1);
pMAE_phi22_tp = zeros(N_repeats,1);
pMAE_nu_tp = zeros(N_repeats,1);

aMAE_mtp1 = zeros(N_repeats,1);
aMAE_mtp2 = zeros(N_repeats,1);
for i = 1: N_repeats
    pMAE_sn_gp(i) = (mGPpredictor{i}.hyp.sn - sn)/sn;
    pMAE_sf_gp(i) = (mGPpredictor{i}.hyp.sf - sf)/sf;
    pMAE_ell_gp(i) = (mGPpredictor{i}.hyp.ell - ell)/ell;
    pMAE_psi12_gp(i) = (mGPpredictor{i}.hyp.non_diag_Omega - psi12)/psi12;
    pMAE_phi11_gp(i) = (mGPpredictor{i}.hyp.diag_Omega(1) - phi11)/phi11;
    pMAE_phi22_gp(i) = (mGPpredictor{i}.hyp.diag_Omega(2) - phi22)/phi22;
    aMAE_mgp1(i) = mae(mGPpredictor{i}.mean(:,1),yte_gp{i}(:,1));
    aMAE_mgp2(i) = mae(mGPpredictor{i}.mean(:,2),yte_gp{i}(:,2));
    
    
    pMAE_sn_tp(i) = (mTPpredictor{i}.hyp.sn - sn)/sn;
    pMAE_sf_tp(i) = (mTPpredictor{i}.hyp.sf - sf)/sf;
    pMAE_ell_tp(i) = (mTPpredictor{i}.hyp.ell - ell)/ell;
    pMAE_psi12_tp(i) = (mTPpredictor{i}.hyp.non_diag_Omega - psi12)/psi12;
    pMAE_phi11_tp(i) = (mTPpredictor{i}.hyp.diag_Omega(1) - phi11)/phi11;
    pMAE_phi22_tp(i) = (mTPpredictor{i}.hyp.diag_Omega(2) - phi22)/phi22;
    pMAE_nu_tp(i) = (mTPpredictor{i}.hyp.nu - nu)/nu;
    
    aMAE_mtp1(i) = mae(mTPpredictor{i}.mean(:,1),yte_tp{i}(:,1));
    aMAE_mtp2(i) = mae(mTPpredictor{i}.mean(:,2),yte_tp{i}(:,2));
    
end


pMAE_GP = [median(abs(pMAE_sn_gp)), median(abs(pMAE_sf_gp)), ...
    median(abs(pMAE_ell_gp)),median(abs(pMAE_phi11_gp)), ...
    median(abs(pMAE_phi22_gp)),median(abs(pMAE_psi12_gp))]

pMAE_TP = [median(abs(pMAE_sn_tp)), median(abs(pMAE_sf_tp)), ...
    median(abs(pMAE_ell_tp)),median(abs(pMAE_phi11_tp)), ...
    median(abs(pMAE_phi22_tp)),median(abs(pMAE_psi12_tp)), ...
    median(abs(pMAE_nu_tp))]

accuracy= [median(aMAE_mgp1), median(aMAE_mtp1);
    median(aMAE_mgp2), median(aMAE_mtp2)]

