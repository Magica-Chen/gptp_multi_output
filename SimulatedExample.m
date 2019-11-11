% Example code for multi-output prediction using some simulated data (It is
% not the code for the following paper, there are some sight differences)
%
% Copyright: Magica Chen 2019/07/16
%     email: sxtpy2010@gmail.com
%
% Reference :
%    [1] Chen, Zexun, Bo Wang, and Alexander N. Gorban. "Multivariate
%        Gaussian and Student $-t $ Process Regression for Multi-output
%        Prediction." arXiv preprint arXiv:1703.04455 (2017).
%%
clc
clear
close all
%% Global variable
seeds = 17; rng(seeds);  gpORtp = 'GP'; % for GP noise
% seeds = 189;  rng(seeds); gpORtp = 'GP'; % for GP noise
% seeds = 31;  rng(seeds); gpORtp = 'TP'; % for TP noise

% Sample points
N_sample = 100;
train_series = [1:3:floor(0.45*N_sample)  ...
    floor(0.65*N_sample):3:N_sample]; % split the training and test

cov_row = [1 0.25; 0.25 1];
hyp_init = log([1.001,5]); 
nu =3; % only for t process

%%  Generate noise
% if you would like to achieve the same result for any run of this code
% please set this random seed

cov_col= @covSEiso;
x = linspace(-10,10,N_sample)';

[y_noise_gp,y_noise_tp] = mv_gptp_sample(cov_col,cov_row,x,...
    hyp_init,nu);

% Choose G-noise or T-noise
switch gpORtp
    case 'GP'
        y_noise = y_noise_gp;
    case 'TP'
        y_noise = y_noise_tp;
end
%%  Generate samples
y1 = 2*cos(x).* (x) ;           
y2 = 1.5.*cos(x +pi/5).*(x);      

xtr = x(train_series);
xte = x;

y = [y1 y2] + y_noise;

ytr = y(train_series,:);
yte = [y1 y2];

%% regression/prediction
% If you input is multi-dimensional, you have to choose covSEard, or any
% other ard kernels.
% If you want to obtain better results, please pay attention to the
% initialisation of hyperparameters, e,g, SE_init.m and nu_init(if you use TP)
kernel = @covSEiso; init_func = @SE_init;

[mGPpredictor, mTPpredictor, GPpredictor, TPpredictor] = gptp_general(...
    xtr, ytr, xte, 0.1, kernel, init_func, 'All');

%%  error measure

% multi-output regression
RMSE_mgp1 = sqrt(mse(mGPpredictor.mean(:,1),yte(:,1)));
RMSE_mgp2 = sqrt(mse(mGPpredictor.mean(:,2),yte(:,2)));

RMSE_mtp1 = sqrt(mse(mTPpredictor.mean(:,1),yte(:,1)));
RMSE_mtp2 = sqrt(mse(mTPpredictor.mean(:,2),yte(:,2)));

% mME = [RMSE_mgp1 ;
%        RMSE_mtp1 ];
% ME = [RMSE_mgp2 ;
%       RMSE_mtp2];

% independent output
RMSE_gp1 = sqrt(mse(GPpredictor{1}.mean,yte(:,1)));
RMSE_gp2 = sqrt(mse(GPpredictor{2}.mean,yte(:,2)));

RMSE_tp1 = sqrt(mse(TPpredictor{1}.mean,yte(:,1)));
RMSE_tp2 = sqrt(mse(TPpredictor{2}.mean,yte(:,2)));

% mMEindep = [RMSE_gp1 ;
%             RMSE_tp1];
% MEindep = [RMSE_gp2 ;
%            RMSE_tp2];

error_measure = [RMSE_mgp1, RMSE_mtp1, RMSE_gp1, RMSE_tp1;
    RMSE_mgp2, RMSE_mtp2, RMSE_gp2, RMSE_tp2];
%%
figure(1);
bar(error_measure(:,1:4),'grouped');
legend('MV-GP','MV-TP','GP','TP','Location','northwest')
set(gca,'XTick',1:1:2)
set(gca,'XtickLabel',{'y_1','y_2'})
title('RMSE')

