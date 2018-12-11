% Example code for multi-output prediction using some simulated data (It is
% not the code for the following paper, there are some sight differences)
%
% Copyright: Magica Chen 2018/12/11
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
% Sample points
N_sample = 100;

train_series = [1:3:floor(0.45*N_sample)  ...
    floor(0.65*N_sample):3:N_sample]; % split the training and test
%----------------------------------------------------------------
% Generate noise
seeds =1;
rng(seeds)
% if you would like to achieve the same result for any run of this code
% please set this random seed

cov_col= @covSEard;
x1 = linspace(-10,10,N_sample)';
x2 = linspace(-10,10,N_sample)';
x = [x1 x2];
%%  Generate samples

y1 = 2*cos((x1 + x2)/2).* (x1 + x2)/2 ;           
y2 = 1.5.*cos((x1 + x2)/2 + pi/5).*(x1 + x2)/2;      

xtr = x(train_series,:);
xte = x;

y = [y1 y2] + randn(size([y1 y2]));
ytr = y(train_series,:);
yte = [y1 y2];

%% regression/prediction
% If you input is multi-dimensional, you have to choose covSEard, or any
% other ard kernels.
% If you want to obtain better results, please pay attention to the
% initialisation of hyperparameters, e,g, SE_init.m and nu_init(if you use TP)
kernel = @covSEard; init_func = @SE_init;

[mGPpredictor, mTPpredictor, GPpredictor, TPpredictor] = gptp_general(...
    xtr, ytr, xte, kernel, init_func, 'All');

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
legend('MV-GPR','MV-TPR','GPR','TPR','Location','best')
set(gca,'XTick',1:1:2)
set(gca,'XtickLabel',{'y_1','y_2'})
title('RMSE')
