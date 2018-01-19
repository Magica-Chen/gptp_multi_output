% Example code for multi-output prediction using some simulated data (It is
% not the code for the following paper, there are some sight differences)
%
% Copyright: Magica Chen 2018/01/19
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
seeds = 26;  gpORtp = 'TP'; % for TP noise
% seeds = 56; gpORtp = 'GP'; % for GP noise

% Sample points
N_sample = 100;

% train_series = [1:3:floor(0.72*N_sample)  ...
%     floor(0.86*N_sample):3:N_sample]; % split the training and test

train_series = [1:3:floor(0.45*N_sample)  ...
    floor(0.65*N_sample):3:N_sample]; % split the training and test

cov_row = [1 0.25;0.25 1];
hyp_init = log([1.001,5]);
nu =3; % Only for t-disribution

%----------------------------------------------------------------
% Generate noise

rng(seeds)
cov_col= @covSEiso;
noise_level = linspace(0,1,N_sample)';

[y_noise_gp,y_noise_tp] = mv_gptp_sample(cov_col,cov_row,noise_level,...
    hyp_init,nu);

% Choose G-noise or T-noise
switch gpORtp
    case 'GP'
        y_noise = y_noise_gp;
    case 'TP'
        y_noise = y_noise_tp;
end
%%  Generate samples
x = linspace(-10,10,N_sample);

% y1 = 2.*cos(x).* (x);  %+ 0.08*randn(1,n);
% y2 = 1.5.*cos(x + 0.9).*(x);    %+ 0.06*randn(1,n);

y1 = 2*cos(x).* (x) ;           y1 = y1';
y2 = 1.5.*cos(x +pi/5).*(x);      y2 = y2';

xtr = x(train_series);
xtr = xtr';

xte = x;
xte = xte';

y = [y1 y2] + y_noise;

ytr = y(train_series,:);
yte = [y1 y2];

%% regression/prediction
kernel = @covSEiso; init_func = @SE_init;

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
legend('MV-GP','MV-TP','GP','TP','Location','northwest')
set(gca,'XTick',1:1:2)
set(gca,'XtickLabel',{'y_1','y_2'})
title('RMSE')

%%
% multi
figure(2);
subplot(241)
plot(xte(:,1),mGPpredictor.mean(:,1),'b')
hold on
% plot(xte(:,1),yte(:,1),'r:')
plot(xte,yte(:,1),'m')
hold on
plot(xtr(:,1),ytr(:,1),'ro')
hold on
plot(xte(:,1),mGPpredictor.lb(:,1),'b:')
hold on
plot(xte(:,1),mGPpredictor.ub(:,1),'b:')
title('MV-GP, y_1')
% legend('Prediction','True function','Observation','95% Confidence Interval')


subplot(242)
plot(xte(:,1),mTPpredictor.mean(:,1),'b')
hold on
% plot(xte(:,1),yte(:,1),'r:')
plot(xte,yte(:,1),'m')
hold on
plot(xtr(:,1),ytr(:,1),'ro')
hold on
plot(xte(:,1),mTPpredictor.lb(:,1),'b:')
hold on
plot(xte(:,1),mTPpredictor.ub(:,1),'b:')
title('MV-TP, y_1')

subplot(245)
plot(xte(:,1),mGPpredictor.mean(:,2),'b')
hold on
% plot(xte(:,1),yte(:,2),'r:')
plot(xte,yte(:,2),'m')
hold on
plot(xtr(:,1),ytr(:,2),'ro')
hold on
plot(xte(:,1),mGPpredictor.lb(:,2),'b:')
hold on
plot(xte(:,1),mGPpredictor.ub(:,2),'b:')
title('MV-GP, y_2')

subplot(246)
plot(xte(:,1),mTPpredictor.mean(:,2),'b')
hold on
% plot(xte(:,1),yte(:,2),'r:')
plot(xte,yte(:,2),'m')
hold on
plot(xtr(:,1),ytr(:,2),'ro')
hold on
plot(xte(:,1),mTPpredictor.lb(:,2),'b:')
hold on
plot(xte(:,1),mTPpredictor.ub(:,2),'b:')
title('MV-TP, y_2')

% indep
subplot(243)
plot(xte(:,1),GPpredictor{1}.mean,'b')
hold on
% plot(xte(:,1),yte(:,1),'r:')
plot(xte,yte(:,1),'m')
hold on
plot(xtr(:,1),ytr(:,1),'ro')
hold on
plot(xte(:,1),GPpredictor{1}.lb,'b:')
hold on
plot(xte(:,1),GPpredictor{1}.ub,'b:')
title('GP, y_1')

subplot(244)
plot(xte(:,1),TPpredictor{1}.mean(:,1),'b')
hold on
% plot(xte(:,1),yte(:,1),'r:')
plot(xte,yte(:,1),'m')
hold on
plot(xtr(:,1),ytr(:,1),'ro')
hold on
plot(xte(:,1),TPpredictor{1}.lb,'b:')
hold on
plot(xte(:,1),TPpredictor{1}.ub,'b:')
title('TP, y_1')
legend('Prediction','True function','Observation','95% Confidence Interval')
legend('boxoff')

subplot(247)
plot(xte(:,1),GPpredictor{2}.mean,'b')
hold on
% plot(xte(:,1),yte(:,2),'r:')
plot(xte,yte(:,2),'m')
hold on
plot(xtr(:,1),ytr(:,2),'ro')
hold on
plot(xte(:,1),GPpredictor{2}.lb,'b:')
hold on
plot(xte(:,1),GPpredictor{2}.ub,'b:')
title('GP, y_2')

subplot(248)
plot(xte(:,1),TPpredictor{2}.mean,'b')
hold on
% plot(xte(:,1),yte(:,2),'r:')
plot(xte,yte(:,2),'m')
hold on
plot(xtr(:,1),ytr(:,2),'ro')
hold on
plot(xte(:,1),TPpredictor{2}.lb,'b:')
hold on
plot(xte(:,1),TPpredictor{2}.ub,'b:')
title('TP, y_2')
legend('Prediction','True function','Observation','95% Confidence Interval')
legend('boxoff')

set(gcf,'Position',[213 355  1383  612]);
