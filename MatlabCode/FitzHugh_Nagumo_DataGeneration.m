clear; clc; close all

%% Parameters

% FitzHugh-Nagumo PDE coefficients
c0 = -0.03;
c1 = 2.0;
delta = 3.0;
epsilon = 0.02;

% Domain
L = 20;

% Perturbation parameters
Dt_pert = 50;
x1 = 7.5;
x2 = 10;
x3 = 12.5;
sig_pert = 0.5;

% Simulation parameters
Tsim = 500;
Dt = 1.0;
N_cos = 128;

filname = 'FitzHugh_Nagumo_Data_6.txt';
fileID = fopen(filname, 'w'); % write
%fileID = fopen(filname, 'a'); % append

%% Initialize Simulation
params.c0 = c0;
params.c1 = c1;
params.delta = delta;
params.epsilon = epsilon;
params.L = L;
params.Dt_pert = Dt_pert;
params.x1 = x1;
params.x2 = x2;
params.x3 = x3;
params.sig_pert = sig_pert;
params.N_cos = N_cos;

% load initial condition
% IC_data = load('FitzHughNagumo_IC.mat');
% x_pts = L*(2*(1:N_cos) - 1)/(2*N_cos);
% % v_guess = IC_data.v_init;
% % w_guess = IC_data.w_init;
% % v_hat_guess = IC_data.v_hat_init;
% % w_hat_guess = IC_data.w_hat_init;

x_pts = L*(2*(1:N_cos) - 1)/(2*N_cos);

v_guess = 0.8*tanh(x_pts-L/2);
w_guess = 0.1*tanh(x_pts-L/2);

v_hat_guess = dct(v_guess/sqrt(N_cos));
w_hat_guess = dct(w_guess/sqrt(N_cos));

% solve for steady state initial condition
y_init = fsolve(@(y) FitzHugh_Nagumo_TimeDeriv( 0, y, params ),...
    [v_hat_guess';w_hat_guess']);

%% Simulate equation

N_perts = Tsim/Dt_pert;
t_vec = 0:Dt:Tsim;
Y_sim = zeros(Tsim+1, 2*N_cos);
Y_sim(1,:) = y_init;
for nn = 1:N_perts
    %solution
    n = (nn-1)*Dt_pert + 1;
    v_hat = Y_sim(n,1:N_cos)';
    w_hat = Y_sim(n,N_cos+1:end)';
    
    % Introduce perturbationssize
    v = sqrt(N_cos) * idct(v_hat);
    pert = zeros(N_cos,1);
    for i=1:N_cos
        u = sig_pert*randn(3,1);
        pert(i) = u(1)*exp(-(x_pts(i)-x1)^2) +...
            u(2)*exp(-(x_pts(i)-x2)^2) + u(3)*exp(-(x_pts(i)-x3)^2);
    end
    v = v + pert;
    v_hat = dct(v/sqrt(N_cos));
    
    y_init = [v_hat; w_hat];
    
    % simulate equation using cosine pseudo-spectral method
    [t, Y] = ode15s(@(t,y) FitzHugh_Nagumo_TimeDeriv( t, y, params ),...
        t_vec(n):Dt:t_vec(n+Dt_pert), y_init);
    
    Y_sim((n+1):(n+Dt_pert),:) = Y(2:end,:);
end

% transform back into physical space
V_sim = zeros(Tsim+1, N_cos);
W_sim = zeros(Tsim+1, N_cos);
for n=1:Tsim+1
    v_hat = Y_sim(n,1:N_cos);
    w_hat = Y_sim(n,N_cos+1:end);
    
    V_sim(n,:) = sqrt(N_cos) * idct(v_hat);
    W_sim(n,:) = sqrt(N_cos) * idct(w_hat);
end

[tt, xx] = meshgrid(t_vec, x_pts);

figure()
subplot(2,1,1)
contourf(tt,xx,V_sim',15)
ylim([0,20])
colorbar
title('v')
xlabel('t')
ylabel('x')
subplot(2,1,2)
contourf(tt,xx,W_sim',15)
ylim([0,20])
colorbar
title('w')
xlabel('t')
ylabel('x')

%% Write data to file

fprintf('\n ***Writing Data to File*** \n')
for n=1:length(t_vec)
    fprintf(fileID, '%f\t', t_vec(n));
    for j=1:N_cos
        fprintf(fileID, '%f\t', V_sim(n,j));
        fprintf(fileID, '%f\t', W_sim(n,j));
    end
    fprintf(fileID, '\r\n');
end

fclose(fileID);