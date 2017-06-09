function [ ydot ] = FitzHugh_Nagumo_TimeDeriv( t, y, params )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

c0 = params.c0;
c1 = params.c1;
delta = params.delta;
epsilon = params.epsilon;
L= params.L;
Dt_pert = params.Dt_pert;
x1 = params.x1;
x2 = params.x2;
x3 = params.x3;
sig_pert = params.sig_pert;
N_cos = params.N_cos;

% domain
x_pts = L*(2*(1:N_cos) - 1)/(2*N_cos);
v_hat = y(1:N_cos);
w_hat = y(N_cos+1:end);

% evaluate nonlinearity using DCT
v = sqrt(N_cos) * idct(v_hat);
v3_hat = dct(v.^3 / sqrt(N_cos));

% evaluate derivatives
d_coeff = -(((1:N_cos)-1)*pi/L).^2;
vdot = d_coeff'.*v_hat + v_hat - w_hat - v3_hat;
wdot = delta*d_coeff'.*w_hat + ...
    epsilon*(v_hat - c1*w_hat - c0*[1;zeros(N_cos-1,1)]);

ydot = [vdot; wdot];

end

