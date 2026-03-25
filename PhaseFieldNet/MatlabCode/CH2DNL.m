clc;
clear;
close all

%% Parameter Initialization

% Spatial Parameters 
nx = 64; 
ny = nx;
xL = -3; 
xR = 3; 
yL = -3; 
yR = 3;
h = (xR-xL)/nx;
x = linspace (xL+0.5*h,xR-0.5*h,nx);
y = linspace (yL+0.5*h,yR-0.5*h,ny) ;

% Time Discritization
T = 0.5;
dt=0.0005; 
nt = T/dt;

% Discrete Fourier Transform
xi = pi*(0:nx-1)/(xR-xL);
eta = pi*(0:ny-1)/(yR-yL);

% Constant
m = 4; 
eps1 = h*m/(2*sqrt (2)*atanh(0.9)); 
% eps1 = 0.08;
% Maximum = 0.0003011
% Minimum = -0.00030817
% eps1 = 0.04;
% Maximum = 0.62164
% Minimum = -0.62862
% eps1 = 0.02;
% Maximum = 0.8747
% Minimum = -0.88032
% eps1 = 0.025;
% Maximum = 0.85057
% Minimum = -0.85062
eps1 = 0.0225;
eps1 = 0.05;
eps1 = 0.07;
% eps1 = 4*eps1;

eps2 = eps1^2;

%% Initial Condition
% tau = 400;
% alpha = 115;
tau = 5;
alpha = 2.5;
bar_psi = 0.0*ones(nx,ny) ;
%bar_psi = 0.01*ones(nx,ny) ;
sig = 80;
sig = 20;
sig = 10;

% psi = bar_psi+1*(1-2*rand(nx,ny));
% psi = bar_psi+GRF(alpha, tau, nx);

norm_a = GRF(alpha, tau, nx);
% min(norm_a,[],'all')
% max(norm_a,[],'all')
% norm_a = norm_a - 0.5 * std(norm_a(:));
% % 
psi = zeros(nx,nx);
psi(norm_a >= 0) = 1;
psi(norm_a < 0) = -1;
% % psi = bar_psi+psi;
% 
% psi = GRF(alpha, tau, nx);

%% Update
for it = 1:nt
    disp(['iteration = ' num2str(it)])
    disp(['Maximum = ' num2str(max(psi, [],'all'))])
    disp(['Minimum = ' num2str(min(psi, [],'all'))])
    
    f = psi .^3-3* psi ;
    hat_psi = dct2( psi ) ;
    hat_f = dct2( f ) ;
    psi = idct2 (( hat_psi + (sig*dct2( bar_psi)-(xi'.^2+ eta .^2).*hat_f)*dt) ...
    ./(1+(sig+2*(xi'.^2+ eta .^2)+eps2*(xi'.^2+ eta .^2) .^2)*dt) ) ;
    if mod(it ,10)==0
        %contourf(x,y,real(psi'),[0 0]);
        contourf(x,y,real(psi'), 'LineStyle', 'none'); colormap("jet"); colorbar;
        title(['\Delta t = '  num2str(it)])
        axis image
        axis([x(1) x(nx) y(1) y(ny)])
        pause(0.1)
    end
end