clc;
clear;
close all

%% Parameter Initialization

% Spatial Parameters
Nx=64; 
Ny=64; 
Lx=1; 
Ly=1; 
hx=Lx/Nx; 
hy=Ly/Ny;
x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);

% Interfacial energy constant
%m = 6; 
%eps1 = hx*m/(2*sqrt (2)*atanh(0.9)) ; 
epsilon=0.0125; 
Cahn=epsilon^2;

% Discrete Fourier Transform
p=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
q=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
p2=p.^2; 
q2=q.^2; 
[pp2,qq2]=ndgrid(p2,q2);

% Time Discritization
dt=0.0025; 
T=0.5; 
Nt=round(T/dt);
Np=200;
ns=Nt/Np;

%% Initial Condition
%u = -0.045 + 0.05*(2*rand(Nx,Ny)-1) ;
%u=0.05*(2*rand(Nx,Ny)-1);
tau = 2000;
alpha = 1;
u=GRF(alpha, tau, Nx);
%% Initial Preview
figure(1);
clf;
contourf(x,y,real(u'),[0 0]); 
axis image
axis([x(1) x(Nx) y(1) y(Ny)])
pause(2)

%% Update
for iter = 1:Nt
    disp(['iteration = ' num2str(iter)])
    disp(['Maximum = ' num2str(max(u, [],'all'))])
    disp(['Minimum = ' num2str(min(u, [],'all'))])
    u=real(u);
    s_hat=fft2(u)-dt*(pp2+qq2).*fft2(u.^3-3*u);
    v_hat=s_hat./(1.0+dt*(2.0*(pp2+qq2)+Cahn*(pp2+qq2).^2));
    u=ifft2(v_hat);
    if (mod(iter,ns)==0)
        contourf(x,y,real(u'),[0 0]);
        title(['\Delta t = '  num2str(iter)])
        axis image
        axis([x(1) x(Nx) y(1) y(Ny)])
        pause(0.1)
    end
end