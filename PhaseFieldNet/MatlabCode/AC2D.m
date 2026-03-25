clc;
clear;
close all

%% Parameter Initialization

% Spatial Parameters
Nx=128; 
Ny=128; 
Lx=2*pi; 
Ly=2*pi; 
hx=Lx/Nx; 
hy=Ly/Ny;

x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
[xx,yy]=ndgrid(x,y); 

% Interfacial energy constant
epsilon=0.05; 
Cahn=epsilon^2;

% Discrete Fourier Transform
p=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
q=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
p2=p.^2; 
q2=q.^2; 
[pp2,qq2]=ndgrid(p2,q2);

% Time Discritization
dt=0.01; 
T=30; 
Nt=round(T/dt); 
ns=10;

%% Initial Condition
% u=tanh((2-sqrt(xx.^2+yy.^2))/(sqrt(2)*epsilon));
u=tanh((1.7 + 1.2*cos(6*atan(yy./xx))-sqrt(xx.^2+yy.^2))/(sqrt(2)*epsilon));
u(isnan(u))=0;
%% Initial Preview
figure(1); 
clf;
contourf(x,y,real(u'),[0 0]); 
axis image
axis([x(1) x(Nx) y(1) y(Ny)])
pause(2)

%% Update
for iter=1:Nt
    u=real(u);
    s_hat=fft2(Cahn*u-dt*(u.^3-3*u));
    v_hat=s_hat./(Cahn+dt*(2+Cahn*(pp2+qq2)));
    u=ifft2(v_hat);
    if (mod(iter,ns)==0)
        contourf(x,y,real(u'),[0 0]);
        axis image
        axis([x(1) x(Nx) y(1) y(Ny)])
        pause(0.01)
    end
end