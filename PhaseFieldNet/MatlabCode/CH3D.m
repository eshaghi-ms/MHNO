clc;
clear;
close all

%% Parameter Initialization

% Spatial Parameters
Nx=32; 
Ny=32; 
Nz=32; 
Lx=1.0; 
Ly=1.0; 
Lz=1.0; 
hx=Lx/Nx;
hy=Ly/Ny;
hz=Lz/Nz;
x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
z=linspace(-0.5*Lz+hz,0.5*Lz,Nz);
[xx,yy,zz]=ndgrid(x,y,z);

% Interfacial energy constant
%epsilon=1/128; 
epsilon=0.0125; 
Cahn=epsilon^2;

% Discrete Fourier Transform
kx=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
ky=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
kz=2*pi/Lz*[0:Nz/2 -Nz/2+1:-1];
k2x=kx.^2; 
k2y=ky.^2; 
k2z=kz.^2;
[kxx,kyy,kzz]=ndgrid(k2x,k2y,k2z);

% Time Discritization
dt=0.001; 
T=10; 
Nt=round(T/dt); 
ns=100;

%% Initial Condition
%u=rand(Nx,Ny,Nz)-0.5;
%u=-0.45 + 0.05*(2*rand(Nx,Ny,Nz)-1);
%u=0.45 - 0.05*(2*rand(Nx,Ny,Nz)-1);

%tau = 400;
%alpha = 115;
tau = 10;
alpha = 5.5;
norm_a = GRF3D(alpha, tau, Nx);
norm_a = norm_a + 0.2 * std(norm_a(:));   
u = ones(Nx,Ny,Nz);
u(norm_a < 0) = -1;
%u = GRF3D(alpha, tau, Nx);

%% Initial Preview
% figure(1);
% clf;
% p1=patch(isosurface(xx,yy,zz,real(u),0.));
% set(p1,'FaceColor','g','EdgeColor','none'); 
% daspect([1 1 1])
% camlight;
% lighting phong; 
% box on; 
% axis image;
% view(45,45);
% pause(2)

%% Update
for iter=1:Nt
    disp(['Iteration = ' num2str(iter)]);
    u=real(u);
    s_hat=fftn(u)-dt*(kxx+kyy+kzz).*fftn(u.^3-3*u);
    v_hat=s_hat./(1.0+dt*(2.0*(kxx+kyy+kzz)+Cahn*(kxx+kyy+kzz).^2));
    u=ifftn(v_hat);
    if (mod(iter,ns)==0)
        figure(1);
        clf;
        p1=patch(isosurface(xx,yy,zz,real(u),0.));
        set(p1,'FaceColor','g','EdgeColor','none'); 
        daspect([1 1 1])
        camlight;
        lighting phong; 
        box on; 
        axis image;
        view(45,45);
        pause(0.01)
    end
end