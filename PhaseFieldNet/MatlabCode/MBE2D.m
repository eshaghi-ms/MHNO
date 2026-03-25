
clc;
clear;
% close all

%% Parameter Initialization

% Spatial Parameters
Nx=64; 
Ny=64; 
Lx=2*pi; 
Ly=2*pi; 
hx=Lx/Nx; 
hy=Ly/Ny;

x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
[xx,yy]=ndgrid(x,y);

% Constant
epsilon=0.1;

% Discrete Fourier Transform
p=1i*2*pi/Lx*[0:Nx/2-1 0 -Nx/2+1:-1];
q=1i*2*pi/Ly*[0:Ny/2-1 0 -Ny/2+1:-1]; 
[pp,qq]=ndgrid(p,q);
p2=(2*pi/Lx*[0:Nx/2 -Nx/2+1:-1]).^2;
q2=(2*pi/Ly*[0:Ny/2 -Ny/2+1:-1]).^2; 
[pp2,qq2]=ndgrid(p2,q2);

u=1*(sin(3*xx).*sin(2*yy)+sin(5*xx).*sin(5*yy));
%u=1*(2*rand(Nx,Ny)-1);
tau = 300%1.0;%2.0;
alpha = 10%1.5;%2.5;
tau = 2.5;
alpha = 2.0;
tau = 150.;
alpha = 100.0;
u = 10*GRF(alpha, tau, Nx);

% Time Discritization
dt=0.0001; 
Nt = 50000;
T=Nt*dt; 
Np=100;
ns=Nt/Np;

iter_draw = 0;

figure(1); clf; colormap jet;
surf(x,y,real(u')); 
axis image; 
%view(0,90); 
shading interp;
colorbar; 
%caxis([-1 1]); 
pause(3);
for iter=1:Nt
    disp("Iteration Number = " + num2str(iter))
    u=real(u); tu=fft2(u);
    fx=real(ifft2(pp.*tu)); fy=real(ifft2(qq.*tu));
    f1=(fx.^2+fy.^2).*fx; f2=(fx.^2+fy.^2).*fy;
    s_hat=fft2(u/dt)+(pp.*fft2(f1)+qq.*fft2(f2));
    v_hat=s_hat./(1/dt-(pp2+qq2)+epsilon*(pp2+qq2).^2);
    u=ifft2(v_hat);
    if (mod(iter,ns)==0)
        iter_draw = iter_draw + 1;
        disp("Drawing Iteration Number = " + num2str(iter_draw))
        clf;
        surf(x,y,real(u')); 
        %view(0,90); 
        shading interp;
        axis image; 
        colorbar; 
        %caxis([-1 1]);
        pause(1)
    end
end

