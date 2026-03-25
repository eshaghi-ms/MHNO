clc;
clear;

%% Parameter Initialization

% Spatial Parameters
Nx=64;
Ny=Nx;
Lx=2*pi;
Ly=2*pi;
hx=Lx/Nx;
hy=Ly/Ny;

x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
[X,Y]=ndgrid(x,y); 

% Interfacial energy constant
epsilon=0.1; 
Cahn=epsilon^2;

% Discrete Fourier Transform
p=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
q=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
p2=p.^2; 
q2=q.^2; 
[pp2,qq2]=ndgrid(p2,q2);

% Time Discritization
dt=0.01; 
T=1; 
Nt=round(T/dt); 
ns=1;

% Dataset
data_size = 2000;
phi = NaN(data_size, Nt+1, Nx, Ny,'single');

%% Initial Condition
%u=rand(Nx,Ny)-0.5;

% Parameters of covariance C = tau^(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
% Note that we need alpha > d/2 (here d= 2) 
% alpha and tau control smoothness; the bigger they are, the smoother the
% function
tau = 400%300%250;%5;
alpha = 115%80%115;%2.5;
%figure;

for data_num = 1:data_size
    disp("data number = " + num2str(data_num))
    norm_a = GRF(alpha, tau, Nx);
    %norm_a = norm_a - abs(min(norm_a(:))) + 1.5*std(norm_a(:));
    norm_a = norm_a - 0.5 * std(norm_a(:));
    
    %u = norm_a;
    u = zeros(Nx,Nx);
    u(norm_a >= 0) = 1;
    u(norm_a < 0) = -1;
    
    %% Initial Preview
    clf;
    set(gcf, 'Position', [200, 200, 700, 700]);
    colormap(hsv)
    contourf(x,y,real(u'),[0 0], 'LineStyle', 'none');
    axis image
    axis([x(1) x(Nx) y(1) y(Ny)])
    pause(2)
    
    %% Update
    
    for iter=1:Nt
        phi(data_num, iter,:,:) = u;
        u=real(u);
        s_hat=fft2(Cahn*u-dt*(u.^3-3*u));
        v_hat=s_hat./(Cahn+dt*(2+Cahn*(pp2+qq2)));
        u=ifft2(v_hat);
        if (mod(iter,ns)==0)
            contourf(x,y,real(u'),[0 0], 'LineStyle', 'none');
            axis image
            %colorbar
            %clim([-1.5, 1.5])
            axis([x(1) x(Nx) y(1) y(Ny)])
            pause(0.01)
        end
    end
    phi(data_num, iter+1,:,:) = u;
end


save("AC2D_" + num2str(data_size) +  "_Nt_" + num2str(Nt+1) + ...
     "_Nx_" + num2str(Nx) + ".mat", '-v7.3');
