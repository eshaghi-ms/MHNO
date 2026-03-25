clc;
clear;
close all

%% Parameter Initialization

% Spatial Parameters
Nx=80; 
Ny=80; 
Nz=80; 
Lx=90; 
Ly=90; 
Lz=90; 
hx=Lx/Nx; 
hy=Ly/Ny; 
hz=Lz/Nz;
x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
z=linspace(-0.5*Lz+hz,0.5*Lz,Nz);
[xx,yy,zz]=ndgrid(x,y,z); 

% constant
epsilon=0.15;

% Discrete Fourier Transform
kx=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
ky=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
kz=2*pi/Lz*[0:Nz/2 -Nz/2+1:-1];
k2x = kx.^2; 
k2y = ky.^2; 
k2z = kz.^2;
[kxx,kyy,kzz]=ndgrid(k2x,k2y,k2z);

% Time Discritization
dt=0.01; 
T=5; 
Nt=round(T/dt); 
ns=Nt/10; 
t=0;

%% Initial Condition
% u=rand(Nx,Ny,Nz)-0.5;

tau = 2;
alpha = 3;
u = GRF3D(alpha, tau, Nx);

%% Initial Preview

for iter=1:Nt
    disp(['Iteration = ' num2str(iter)]);
    u=real(u);
    s_hat=fftn(u/dt)-fftn(u.^3)+2*(kxx+kyy+kzz).*fftn(u);
    v_hat=s_hat./(1.0/dt+(1-epsilon)+(kxx+kyy+kzz).^2);
    u=ifftn(v_hat);
    t=t+dt;

    if mod(iter, ns) == 0 
        % Extract mid-plane slices
        slice_x = squeeze(u(Nx/2, :, :));
        slice_y = squeeze(u(:, Ny/2, :));
        slice_z = squeeze(u(:, :, Nz/2));

        % Plot star layout
        clf; % Clear current figure
        hold on;
        % Midplane along X
        surf(squeeze(yy(Nx/2, :, :)), squeeze(zz(Nx/2, :, :)), slice_x, 'EdgeColor', 'none');
        % Midplane along Y
        surf(squeeze(xx(:, Ny/2, :)), squeeze(zz(:, Ny/2, :)), slice_y, 'EdgeColor', 'none');
        % Midplane along Z
        surf(squeeze(xx(:, :, Nz/2)), squeeze(yy(:, :, Nz/2)), slice_z, 'EdgeColor', 'none');

        % Visualization settings
        view(3); % 3D view
        axis tight;
        caxis([-0.6, 0.6]); % Adjust color range
        colormap(jet);
        colorbar;
        title(['Time t = ', num2str(t, '%.2f')]);
        drawnow; % Update the figure

    end

end