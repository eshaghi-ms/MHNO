clc
clear;

%% Parameter Initialization

% Spatial Parameters
Nx=32; 
Ny=Nx; 
Nz=Nx; 
Lx=3; 
Ly=3; 
Lz=3; 
hx=Lx/Nx; 
hy=Ly/Ny; 
hz=Lz/Nz;

x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
z=linspace(-0.5*Lz+hz,0.5*Lz,Nz);

[xx,yy,zz]=ndgrid(x,y,z); 

% Interfacial energy constant
epsilon=0.1;
Cahn=epsilon^2;

% Discrete Fourier Transform
kx=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
ky=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
kz=2*pi/Lz*[0:Nz/2 -Nz/2+1:-1];
k2x = kx.^2; 
k2y = ky.^2; 
k2z = kz.^2;
[kxx,kyy,kzz]=ndgrid(k2x,k2y,k2z);

% Time Discritization
dt=0.001;  % 0.001; 
T=100*0.001; 
Nt=round(T/dt); 
ns=1;

% Dataset
data_size = 1200;
phi = NaN(data_size, Nt+1, Nx, Ny, Nz, 'single');

%% Initial Condition

pad=5;
tau = 400;
alpha = 115;
figure(1); 

for data_num = 1:data_size
    disp("data number = " + num2str(data_num))
    
    norm_a = GRF3D(alpha, tau, Nx);
    norm_a = norm_a + 0.2 * std(norm_a(:));   
    u = ones(Nx,Nx,Nz);
    u(norm_a < 0) = -1;
    % [m, n, p] = ndgrid(1:Nx, 1:Nx, 1:Nz);
    % boundary_mask = (m <= pad | m > Nx-pad | ...
    %                  n <= pad | n > Nx-pad | ...
    %                  p <= pad | p > Nz-pad);
    % u(boundary_mask) = 1;

    %% Initial Preview
    clf;
    p1=patch(isosurface(xx,yy,zz,real(u),0.));
    set(p1,'FaceColor','g','EdgeColor','none'); 
    daspect([1 1 1])
    camlight;
    lighting phong; 
    box on; 
    axis image;
    xlim([-Lx/2 Lx/2])
    ylim([-Ly/2 Ly/2])
    zlim([-Lz/2 Lz/2])
    view(45,45);
    pause(2)

    %% Update
    for iter=1:Nt
        phi(data_num, iter,:,:,:) = u;
        u=real(u); 

        s_hat=fftn(Cahn*u-dt*(u.^3-3*u));
        v_hat=s_hat./(Cahn+dt*(2+Cahn*(kxx+kyy+kzz)));
        u=ifftn(v_hat);
        if (mod(iter,ns)==0)
            figure(1); 
            clf;
            p1=patch(isosurface(xx,yy,zz,real(u),0.));
            set(p1,'FaceColor','g','EdgeColor','none'); 
            daspect([1 1 1])
            camlight;lighting phong; 
            box on; 
            axis image;
            xlim([-Lx/2 Lx/2])
            ylim([-Ly/2 Ly/2])
            zlim([-Lz/2 Lz/2])
            view(45,45);
            pause(0.01)
        end
    end
    phi(data_num, iter+1,:,:,:) = u;
end

save("AC3D_" + num2str(Nx) +  "_" + num2str(data_size) + ".mat", '-v7.3');
