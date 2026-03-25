clc;
clear;
close all;
fclose('all');
disp('START')
%% Parameter Initialization

FigDraw = 1;

% Spatial Parameters
Nx=64;
Ny=Nx;
Lx=64;
Ly=Lx;
hx=Lx/Nx; 
hy=Ly/Ny;

x=linspace(-0.5*Lx+hx,0.5*Lx,Nx);
y=linspace(-0.5*Ly+hy,0.5*Ly,Ny);
[X,Y]=ndgrid(x,y); 

% Constant
epsilon=0.5;

% Discrete Fourier Transform
p=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
q=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
p2=p.^2; 
q2=q.^2; 
[pp2,qq2]=ndgrid(p2,q2);

% Time Discritization
dt=0.005; 
Nt = 10000;
T=Nt*dt;
num_saved_steps = 101;
ns=Nt/(num_saved_steps-1);

% Dataset
data_size = 11100;
binary_filename = "SH2D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
                  "_Nx_" + num2str(Nx) + ".bin";
mat_filename = "SH2D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
               "_Nx_" + num2str(Nx) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Initial Condition
%u=0.2*(2*rand(Nx,Ny)-1);

% Parameters of covariance C = tau^(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
% Note that we need alpha > d/2 (here d= 2) 
% alpha and tau control smoothness; the bigger they are, the smoother the
% function
tau = 8.;
alpha = 4.;
%tau = 3.5;
%alpha = 2.;
if FigDraw
    figure;
end

for data_num = 1:data_size
    disp("data number = " + num2str(data_num))
    norm_a = GRF(alpha, tau, Nx);
    %norm_a = norm_a - abs(min(norm_a(:))) + 1.5*std(norm_a(:));
    norm_a = norm_a - 0.5 * std(norm_a(:));
    
    %u = norm_a;
    u = zeros(Nx,Nx);
    u(norm_a >= 0) = 1;
    u(norm_a < 0) = -1;

    all_iterations = zeros(num_saved_steps, Nx, Ny, 'single');

    %% Initial Preview
    if FigDraw
        clf;
        % set(gcf, 'Position', [100, 100, 900, 900]);
        surf(x,y,real(u')); 
        shading interp; 
        view(0,90); 
        axis image;
        colorbar;
        pause(1)
    end

    %% Update
    save_idx = 1;
    for iter=1:Nt
        % disp("Iteration = " + num2str(iter))
        if iter == 1 || mod(iter,ns) == 0 || iter == Nt
            all_iterations(save_idx, :, :, :) = u;
            % disp("Saved = " + num2str(iter) + "at" + num2str(save_idx))
            save_idx = save_idx + 1;
        end

        u=real(u);
        s_hat=fft2(u/dt)-fft2(u.^3)+2*(pp2+qq2).*fft2(u);
        v_hat=s_hat./(1.0/dt+(1-epsilon)+(pp2+qq2).^2);
        u=ifft2(v_hat);
        if (mod(iter,ns)==0) && FigDraw
           surf(x,y,real(u'));
           shading interp; 
           view(0,90);
           axis image
           clim([-0.6 0.6]);
           colorbar;
           pause(0.01)
        end
    end
    fwrite(fileID, all_iterations, 'single');
end

%% Convert Binary Data to MAT File
fclose(fileID);
fileID = fopen(binary_filename, 'rb');
if fileID == -1
    error("Cannot open binary file for reading.");
end

phi_mat = matfile(mat_filename, 'Writable', true);
phi_mat.phi = zeros([data_size, num_saved_steps, Nx, Ny], 'single');

for data_num = 1:data_size
    disp("Saving dataset " +  num2str(data_num));
    data_chunk = fread(fileID, num_saved_steps * Nx * Ny, 'single');
    phi_mat.phi(data_num, :, :, :) = reshape(data_chunk, [1, num_saved_steps, Nx, Ny]);
end

fclose(fileID);