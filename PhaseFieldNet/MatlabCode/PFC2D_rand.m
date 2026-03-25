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

% Constant
epsilon=0.025;

% Discrete Fourier Transform
p=2*pi/Lx*[0:Nx/2 -Nx/2+1:-1];
q=2*pi/Ly*[0:Ny/2 -Ny/2+1:-1];
p2=p.^2; 
q2=q.^2; 
[pp2,qq2]=ndgrid(p2,q2);

% Time Discritization
dt=0.05;
dt=0.1;
Nt = 20000;
Nt = 10000;
T=Nt*dt; 
num_saved_steps = 101;
ns=Nt/(num_saved_steps-1);

% Dataset
data_size = 4440;
binary_filename = "PFC2D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
                  "_Nx_" + num2str(Nx) + ".bin";
mat_filename = "PFC2D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
               "_Nx_" + num2str(Nx) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Initial Condition

u_mean = 0.07;

%u=0.2*(2*rand(Nx,Ny)-1);

tau = 3.5;
alpha = 2.0;

if FigDraw
    figure;
end

for data_num = 1:data_size
    tic;
    disp("data number = " + num2str(data_num))

    u = u_mean + u_mean*GRF(alpha, tau, Nx);
    all_iterations = zeros(num_saved_steps, Nx, Ny, 'single');

    %% Initial Preview
    if FigDraw
        clf;
        % set(gcf, 'Position', [100, 100, 900, 900]);
        surf(x,y,real(u'));
        colormap jet
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
        s_hat=fft2(u/dt)-(pp2+qq2).*fft2(u.^3)+2*(pp2+qq2).^2.*fft2(u);
        v_hat=s_hat./(1.0/dt+(1-epsilon)*(pp2+qq2)+(pp2+qq2).^3);
        u=ifft2(v_hat);
        if (mod(iter,ns)==0) && FigDraw
            %disp(['Maximum = ' num2str(max(u, [],'all'))])
            %disp(['Minimum = ' num2str(min(u, [],'all'))])
    
            surf(x,y,real(u')); 
            colormap jet
            shading interp; 
            view(0,90);
            axis image;
            % clim([u_mean-0.2 u_mean+0.2]);
            colorbar;
            pause(0.01)
        end
    end
    fwrite(fileID, all_iterations, 'single');
    toc;
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