clc;
clear;
close all;
fclose('all');
disp('START')
%% Parameter Initialization

FigDraw = 0;

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

% Time Discritization
% dt=0.0001; 
% Nt=50000;
dt=0.0002;
Nt=25000;
T=Nt*dt;
num_saved_steps = 101;
ns=Nt/(num_saved_steps-1);

Nts = linspace(0, 1, num_saved_steps); 
Nts = round(1 + (Nt - 1) * Nts.^2);
Nts = unique(Nts);
size(Nts)

% Dataset
data_size = 4440;
binary_filename = "MBE2D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
                  "_Nx_" + num2str(Nx) + ".bin";
mat_filename = "MBE2D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
               "_Nx_" + num2str(Nx) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Initial Condition
% Parameters of covariance C = tau^(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
% Note that we need alpha > d/2 (here d= 2) 
% alpha and tau control smoothness; the bigger they are, the smoother the
% function

%u=1*(sin(3*xx).*sin(2*yy)+sin(5*xx).*sin(5*yy));
%u=1*(2*rand(Nx,Ny)-1);
tau = 150.;
alpha = 100.0;

if FigDraw
    figure;
end

for data_num = 1:data_size
    tic;
    disp("data number = " + num2str(data_num))

    u = 10*GRF(alpha, tau, Nx);
    all_iterations = zeros(num_saved_steps, Nx, Ny, 'single');

    %% Initial Preview
    if FigDraw
        clf;
        set(gcf, 'Position', [200, 200, 700, 700]);
        colormap jet;
        surf(x,y,real(u'));
        axis image;
        view(0,90);
        shading interp;
        colorbar;
        pause(2)
    end

    %% Update
    save_idx = 1;
    iter = 1;
    while iter <= Nt
        % disp("Iteration = " + num2str(iter))
        if iter == 1 || ismember(iter, Nts) || iter == Nt
            all_iterations(save_idx, :, :, :) = u;
            % disp("Saved = " + num2str(iter) + "at" + num2str(save_idx))
            save_idx = save_idx + 1;
        end

        u=real(u); 
        tu=fft2(u);
        fx=real(ifft2(pp.*tu)); 
        fy=real(ifft2(qq.*tu));
        f1=(fx.^2+fy.^2).*fx; 
        f2=(fx.^2+fy.^2).*fy;
        s_hat=fft2(u/dt)+(pp.*fft2(f1)+qq.*fft2(f2));
        v_hat=s_hat./(1/dt-(pp2+qq2)+epsilon*(pp2+qq2).^2);
        u=ifft2(v_hat);

        if isnan(sum(sum(u)))
            disp('The iteration is repeated')
            disp("data number = " + num2str(data_num))
            u = 10*GRF(alpha, tau, Nx);
            all_iterations = zeros(num_saved_steps, Nx, Ny, 'single');
        
            if FigDraw
                clf;
                set(gcf, 'Position', [200, 200, 700, 700]);
                colormap jet;
                surf(x,y,real(u'));
                axis image;
                view(0,90);
                shading interp;
                colorbar;
                pause(2)
            end
            save_idx = 1;
            iter=1;
            
            disp("Iteration = " + num2str(iter))
            if iter == 1 || ismember(iter, Nts) || iter == Nt
                all_iterations(save_idx, :, :, :) = u;
                disp("Saved = " + num2str(iter) + "at" + num2str(save_idx))
                save_idx = save_idx + 1;
            end
    
            u=real(u); 
            tu=fft2(u);
            fx=real(ifft2(pp.*tu)); 
            fy=real(ifft2(qq.*tu));
            f1=(fx.^2+fy.^2).*fx; 
            f2=(fx.^2+fy.^2).*fy;
            s_hat=fft2(u/dt)+(pp.*fft2(f1)+qq.*fft2(f2));
            v_hat=s_hat./(1/dt-(pp2+qq2)+epsilon*(pp2+qq2).^2);
            u=ifft2(v_hat);
        end

        if ismember(iter, Nts) && FigDraw
            clf;
            surf(x,y,real(u'));
            view(0,90);
            shading interp;
            axis image;
            colorbar;
            pause(0.01)
        end
        iter = iter + 1;
    end
    disp('Writing')
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