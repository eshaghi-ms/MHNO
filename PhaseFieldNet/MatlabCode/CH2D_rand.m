clc;
clear;
close all;
fclose('all');

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
[X,Y]=ndgrid(x,y); 

% Interfacial energy consta
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
Nt = 200;
T=Nt*dt;
Nt=round(T/dt);
num_saved_steps = 101;
ns=Nt/(num_saved_steps-1);

% Dataset
data_size = 4400;
binary_filename = "CH2D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
                  "_Nx_" + num2str(Nx) + ".bin";
mat_filename = "CH2D_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
               "_Nx_" + num2str(Nx) + ".mat";

%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end


%% Initial Condition
%u=0.05*(2*rand(Nx,Ny)-1);

% Parameters of covariance C = tau^(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
% Note that we need alpha > d/2 (here d= 2) 
% alpha and tau control smoothness; the bigger they are, the smoother the
% function
%tau = 2000;
%alpha = 1;

tau = 400;
alpha = 115;

% figure;

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
    % clf;
    % set(gcf, 'Position', [200, 200, 700, 700]);
    % colormap(hsv)
    % contourf(x,y,real(u'),[0 0], 'LineStyle', 'none');
    % axis image
    % axis([x(1) x(Nx) y(1) y(Ny)])
    % pause(2)

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
        s_hat=fft2(u)-dt*(pp2+qq2).*fft2(u.^3-3*u);
        v_hat=s_hat./(1.0+dt*(2.0*(pp2+qq2)+Cahn*(pp2+qq2).^2));
        u=ifft2(v_hat);
        % if (mod(iter,ns)==0)
        %    contourf(x,y,real(u'),[0 0], 'LineStyle', 'none');
        %    axis image
        %    %colorbar
        %    %clim([-1.5, 1.5])
        %    axis([x(1) x(Nx) y(1) y(Ny)])
        %    pause(0.01)
        % end
    end
    fwrite(fileID, all_iterations, 'single');
end

fclose(fileID);

%% Convert Binary Data to MAT File
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
