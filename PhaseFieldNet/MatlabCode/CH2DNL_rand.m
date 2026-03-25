clc;
clear;
close all
fclose('all');

%% Parameter Initialization

FigDraw = false;

% Spatial Parameters
nx = 64;
ny = nx;
xL = -3;
xR = 3;
yL = -3;
yR = 3;
h = (xR-xL)/nx;
x = linspace (xL+0.5*h,xR-0.5*h,nx);
y = linspace (yL+0.5*h,yR-0.5*h,ny) ;

% Constant
eps1 = 0.07;
eps2 = eps1^2;

% Discrete Fourier Transform
xi = pi*(0:nx-1)/(xR-xL);
eta = pi*(0:ny-1)/(yR-yL);

% Time Discritization
dt=0.0005;
nt=1000;
T = nt*dt;
num_saved_steps = 101;
ns=nt/(num_saved_steps-1);

% Dataset
data_size = 4400;
binary_filename = "CH2DNL_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
                  "_Nx_" + num2str(nx) + ".bin";
mat_filename = "CH2DNL_" + num2str(data_size) + "_Nt_" + num2str(num_saved_steps) + ...
               "_Nx_" + num2str(nx) + ".mat";



%% Prepare Binary File
fileID = fopen(binary_filename, 'wb');
if fileID == -1
    error("Cannot open binary file for writing.");
end

%% Initial Condition
% tau = 400;
% alpha = 115;
tau = 5;
alpha = 2.5;

bar_psi = 0.0*ones(nx,ny);
sig = 20;

if FigDraw
    figure;
end

for data_num = 1:data_size
    disp("data number = " + num2str(data_num))

    norm_a = GRF(alpha, tau, nx);
    psi = zeros(nx,nx);
    psi(norm_a >= 0) = 1;
    psi(norm_a < 0) = -1;
    
    all_iterations = zeros(num_saved_steps, nx, ny, 'single');

    %% Initial Preview
    if FigDraw
        set(gcf, 'Position', [200, 200, 700, 700]);
        contourf(x,y,real(psi'), 'LineStyle', 'none');
        colormap("jet");
        colorbar;
        title('Initial State')
        axis image
        axis([x(1) x(nx) y(1) y(ny)])
        pause(2)
    end

    %% Update
    save_idx = 1;
    for it = 1:nt
        % disp(['iteration = ' num2str(it)])
        % disp(['Maximum = ' num2str(max(psi, [],'all'))])
        % disp(['Minimum = ' num2str(min(psi, [],'all'))])

        if it == 1 || mod(it,ns) == 0 || it == nt
            all_iterations(save_idx, :, :, :) = psi;
            % disp("Saved = " + num2str(it) + "at" + num2str(save_idx))
            save_idx = save_idx + 1;
        end

        f = psi .^3-3* psi ;
        hat_psi = dct2( psi ) ;
        hat_f = dct2( f ) ;
        psi = idct2 (( hat_psi + (sig*dct2( bar_psi)-(xi'.^2+ eta .^2).*hat_f)*dt) ...
        ./(1+(sig+2*(xi'.^2+ eta .^2)+eps2*(xi'.^2+ eta .^2) .^2)*dt) ) ;
        if mod(it ,ns)==0 && FigDraw
            %contourf(x,y,real(psi'),[0 0]);
            contourf(x,y,real(psi'), 'LineStyle', 'none'); colormap("jet"); colorbar;
            title(['\Delta t = '  num2str(it)])
            axis image
            axis([x(1) x(nx) y(1) y(ny)])
            pause(0.1)
        end
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
phi_mat.phi = zeros([data_size, num_saved_steps, nx, ny], 'single');

for data_num = 1:data_size
    disp("Saving dataset " +  num2str(data_num));
    data_chunk = fread(fileID, num_saved_steps * nx * ny, 'single');
    phi_mat.phi(data_num, :, :, :) = reshape(data_chunk, [1, num_saved_steps, nx, ny]);
end

fclose(fileID);