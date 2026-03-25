% Define parameters
s = 64;                    % Fixed grid size for all plots
% tau_values = linspace(1500, 2000, 5);   % 5 different tau values
% alpha_values = linspace(0.1, 10, 5);   % 5 different alpha values
tau_values = linspace(2, 3, 5);   % 5 different tau values
alpha_values = linspace(3, 5, 5);   % 5 different alpha values

% Initialize figure
colormap(jet);              % Set colormap for all subplots
for i = 1:5
    for j = 1:5
        % Select current alpha and tau
        alpha = alpha_values(i);
        tau = tau_values(j);
        
        % Generate Gaussian Random Field with specified alpha and tau
        U = GRF(alpha, tau, s);
        U(U >= 0) = 1;
        U(U < 0) = -1;
        
        % Plot in the correct subplot location
        subplot(5, 5, (i-1)*5 + j);
        imagesc(U);
        axis equal tight;
        set(gca, 'XTick', [], 'YTick', []);   % Remove tick marks for clarity
        
        % Add title with alpha and tau values
        title(sprintf('\\alpha = %.1f, \\tau = %.1f', alpha, tau), 'FontSize', 8);
    end
end

% Function to generate Gaussian Random Field
function U = GRF(alpha, tau, s)
    % Random variables in KL expansion
    xi = normrnd(0, 1, s);
    
    % Define the (square root of) eigenvalues of the covariance operator
    [K1, K2] = meshgrid(0:s-1, 0:s-1);
    coef = tau^(alpha-1) .* (pi^2 * (K1.^2 + K2.^2) + tau^2).^(-alpha/2);
    
    % Construct the KL coefficients
    L = s * coef .* xi;
    L(1,1) = 0;  % Ensure the mean is zero
    
    % Apply inverse DCT to obtain the field in spatial domain
    U = idct2(L);
end
