function U = GRF3D(alpha, tau, s)
    % GRF3D generates a 3D Gaussian Random Field (GRF) using the Karhunen-Loève (KL) expansion.
    
    % Random variables in the KL expansion
    xi = normrnd(0, 1, [s, s, s]);
    
    % Define the (square root of) eigenvalues of the covariance operator
    [K1, K2, K3] = ndgrid(0:s-1, 0:s-1, 0:s-1);
    coef = tau^(alpha-1) .* (pi^2 * (K1.^2 + K2.^2 + K3.^2) + tau^2).^(-alpha/2);
    
    % Construct the KL coefficients
    L = s * coef .* xi;
    L(1,1,1) = 0; % Ensure mean is zero in the GRF
    
    % Inverse discrete cosine transform (3D)
    U = idctn(L);
end

function U = idctn(L)
    % Perform 3D inverse discrete cosine transform
    U = idct(idct(idct(L, [], 1), [], 2), [], 3);
end