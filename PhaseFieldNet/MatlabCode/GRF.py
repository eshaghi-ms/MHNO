import numpy as np
from scipy.fftpack import idct


def GRF(alpha, tau, s):
    # Random variables in KL expansion
    xi = np.random.randn(s, s)

    # Define the (square root of) eigenvalues of the covariance operator
    k1, k2 = np.meshgrid(np.arange(s), np.arange(s))
    coef = tau ** (alpha - 1) * (np.pi ** 2 * (k1 ** 2 + k2 ** 2) + tau ** 2) ** (-alpha / 2)

    # Construct the KL coefficients
    l = s * coef * xi
    l[0, 0] = 0

    # 2D inverse discrete cosine transform
    u = idct(idct(l, axis=0, norm='ortho'), axis=1, norm='ortho')
    return u