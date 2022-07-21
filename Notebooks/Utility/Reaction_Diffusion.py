import numpy as np

def reaction_diffusion(t, uvt, K22, d1, d2, beta, n, N):
    # Calculate u and v terms
    ut = np.reshape(uvt[:N], (n, n))
    vt = np.reshape(uvt[N:], (n, n))
    u = np.real(np.fft.ifft2(ut))
    v = np.real(np.fft.ifft2(vt))
    
    # reaction terms
    u3 = u ** 3
    v3 = v ** 3
    u2v = (u ** 2) * v
    uv2 = u * (v ** 2)
    utrhs = np.reshape((np.fft.fft2(u - u3 - uv2 + beta * u2v + beta * v3)), (N, 1))
    vtrhs = np.reshape((np.fft.fft2(v - v3 - u2v - beta * u3 - beta * uv2)), (N, 1))
    
    rhs = np.concatenate([-d1 * K22 * uvt[:N] + utrhs,
           -d2 * K22 * uvt[N:] + vtrhs])[:, 0]

    return rhs