def lorenz_deriv(t, k, sigma, beta, rho):
    x, y, z = k
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]