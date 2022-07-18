def LotkaVolterra(t, x0, b, p, d, r):
    x, y = x0
    return [(b - p * y) * x , (r * x - d) * y]