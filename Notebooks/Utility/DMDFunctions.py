import numpy as np

def DMD(X, dX, r, time):
    """Apply Dynamic Mode Decomposition (DMD)
    
        Parameters
    ----------
    X : array
        State matrix size: features x time.
    dX : array
        Time shifted state matrix.
    time : array
        Time steps.
    r : int
        Rank truncation.

    Returns
    -------
    U, V, S : array
        Singular Value Decomposition (SVD) of X.
    x_dmd : array
        DMD reconstruction.
    
    """
    dt = time[1] - time[0]
    
    # SVD on the state matrix
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.conj().T
    U = U[:, :r]
    V = V[:, :r]
    S = S[:r]
    
    # Projection of state matrix onto U
    A_tilde = U.conj().T @ dX @ V @ np.linalg.inv(np.diag(S))
    
    # Eigenvalues
    Lambda, W = np.linalg.eig(A_tilde)
    
    # Eigenvectors
    Phi = dX @ V @ np.linalg.inv(np.diag(S)) @ W
    
    
    X0 = X[:, 0] # initial conditions

    omega = np.log(Lambda) / dt

    b = np.linalg.pinv(Phi) @ X0


    # Reconstruction
    x_dmd = np.zeros((r, len(time)), dtype=omega.dtype)
    for k in range(len(time)):
        x_dmd[:, k] = b * np.exp(omega * time[k])
    x_dmd = np.dot(Phi, x_dmd)
    
    #x_k = np.zeros((r, len(t_new)), dtype=omega.dtype)
    #for k in range(len(t_new)):
    #    x_k[:, k] = Phi @ (np.diag(Lambda) ** (k)) @ b
    
    return U, S, V, x_dmd   
