import numpy as np

def DMD(X,Xprime,r):

    """
    Taken from:
        - https://github.com/dynamicslab/databook_python/blob/master/CH07/CH07_SEC02_DMD_Cylinder.ipynb
    """
    
    U,Sigma,VT = np.linalg.svd(X,full_matrices=False) # Step 1
    Ur = U[:,:r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r,:]
    
    Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ Xprime @ VTr.T).T).T # Step 2
    
    Lambda, W = np.linalg.eig(Atilde) # Step 3
    Lambda = np.diag(Lambda)
    
    Phi = Xprime @ np.linalg.solve(Sigmar.T,VTr).T @ W # Step 4
    alpha1 = Sigmar @ VTr[:,0]
    b = np.linalg.solve(W @ Lambda,alpha1)
    
    return Phi, Lambda, b

def DMDprediction(Phi, Lambda, b, time, r):
    dt = time[1] - time[0]

    logLambda = np.copy(Lambda)

    for i in range(len(logLambda)):
        for j in range(len(logLambda[0])):
            if(logLambda[i,j] != 0):
                logLambda[i,j] = np.log(logLambda[i,j]) / dt

    X_pred = np.zeros((r, len(time)), dtype=complex)

    for i in range(len(time)):
        s1 = logLambda * time[i]
        s2 = np.diag(np.exp(s1))
        s3 = b * s2
        s4 = Phi @ s3
        X_pred[:, i] = s4

    return X_pred

def DMDtizio(X, dX, r, time):
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