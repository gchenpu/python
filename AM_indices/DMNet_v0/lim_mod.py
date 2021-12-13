#=============================================================
# Module for Linear Inverse Model (LIM)
#=============================================================
import numpy as np

import matplotlib.pyplot as plt

#=============================================================
# function svds_trunc(a, k=None)
# truncating 'a' by the first 'k' svds
#=============================================================
def svds_trunc(a, k=None, verbose=True):
    """
    function svds_trunc(a, k=None):
    truncating 'a' by the first 'k' svds
    """

    if k is None:
        n_svds = min(a.shape)-1
    else:
        n_svds = min(min(a.shape)-1, k)
    
    if verbose:
        print(f'Truncating the input data to {n_svds} SVDs')

    import scipy.sparse.linalg as la
    U, s, Vh = la.svds(a, k=k)

    return U @ np.diag(s) @ Vh

#=============================================================
# function eig_m()
# modified eigendecomposition from scipy.linalg.eig
#=============================================================
def eig_m(a, verbose=False):
    """
    modified version of eigenvector decomposition:
    w, vl, vr = eig_m(a)

    Factorizes the square matrix `a` into the normalized eigenvector matrix ``vr`` and its inverse ``vl.H``, and
    a 1-D array ``w`` of eigenvalues such that ``a == vr @ np.diag(w) @ vl.H``.
    """

    # standard eigendecomposition from scipy.linalg
    import scipy.linalg as la
    w, vl, vr = la.eig(a, left=True)

    # sort eigenvectors by eigenvalues and form a biorthogonal system
    idx_w_sort = w.argsort()[::-1]
    vlH_vr = vl.conj().T @ vr

    w_m = np.zeros_like(w)
    vl_m = np.zeros_like(vl)
    vr_m = np.zeros_like(vr)
    for idx in range(len(w)):
        idx_sort = idx_w_sort[idx]
        w_m[idx] = w[idx_sort]
        vl_m[:, idx] = vl[:, idx_sort]/(vlH_vr[idx_sort, idx_sort].conj())    
        vr_m[:, idx] = vr[:, idx_sort]
    
    if verbose:
        # To ensure the biorthogonality, vr is normalized, but vl is not normalized in the modified version 
        vrH_vr = vr_m.conj().T @ vr_m
        print(f"diagonal(vrH_vr)=\n{np.diagonal(vrH_vr).real}")

        vlH_c = vl_m.conj().T @ vr_m @ vl_m.conj().T
        print(f"norm(vlH - vlH @ vr @vlH) = {la.norm(vl_m.conj().T-vlH_c)}")

        vr_c = vr_m @ vl_m.conj().T @ vr_m
        print(f"norm(vr - vr @ vlH @vr) = {la.norm(vr_m-vr_c)}")

        a_c = vr_m @ np.diag(w_m) @ vl_m.conj().T
        print(f"norm(a - vr @ diag(w) @vlH) = {la.norm(a-a_c)}")

        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(1, 2, 1)
        plt.contourf(a.real, cmap='jet')
        plt.title('a')
        plt.colorbar()

        ax = fig.add_subplot(1, 2, 2)
        plt.contourf((a-a_c).real, cmap='jet')
        plt.title('a - vr @ diag(w) @vlH')
        plt.colorbar()
        
    return w_m, vl_m, vr_m

#=============================================================
# class LIM()
# Linear Inverse Model
#=============================================================
class LIM():
    def __init__(self, X, lag, verbose=False):
        """
        POP analysis described in Penland (1989) 
        X(t, x): input data, where t is time, and x is space
        lag: lag time used to compute Ct

        `Gt = Ct @ C0_inv` \\
        The eigenvector decomposition is: 
        `Gt @ vr = vr @ np.diag(w)` \\
        or `Gt = vr @ np.diag(w) @ vl.H` \\
        where vl and vr are biorthogonal bases, `vr @ vl.H = I`, and `vl.H @ vr = I`

        `pc(t, eig) = (vl.H @ X.H).H = X @ vl` \\
        `X(t, x) = (vr @ pc.H).H = pc @ vr.H`
        """

        self.w, self.vl, self.vr, self.b, self.B = self.pop(X, lag, verbose=verbose)

    def pop(self, X, lag, verbose=False):
        data_size = len(X)
        C0 = X.T @ X / data_size
        Ct = X[lag:].T @ X[0:data_size-lag] / (data_size-lag)

        import scipy.linalg as la
        # C0_inv = la.inv(C0)
        C0_inv = la.pinv(C0)    # pseudo-inverse for a singular matrix
        Gt = Ct @ C0_inv

        w, vl, vr = eig_m(Gt, verbose=verbose)
        b = np.log(w)/lag
        B = vr @ np.diag(b) @ vl.conj().T
        if verbose:
            print(f"POP e-folding timescales =\n {-1/b.real}")

        return w, vl, vr, b, B

    def forecast(self, X, lag):
        """
        X(batch, x): batch of initial conditions
        lag: time lags to be forecasted
        y(lag, batch, x): batch forecasts at time lag starting from X
        """
        
        y = np.zeros((0, *X.shape))
        for idx in range(len(lag)):
            Gt = self.vr @ np.diag(np.exp(self.b * lag[idx])) @ self.vl.conj().T
            Xt = (Gt @ X.T).T
            y = np.vstack((y, Xt[None, :]))
        
        return y.real

#=============================================================
# class LIMOptim()
# Linear Inverse Model
# truncating input data to first k svds with a relative threshold for norm(model.B)
#=============================================================
class LIMOptim():
    def __init__(self, X, lag, lag_val=None, B_threshold=0.5, verbose=False):
        """
        X(t, x): input data, where t is time, and x is space
        lag: lag time used to compute Ct
        lag_val: lag time used to evaluate predictions, optional
        B_threshold: relative threshold for norm(self.model.B)
        """

        self.k = self.optim_k(X, lag, lag_val=lag_val, B_threshold=B_threshold, verbose=verbose)
        if self.k < X.shape[1]:
            X_trunc = svds_trunc(X, k=self.k, verbose=True)
        else:
            print(f'No svd truncation')
            X_trunc = X
        self.model = LIM(X_trunc, lag=lag, verbose=True)
    
    def optim_k(self, X, lag, lag_val=None, B_threshold=0.5, verbose=False):
        """
        finding the optimal truncations from svd_min to svd_max-1
        """

        svd_min = 5             # min number of svds tested
        svd_max = X.shape[1]    # max number of svds tested
        B_norm = np.zeros(svd_max-svd_min+1)     # B matrix
        Ct_norm = np.zeros(svd_max-svd_min+1)    # Ct errors

        t = np.linspace(0, len(X), len(X)+1)
        
        for svd_idx in range(svd_max-svd_min+1):
            if svd_idx < svd_max-svd_min:
                X_trunc = svds_trunc(X, k=svd_idx+svd_min, verbose=False)
            else:
                # no truncation
                X_trunc = X
            model = LIM(X_trunc, lag=lag)
            B_norm[svd_idx] = np.linalg.norm(model.B)
            
            if lag_val is not None:
                C0 = X.T @ X / len(X)
                Ctf = model.forecast(C0, t[:lag_val+1])
                Ct = X[lag_val:].T @ X[:-lag_val] / (len(X)-lag_val)
                Ct_norm[svd_idx] = np.linalg.norm(Ct - Ctf[lag_val,:,:].T)
        
        B_norm2 = np.array([B_norm[idx]/B_norm[:idx+1].mean()-1.0 for idx in range(len(B_norm))])
        B_idx = np.argwhere(B_norm2 < B_threshold)[-1][0]
        
        if verbose:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(1, 2, 1)
            plt.plot(np.linspace(svd_min, svd_max, svd_max-svd_min+1), B_norm, '-o')
            plt.plot(svd_min+B_idx, B_norm[B_idx], '-*r')
            # plt.yscale('log')
            plt.title('norm of B')
            plt.xlabel('# of svds retained')
            print(f'norm of B with respect to # of svds ({svd_min},...,{svd_max-1}):\n', B_norm, '\n')

            if lag_val is not None: 
                ax = fig.add_subplot(1, 2, 2)
                plt.plot(np.linspace(svd_min, svd_max, svd_max-svd_min+1), Ct_norm, '-o')
                plt.plot(svd_min+B_idx, Ct_norm[B_idx], '-*r')
                plt.title('norm of prediction error')
                plt.xlabel('# of svds retained')
                print('norm of prediction error:\n', Ct_norm, '\n')

        return svd_min + B_idx
