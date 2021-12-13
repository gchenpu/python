#=============================================================
# Module for Linear Inverse Model (LIM)
#=============================================================
import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt

#=============================================================
# function eig_m()
# modified version of eigenvector decomposition: ``w, vl, vr = eig_m(a)``
#=============================================================
def eig_m(a, method='eig2', verbose=False):
    """
    modified version of eigenvector decomposition:
    ``w, vl, vr = eig_m(a)``

    Factorizes the square matrix `a` into the normalized eigenvector matrix ``vr``, its inverse ``vl.H``, and
    a 1-D array ``w`` of eigenvalues: \\
    ``a @ vr = vr @ np.diag(w)`` or ``a = vr @ np.diag(w) @ vl.H`` \\
    where ``vl`` and ``vr`` are biorthogonal bases, `vr @ vl.H = I`, and `vl.H @ vr = I`.

    method options:
    eig1: solve vl and vr simulateneously by calling `eig` in scipy
    eig2: solve vl and vr separately by calling `eig` twice
    pinv: solve vr by calling `eig`, and then solve vl by calling `pinv`
    """

    # function to sort eigenvectors v by the decending order of eigenvalues w, with the option to sort by w.congj()
    def eig_sort(w, v, sort_by_w=True):
        if sort_by_w:
            idx_w_sort = w.argsort()[::-1]
        else:
            idx_w_sort = w.conj().argsort()[::-1]    # sort by w.conj()

        w_sort = np.zeros_like(w)
        v_sort = np.zeros_like(v)
        for idx, idx_sort in enumerate(idx_w_sort):
            w_sort[idx] = w[idx_sort]
            v_sort[:, idx] = v[:, idx_sort]

        return w_sort, v_sort

    # perform eigendecomposition
    if method == 'eig1':
        w1, vl1, vr1 = la.eig(a, left=True)
        w, vr = eig_sort(w1, vr1)
        _, vl = eig_sort(w1, vl1)
    
    elif method == 'eig2':
        w1, vr1 = la.eig(a)
        w, vr = eig_sort(w1, vr1)

        w2, vl2 = la.eig(a.conj().T)
        _, vl = eig_sort(w2, vl2, sort_by_w=False)
    
    elif method == 'pinv':
        w1, vr1 = la.eig(a)
        w, vr = eig_sort(w1, vr1)

        vl = la.pinv(vr).conj().T
    
    else:
        raise Exception('The method for eigenvector decomposition is not specified!')

    # form a biorthogonal system
    # note the transpose operator (.T) in python differs from the operator (') in matlab for a complex
    vlH_vr = vl.conj().T @ vr
    for idx in range(len(w)):
        vl[:, idx] = vl[:, idx]/(vlH_vr[idx, idx].conj())

    if verbose:
        # To ensure the biorthogonality, vr is normalized, but vl is not normalized in the modified version 
        vrH_vr = vr.conj().T @ vr
        print(f"diagonal(vrH_vr)=\n{np.diagonal(vrH_vr).real}")

        vlH_c = vl.conj().T @ vr @ vl.conj().T
        print(f"norm(vlH - vlH @ vr @vlH) = {la.norm(vl.conj().T-vlH_c)}")

        vr_c = vr @ vl.conj().T @ vr
        print(f"norm(vr - vr @ vlH @vr) = {la.norm(vr-vr_c)}")

        a_c = vr @ np.diag(w) @ vl.conj().T
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
        
    return w, vl, vr

def cal_Gt(X, lag):
        data_size = len(X)
        C0 = X.T @ X / data_size
        Ct = X[lag:].T @ X[0:data_size-lag] / (data_size-lag)

        C0_inv = la.pinv(C0)    # pseudo-inverse for a singular matrix
        Gt = Ct @ C0_inv
        return Gt

def test_eig_m(X, lag):
        Gt = cal_Gt(X, lag) 
        w1, vl1, vr1 = eig_m(Gt, method='eig1', verbose=True)
        w2, vl2, vr2 = eig_m(Gt, method='eig2', verbose=True)
        w3, vl3, vr3 = eig_m(Gt, method='pinv', verbose=True)

#=============================================================
# class LIM()
# Linear Inverse Model
#=============================================================
class LIM():
    def __init__(self, X, lag, method='eig2', verbose=False):
        """
        POP analysis described in Penland (1989) 
        X(t, x): input data, where t is time, and x is space
        lag: lag time used to compute Ct

        The propagation matrix is calculated as
        `Gt = Ct @ C0_inv`.

        The eigenvector decomposition of the propagation matrix (using the function `eig_m`) is 
        `Gt @ vr = vr @ np.diag(w)` or `Gt = vr @ np.diag(w) @ vl.H`, \\
        where vl and vr are biorthogonal bases, `vr @ vl.H = I`, and `vl.H @ vr = I`.

        The projection of `x` in the eigenvector space is
        `pc(t, eig) = (vl.H @ X.H).H = X @ vl`, \\
        and the reconstruction in the physical space is
        `X(t, x) = (vr @ pc.H).H = pc @ vr.H`.

        method options:
        eig1: solve vl and vr simulateneously by calling `eig` in scipy
        eig2: solve vl and vr separately by calling `eig` twice
        pinv: solve vr by calling `eig`, and then solve vl by calling `pinv`
        """

        self.w, self.vl, self.vr, self.b, self.B = self.pop(X, lag, method=method, verbose=verbose)

    def pop(self, X, lag, method='eig2', verbose=False):
        data_size = len(X)
        C0 = X.T @ X / data_size
        Ct = X[lag:].T @ X[0:data_size-lag] / (data_size-lag)

        C0_inv = la.pinv(C0)    # pseudo-inverse for a singular matrix
        Gt = Ct @ C0_inv

        w, vl, vr = eig_m(Gt, method=method, verbose=verbose)
        b = np.log(w)/lag
        B = vr @ np.diag(b) @ vl.conj().T
        if verbose:
            print(f"POP e-folding timescales =\n {-1/b.real}")

        return w, vl, vr, b, B

    def forecast(self, X, lag):
        """
        X(batch, x): batch of initial conditions
        lag: time lags to be forecasted
        return y(lag, batch, x): batch forecasts at time lag starting from X
        """
        
        y = np.zeros((0, *X.shape))
        for idx in range(len(lag)):
            Gt = self.vr @ np.diag(np.exp(self.b * lag[idx])) @ self.vl.conj().T
            Xt = (Gt @ X.T).T
            y = np.vstack((y, Xt[None, :]))
        
        return y.real
