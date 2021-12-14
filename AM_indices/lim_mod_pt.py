#=============================================================
# Module for Linear Inverse Model (LIM)
#=============================================================
import numpy as np
import torch
from torch import linalg as la

import matplotlib.pyplot as plt

#=============================================================
# class LIM()
# Linear Inverse Model
#=============================================================
class LIM():
    def __init__(self, X, lag, eig_method='eig2', verbose=False):
        """
        POP analysis described in Penland (1989) 
        X(t, x): input data, where t is time, and x is space
        lag: lag time used to compute Ct

        The propagation matrix is calculated as
        `Gt = Ct @ C0_inv`.

        The eigenvector decomposition of the propagation matrix (using the function `eig_m`) is 
        `Gt @ vr = vr @ torch.diag(w)` or `Gt = vr @ torch.diag(w) @ vl.H`, \\
        where vl and vr are biorthogonal bases, `vr @ vl.H = I`, and `vl.H @ vr = I`.

        The projection of `x` in the eigenvector space is
        `pc(t, eig) = (vl.H @ X.H).H = X @ vl`, \\
        and the reconstruction in the physical space is
        `X(t, x) = (vr @ pc.H).H = pc @ vr.H`.

        method options:
        eig2: solve vl and vr separately by calling `eig` twice
        pinv: solve vr by calling `eig`, and then solve vl by calling `pinv`
        """

        self.w, self.vl, self.vr, self.b, self.B = self.pop(X, lag, eig_method=eig_method, verbose=verbose)

    def pop(self, X, lag, eig_method='eig2', verbose=False):

        data_size = len(X)
        C0 = X.T @ X / data_size
        Ct = X[lag:].T @ X[0:data_size-lag] / (data_size-lag)

        Gt = pinv(C0, Ct)

        w, vl, vr = eig_m(Gt, eig_method=eig_method, verbose=verbose)
        b = torch.log(w)/lag
        B = vr @ torch.diag(b) @ vl.conj().T
        if verbose:
            print(f"POP e-folding timescales =\n {-1/b.real}")

        return w, vl, vr, b, B

    def forecast(self, X, lag):
        """
        X(batch, x): batch of initial conditions
        lag: time lags to be forecasted
        return y(lag, batch, x): batch forecasts at time lag starting from X
        """
        
        y = torch.zeros((0, *X.shape))
        for idx in range(len(lag)):
            Gt = self.vr @ torch.diag(torch.exp(self.b * lag[idx])) @ self.vl.conj().T
            Xt = (Gt.real @ X.T).T
            y = torch.vstack((y, Xt[None, :]))
        
        return y

#=============================================================
# function eig_m()
# modified version of eigenvector decomposition: ``w, vl, vr = eig_m(a)``
#=============================================================
def eig_m(a, eig_method='eig2', verbose=False):
    """
    modified version of eigenvector decomposition:
    ``w, vl, vr = eig_m(a)``

    Factorizes the square matrix `a` into the normalized eigenvector matrix ``vr``, its inverse ``vl.H``, and
    a 1-D array ``w`` of eigenvalues: \\
    ``a @ vr = vr @ torch.diag(w)`` or ``a = vr @ torch.diag(w) @ vl.H`` \\
    where ``vl`` and ``vr`` are biorthogonal bases, `vr @ vl.H = I`, and `vl.H @ vr = I`.

    method options:
    eig1: solve vl and vr simulateneously by calling `eig` in scipy
    eig2: solve vl and vr separately by calling `eig` twice
    pinv: solve vr by calling `eig`, and then solve vl by calling `pinv`
    """

    # function to sort eigenvectors v by the decending order of eigenvalues w, with the option to sort by w.congj()
    def eig_sort(w, v, sort_by_w=True):
        w_np = w.numpy()
        if sort_by_w:
            idx_w_sort_np = w_np.argsort()[::-1]
        else:
            idx_w_sort_np = w_np.conj().argsort()[::-1]    # sort by w.conj()
        idx_w_sort = torch.from_numpy(idx_w_sort_np.copy())

        w_sort = torch.zeros_like(w)
        v_sort = torch.zeros_like(v)
        for idx, idx_sort in enumerate(idx_w_sort):
            w_sort[idx] = w[idx_sort]
            v_sort[:, idx] = v[:, idx_sort]

        return w_sort, v_sort

    def eig_lr(a, eig_method=None):
        
        # perform eigendecomposition
        if eig_method == 'eig2':
            w1, vr1 = la.eig(a)
            w, vr = eig_sort(w1, vr1)

            w2, vl2 = la.eig(a.conj().T)
            ww, vl = eig_sort(w2, vl2, sort_by_w=False)
    
        elif eig_method == 'pinv':
            w1, vr1 = la.eig(a)
            w, vr = eig_sort(w1, vr1)

            vl = pinv(vr).conj().T    # use pinv defined here 

        else:
            raise Exception('The method for eigenvector decomposition is not specified!')

        return w, vl, vr

    w, vl, vr = eig_lr(a, eig_method=eig_method)

    # form a biorthogonal system
    # note the transpose operator (.T) in python differs from the operator (') in matlab for a complex
    vlH_vr = vl.conj().T @ vr
    for idx in range(len(w)):
        vl[:, idx] = vl[:, idx]/(vlH_vr[idx, idx].conj())

    if verbose:
        # To ensure the biorthogonality, vr is normalized, but vl is not normalized in the modified version 
        vrH_vr = vr.conj().T @ vr
        print(f"diagonal(vrH_vr)=\n{torch.diagonal(vrH_vr).real}")

        vlH_c = vl.conj().T @ vr @ vl.conj().T
        print(f"norm(vlH - vlH @ vr @vlH) = {la.norm(vl.conj().T-vlH_c)}")

        vr_c = vr @ vl.conj().T @ vr
        print(f"norm(vr - vr @ vlH @vr) = {la.norm(vr-vr_c)}")

        a_c = vr @ torch.diag(w) @ vl.conj().T
        print(f"norm(a - vr @ diag(w) @vlH) = {la.norm(a-a_c)}")

        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(1, 2, 1)
        plt.contourf(a, cmap='jet')
        plt.title('a')
        plt.colorbar()

        ax = fig.add_subplot(1, 2, 2)
        plt.contourf((a-a_c).real, cmap='jet')
        plt.title('a - vr @ diag(w) @vlH')
        plt.colorbar()
        
    return w, vl, vr


def pinv(C0, Ct=None, inv_method='pinv'):
        """
        inv_method = 'pinv'; 'lstsq'
        """
    
        print(f'rank of inverted matrix ={la.matrix_rank(C0)}')
        rcond = torch.finfo(C0.dtype).eps * max(C0.shape)
        if inv_method == 'pinv':
                C0_inv = la.pinv(C0, rcond=rcond)
                if Ct is None:
                    Gt = C0_inv
                else:
                    Gt = Ct @ C0_inv
        else:
                if Ct is None:
                    Gt = la.lstsq(C0.T, torch.eye(len(C0), dtype=C0.dtype), rcond=rcond, driver='gelsd')[0].T
                else:
                    Gt = la.lstsq(C0.T, Ct.T, rcond=rcond, driver='gelsd')[0].T
            
        return Gt


def cal_Gt(X, lag, verbose=False):
    data_size = len(X)
    C0 = X.T @ X / data_size
    Ct = X[lag:].T @ X[0:data_size-lag] / (data_size-lag)
    Gt = pinv(C0, Ct)

    if verbose:
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(1, 2, 1)
        plt.contourf(Gt, cmap='jet')
        plt.title('Gt')
        plt.colorbar()

    return Gt


def test_eig_m(X, lag):
        Gt = cal_Gt(X, lag)
        w2, vl2, vr2 = eig_m(Gt, eig_method='eig2', verbose=True)
        w3, vl3, vr3 = eig_m(Gt, eig_method='pinv', verbose=True)

#Gt = cal_Gt(y, 5, verbose=True)
#test_eig_m(y, 5)

