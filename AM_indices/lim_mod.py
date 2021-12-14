#=============================================================
# Module for Linear Inverse Model (LIM)
#=============================================================
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

#=============================================================
# class LIM()
# Linear Inverse Model
#=============================================================
class LIM():
    def __init__(self, X, lag, la_method='numpy', eig_method='eig2', verbose=False):
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

        `linalg` libray used to calculate pinv and eig:
        la_method = 'numpy'; 'scipy'; 'torch'

        eig_method options:
        eig1: solve vl and vr simulateneously by calling `eig` in scipy
        eig2: solve vl and vr separately by calling `eig` twice
        pinv: solve vr by calling `eig`, and then solve vl by calling `pinv`
        """

        self.w, self.vl, self.vr, self.b, self.B = self.pop(X, lag, la_method=la_method, eig_method=eig_method, verbose=verbose)

    def pop(self, X, lag, la_method='numpy', eig_method='eig2', verbose=False):

        data_size = len(X)
        C0 = X.T @ X / data_size
        Ct = X[lag:].T @ X[0:data_size-lag] / (data_size-lag)

        Gt = pinv(C0, Ct, la_method=la_method)
        
        w, vl, vr = eig_m(Gt, la_method=la_method, eig_method=eig_method, verbose=verbose)
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
            Xt = (Gt.real @ X.T).T
            y = np.vstack((y, Xt[None, :]))
        
        return y

#=============================================================
# function eig_m()
# modified version of eigenvector decomposition: ``w, vl, vr = eig_m(a)``
#=============================================================
def eig_m(a, la_method=None, eig_method='eig2', verbose=False):
    """
    modified version of eigenvector decomposition:
    ``w, vl, vr = eig_m(a)``

    Factorizes the square matrix `a` into the normalized eigenvector matrix ``vr``, its inverse ``vl.H``, and
    a 1-D array ``w`` of eigenvalues: \\
    ``a @ vr = vr @ np.diag(w)`` or ``a = vr @ np.diag(w) @ vl.H`` \\
    where ``vl`` and ``vr`` are biorthogonal bases, `vr @ vl.H = I`, and `vl.H @ vr = I`.

    `linalg` libray used to calculate pinv and eig:
    la_method = 'numpy'; 'scipy'; 'torch'

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
    
    def eig_lr(a, la_method=None, eig_method=None):
        if la_method == 'numpy':
            import numpy.linalg as la
        elif la_method == 'scipy':
            import scipy.linalg as la
        elif la_method == 'torch':
            import torch
            import torch.linalg as la
        else:
            raise Exception('la_method not defined!')
        
        # perform eigendecomposition
        if eig_method == 'eig1':
            w1, vl1, vr1 = la.eig(a, left=True)    #scipy only
            w, vr = eig_sort(w1, vr1)
            ww, vl = eig_sort(w1, vl1)
    
        elif eig_method == 'eig2':
            if la_method != 'torch':
                w1, vr1 = la.eig(a)
            else:
                w1, vr1 = la.eig(torch.from_numpy(a))
                w1, vr1 = w1.numpy(), vr1.numpy()
            w, vr = eig_sort(w1, vr1)

            if la_method != 'torch':
                w2, vl2 = la.eig(a.conj().T)
            else:
                w2, vl2 = la.eig(torch.from_numpy(a).conj().T)
                w2, vl2 = w2.numpy(), vl2.numpy()
            ww, vl = eig_sort(w2, vl2, sort_by_w=False)
    
        elif eig_method == 'pinv':
            if la_method != 'torch':
                w1, vr1 = la.eig(a)
            else:
                w1, vr1 = la.eig(torch.from_numpy(a))
                w1, vr1 = w1.numpy(), vr1.numpy()
            w, vr = eig_sort(w1, vr1)

            vl = pinv(vr, la_method=la_method).conj().T    # use pinv defined here 

        else:
            raise Exception('The method for eigenvector decomposition is not specified!')

        return w, vl, vr

    w, vl, vr = eig_lr(a, la_method=la_method, eig_method=eig_method)

    # form a biorthogonal system
    # note the transpose operator (.T) in python differs from the operator (') in matlab for a complex
    vlH_vr = vl.conj().T @ vr
    for idx in range(len(w)):
        vl[:, idx] = vl[:, idx]/(vlH_vr[idx, idx].conj())

    if verbose:
        import numpy.linalg as la
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
        plt.contourf(a, cmap='jet')
        plt.title('a')
        plt.colorbar()

        ax = fig.add_subplot(1, 2, 2)
        plt.contourf((a-a_c).real, cmap='jet')
        plt.title('a - vr @ diag(w) @vlH')
        plt.colorbar()
        
    return w, vl, vr

def pinv(C0, Ct=None, la_method=None, inv_method='pinv'):
        """
        la_method = 'numpy'; 'scipy'; 'torch'
        inv_method = 'pinv'; 'lstsq'
        """

        if la_method == 'numpy':
            import numpy.linalg as la
            print(f'rank of inverted matrix ={la.matrix_rank(C0)}')
            rcond = np.finfo(C0.dtype).eps * max(C0.shape)
            if inv_method == 'pinv':
                C0_inv = la.pinv(C0, rcond=rcond)
                if Ct is None:
                    Gt = C0_inv
                else:
                    Gt = Ct @ C0_inv
            else:
                if Ct is None:
                    Gt = la.lstsq(C0.T, np.eye(len(C0), dtype=C0.dtype), rcond)[0].T
                else:
                    Gt = la.lstsq(C0.T, Ct.T, rcond)[0].T

        elif la_method == 'scipy':
            import scipy.linalg as la
            print(f'rank of inverted matrix ={np.linalg.matrix_rank(C0)}')
            rcond = np.finfo(C0.dtype).eps * max(C0.shape)
            if inv_method == 'pinv':
                C0_inv = la.pinv(C0, rtol=rcond)
                if Ct is None:
                    Gt = C0_inv
                else:
                    Gt = Ct @ C0_inv
            else:
                if Ct is None:
                    Gt = la.lstsq(C0.T, np.eye(len(C0), dtype=C0.dtype), rcond)[0].T
                else:
                    Gt = la.lstsq(C0.T, Ct.T, rcond)[0].T
        
        elif la_method == 'torch':
            import torch
            import torch.linalg as la
            print(f'rank of inverted matrix ={la.matrix_rank(torch.from_numpy(C0))}')
            rcond = torch.finfo(torch.from_numpy(C0).dtype).eps * max(C0.shape)
            if inv_method == 'pinv':
                C0_inv = la.pinv(torch.from_numpy(C0), rcond=rcond).numpy()
                if Ct is None:
                    Gt = C0_inv
                else:
                    Gt = Ct @ C0_inv
            else:
                if Ct is None:
                    Gt = la.lstsq(torch.from_numpy(C0).T, torch.from_numpy(np.eye(len(C0), dtype=C0.dtype)).T, rcond=rcond, driver='gelsd')[0].numpy().T
                else:
                    Gt = la.lstsq(torch.from_numpy(C0).T, torch.from_numpy(Ct).T, rcond=rcond, driver='gelsd')[0].numpy().T
            
        else:
            raise Exception('la_method not defined!')

        return Gt


def cal_Gt(X, lag, la_method='numpy', verbose=False):
    data_size = len(X)
    C0 = X.T @ X / data_size
    Ct = X[lag:].T @ X[0:data_size-lag] / (data_size-lag)
    Gt = pinv(C0, Ct, la_method=la_method)

    if verbose:
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(1, 2, 1)
        plt.contourf(Gt, cmap='jet')
        plt.title('Gt')
        plt.colorbar()

    return Gt

def test_eig_m(X, lag, la_method='numpy'):
        Gt = cal_Gt(X, lag, la_method=la_method)
        if la_method == 'scipy':
            w1, vl1, vr1 = eig_m(Gt, la_method=la_method, eig_method='eig1', verbose=True)
        w2, vl2, vr2 = eig_m(Gt, la_method=la_method, eig_method='eig2', verbose=True)
        w3, vl3, vr3 = eig_m(Gt, la_method=la_method, eig_method='pinv', verbose=True)

# Gt = cal_Gt(y, 5, verbose=True)
# test_eig_m(y, 5, la_method='torch')
