#=============================================================
# Module for Linear Inverse Model (LIM) using PyTorch and Numpy
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
    def __init__(self, X, hyp_param, verbose=False):
        """
        Input: X(years, days, x), where years and days are time (denoted by t), and x is space
        hyper_parm: `lag_time` is the lag time used to compute Ct
                    `method` specifies `LIM` or `DMD` to solve for the dynamic modes

        The propagation matrix is calculated as
        `Gt = Ct @ C0_inv`.

        The eigenvector decomposition of the propagation matrix (using the function `eig_m`) is 
        `Gt @ vr = vr @ torch.diag(w)` or `Gt = vr @ torch.diag(w) @ vl.H`, \\
        where vl and vr are biorthogonal bases, `vr @ vl.H = I`, and `vl.H @ vr = I`.

        The projection of `X` in the eigenvector space is
        `pc(t, eig) = (vl.H @ X.H).H = X @ vl`, \\
        and the reconstruction in the physical space is
        `X(t, x) = (vr @ pc.H).H = pc @ vr.H`.

        method in the function `eig_m`:
        eig2: solve vl and vr separately by calling `eig` twice
        pinv: solve vr by calling `eig`, and then solve vl by calling `pinv`
        """

        if hyp_param['lim']['method'] == 'LIM':
            self.LIM(X, lag=hyp_param['lim']['lag_time'], LIM_param=hyp_param['lim']['LIM'], verbose=verbose)
        elif hyp_param['lim']['method'] == 'DMD':
            self.DMD(X, lag=hyp_param['lim']['lag_time'], DMD_param=hyp_param['lim']['DMD'], verbose=verbose)
        else:
            raise Exception('The method for LIM is not specified!')            

    def LIM(self, X, lag, LIM_param, verbose=False):
        """POP analysis described in Penland (1989)
        """
        
        if LIM_param['r_optimal'] is None:  # truncating X to the first `r_optimal` SVDs if specified
            X_trunc = X
        else:
            X_flat = X.reshape(-1, X.shape[2])
            X_flat_trunc = svds_trunc(X_flat, k=LIM_param['r_optimal'])
            X_trunc = X_flat_trunc.reshape(*X.shape)

        C0 = cov_lag(X_trunc, lag_time=0)
        Ct = cov_lag(X_trunc, lag_time=lag)
        Gt = pinv(C0, Ct, inv_method='pinv')

        self.w, self.vl, self.vr = eig_m(Gt, eig_method='eig2', verbose=False)
        self.b = torch.log(self.w)/lag
        self.B = (self.vr @ torch.diag(self.b) @ self.vl.conj().T).real

        if verbose:
            print(f"POP e-folding timescales =\n {-1/self.b.real}")

    def DMD(self, X, lag, DMD_param, verbose=False):
        """DMD analysis for sequential data
        """
        def DMD_r(U, s, V, r, verbose=False):
            # DMD for the truncation of X to rth SVD 
            Ct_hat = cov_lag(V[:, :, :r], lag_time=lag) * (len_day-lag) * len_year
            Ct_hat_s = torch.diag(s[:r]) @ Ct_hat.to(s.dtype) @ torch.diag(1.0/s[:r])
            self.w, vl_s, vr_s = eig_m(Ct_hat_s, eig_method=DMD_param['eig_method'], verbose=False)

            self.U = U[:, :r]
            self.vl_s = vl_s
            self.vr_s = vr_s
            self.vl = U[:, :r].to(vr_s.dtype) @ vl_s
            self.vr = U[:, :r].to(vr_s.dtype) @ vr_s
            self.b = torch.log(self.w)/lag
            self.B = (self.vr @ torch.diag(self.b) @ self.vl.conj().T).real
            B_norm = la.vector_norm(self.B, ord=ord)**ord

            t = torch.linspace(0, forecast_time, forecast_time+1)    # t=0 included
            Xf = self.forecast(X[:, :-forecast_time, :], t)
            X_err = la.norm(X[:, forecast_time:, :] - Xf[:, forecast_time, :, :])**2.0/(Xf.shape[0]*Xf.shape[2]*Xf.shape[3])    # only consider t=forecast_time
            # X_err = 0
            # for n in range(forecast_time+1):
            #     X_err += la.norm(X[:, n:n+X.shape[1]-forecast_time, :] - Xf[:, n, :, :])**2.0
            # X_err /= Xf.numel()
            # print(B_norm, X_err)

            U_r, s_r, V_r = self.optimals(t, method='svd')

            return B_norm, X_err, s_r[-1]
        
        alpha = DMD_param['r_alpha']
        ord = DMD_param['r_ord']
        r_opt = DMD_param['r_optimal']
        forecast_time = DMD_param['r_forecast']

        # SVD for U, s, V
        len_year, len_day, len_x = X.shape
        Xh_flat = X.reshape(-1, len_x).T
        U, s, Vh_flat = la.svd(Xh_flat, full_matrices=False)
        V = Vh_flat.T.reshape(len_year, len_day, len_x)

        r_min = 2   # minimum SVDs to consider
        rcond = torch.finfo(Xh_flat.dtype).eps * max(*Xh_flat.shape)
        r_max = len(s[s>rcond*s[0]])    # rank of Xh_flat
        s_prct = torch.zeros_like(s)
        for i in range(len(s)):
            s_prct[i] = torch.sum(s[:i+1]*s[:i+1])/torch.sum(s*s)*100
        # if verbose:
        #     print(f'Singular Values of X (rank={r_max}): \n {s[:r_max]}')
        #     print(f'Percentage of Variance: \n {s_prct[:r_max]}')

        if r_opt is None:
            # solve r_opt that minimizes X_err + alpha*B_norm
            B_norm_r = torch.empty((r_max-r_min+1, ))
            X_err_r = torch.empty_like(B_norm_r)
            s_r = torch.empty_like(B_norm_r)
            for r in range(r_min, r_max+1):
                B_norm_r[r-r_min], X_err_r[r-r_min], s_r[r-r_min] = DMD_r(U, s, V, r, verbose=verbose)

            rr = torch.linspace(r_min, r_max, r_max-r_min+1)
            loss_r = X_err_r + alpha*B_norm_r
            r_opt = torch.argmin(loss_r).item() + r_min
            if verbose:
                fig=plt.figure(figsize=(12,5))
                fig.add_subplot(1,2,1)
                plt.plot(rr, X_err_r, label=r'$||X-\hat{X}||^2$')
                plt.plot(rr, loss_r, label=r'$||X-\hat{X}||^2$+$\alpha$||B||')
                plt.plot(rr[r_opt-r_min], loss_r[r_opt-r_min], '-o')
                plt.xlabel('truncations (r)')
                plt.legend()

                fig.add_subplot(1,2,2)
                plt.plot(rr, s_r)
                plt.plot(rr[r_opt-r_min], s_r[r_opt-r_min], '-o')
                plt.xlabel('truncations (r)')
                plt.ylabel(f'amplification')

        B_norm_opt, X_err_opt, s_opt = DMD_r(U, s, V, r_opt, verbose=verbose)
        if verbose:
            print(f'r_opt={r_opt}: % of var={s_prct[r_opt-1]:>8f}, B_norm={B_norm_opt:>8f}, X_err={X_err_opt:>8f}, s={s_opt:>8f}')
            print(f"POP e-folding timescales =\n {-1/self.b.real}")
            
    def forecast(self, X, forecast_time):
        """
        X(years, batch, x): batch of initial conditions
        forecast_time: time lags to forecast; the first of forecast_time is t=0
        return y(years, forecast_time, batch, x): batch forecasts at forecast_time starting from X
        """

        if X.ndim == 2:
            X = X[None, :]  # 1st dim of `X` is size=1
        
        len_forecast = len(forecast_time)
        len_year, len_day, len_x = X.shape
        y = torch.empty((len_year, len_forecast, len_day, len_x))
        for idx in range(len_forecast):
            Gt = (self.vr @ torch.diag(torch.exp(self.b * forecast_time[idx])) @ self.vl.conj().T).real
            for n in range(len_year):
                y[n,idx,:,:] = (Gt @ torch.squeeze(X[n,:,:]).T).T
            
        return y
    
    def optimals(self, forecast_time, method='svd', N=None):
        """
        solve for `optimals` of exp(b t) at a given `t=forecast_time`
        forecast_time: time lags to forecast
        `svd`: U for the pattern at t=forecast_time, Vh for optimals at t=0
        `eig`: vr for optimals at t=0
        """

        len_forecast = len(forecast_time)
        if method == 'svd':
            U1 = torch.empty((len_forecast, len(self.vr)))
            s1 = torch.empty((len_forecast, ))
            V1 = torch.empty_like(U1)
            for idx in range(len_forecast):
                Gt = (self.vr @ torch.diag(torch.exp(self.b * forecast_time[idx])) @ self.vl.conj().T).real
                U, s, Vh = la.svd(Gt)
                U1[idx, :], s1[idx], V1[idx, :] = U[:, 0], s[0], Vh[0, :]

            return U1, s1, V1

        elif method =='eig':
            if N is None:
                N = torch.eye(len(self.vr))
            else:
                eps = 1e-9
                N = N + torch.eye(len(N))*eps
            
            w1 = torch.empty((len_forecast, ))
            vr1 = torch.empty((len_forecast, len(self.vr)))
            for idx in range(len_forecast):
                Gt = (self.vr @ torch.diag(torch.exp(self.b * forecast_time[idx])) @ self.vl.conj().T).real
                w, vr = la.eig(Gt.T @ N @ Gt)
                if forecast_time[idx] == 0:
                    w1[idx], vr1[idx, :] = w[-1].real, vr[:, -1].real  # only for t = 0
                else:   
                    w1[idx], vr1[idx, :] = w[0].real, vr[:, 0].real

            return w1, vr1

        else:
            raise Exception('The method for finding optimals is not specified!')

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
        n_svds = min(a.shape)
    else:
        n_svds = min(min(a.shape), k)
    
    if verbose:
        print(f'Truncating the input data to {n_svds} SVDs')

    U, s, Vh = la.svd(a)
    print(U.shape, s.shape, Vh.shape)

    return U[:, :n_svds] @ torch.diag(s[:n_svds]) @ Vh[:n_svds, :]

#=============================================================
def cov_lag(X, lag_time, X2=None):
    """ Input: X(years, days, x), optional: X2(years, days, x2), where years and days are time, and x and x2 are space
        First calculate the covariance in the dimension of `days`, and then take the average along the dimension of `years`
        return Ct(x, x2)
    """

    if X2 is None:
        X2 = X

    if X.ndim == 2:
        X = X[:, :, None]   # last dim of 'x' is size=1

    if X2.ndim == 2:
        X2 = X2[:, :, None]   # last dim of 'x' is size=1

    len_year, len_day, len_x = X.shape
    len_x2 = X2.shape[2]

    Ct = torch.empty((len_year, len_x, len_x2))
    for n in range(len_year):
        Ct[n, :, :] = torch.squeeze(X[n, lag_time:, :], 0).T @ torch.squeeze(X2[n, 0:len_day-lag_time, :], 0) / (len_day-lag_time)

    return torch.squeeze(Ct.mean(axis=0))

#=============================================================
def pinv(C0, Ct=None, inv_method='pinv'):
    """ Calculate  `Ct @ C0_inv` or `C0_inv` if Ct=None
        inv_method = 'pinv'; 'lstsq'; 'svd??'
    """

    C0_rank = la.matrix_rank(C0)
    # print(f'rank of inverted matrix ={C0_rank}')
    rcond = torch.finfo(C0.dtype).eps * max(C0.shape)
    if inv_method == 'pinv':
        C0_inv = la.pinv(C0, rcond=rcond)
        if Ct is None:
            Gt = C0_inv
        else:
            Gt = Ct @ C0_inv

    elif inv_method[:3] == 'svd':
        if inv_method[3:] == "":
            k = C0_rank
        else:
            k = min(int(inv_method[3:]), C0_rank)
        U, s, Vh = la.svd(C0)
        s_inv = torch.zeros_like(s)
        s_inv[:k] = 1.0/s[:k]
        C0_inv = Vh.conj().T @ torch.diag(s_inv).to(U.dtype) @ U.conj().T

        if Ct is None:
            Gt = C0_inv
        else:
            Gt = Ct @ C0_inv

    elif inv_method == 'lstsq':
        if Ct is None:
            Gt = la.lstsq(C0.T, torch.eye(len(C0), dtype=C0.dtype), rcond=rcond, driver='gelsd')[0].T
        else:
            Gt = la.lstsq(C0.T, Ct.T, rcond=rcond, driver='gelsd')[0].T
            
    else:
        raise Exception('The method for matrix inversion is not specified!')

    return Gt

#=============================================================
# function eig_m()
# modified version of eigenvector decomposition: ``w, vl, vr = eig_m(a)``
#=============================================================
def eig_m(a, eig_method='eig2', inv_method='pinv', verbose=False):
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
        w_np = w.numpy()    # use the `sort` method in numpy; torch.argsort not implemented for 'ComplexFloat'
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

            vl = pinv(vr, inv_method=inv_method).conj().T    # use pinv defined here 

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
        check_eigs(w, vl, vr, a)
        
    return w, vl, vr
    
def check_eigs(w, vl, vr, a=None):
    """To ensure the biorthogonality of the eigenvectors from eig_m()
       vr is normalized, but vl is not normalized in the modified version 
    """

    vrH_vr = vr.conj().T @ vr
    print(f"diagonal(vrH_vr)=\n{torch.diagonal(vrH_vr).real}")

    vlH_c = vl.conj().T @ vr @ vl.conj().T
    print(f"norm(vlH - vlH @ vr @vlH) = {la.norm(vl.conj().T-vlH_c)}")

    vr_c = vr @ vl.conj().T @ vr
    print(f"norm(vr - vr @ vlH @vr) = {la.norm(vr-vr_c)}")

    if a is not None:
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

#=============================================================
# tests of functions
#=============================================================
def cal_Gt(X, hyp_param, verbose=False):
    """ Input: X(years, days, x), where years and days are time, and x is space
        The propagation matrix is calculated as
        `Gt = Ct @ C0_inv`
    """
    lag = hyp_param['lim']['lag_time']
    C0 = cov_lag(X, lag_time=0)
    Ct = cov_lag(X, lag_time=lag)
    Gt = pinv(C0, Ct, inv_method='pinv')

    if verbose:
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(1, 2, 1)
        plt.contourf(Gt, cmap='jet')
        plt.title('Gt')
        plt.colorbar()

    return Gt

def test_eig_m(X, hyp_param):
        Gt = cal_Gt(X, hyp_param)
        w2, vl2, vr2 = eig_m(Gt, eig_method='eig2', inv_method='pinv', verbose=True)

#=============================================================
# # test of Gt
# Gt = cal_Gt(y, hyp_param, verbose=True)

# # test of eig_m
# test_eig_m(y, hyp_param)

# # set hyp_param
# hyp_param = dict(lim = dict(lag_time = 5,
#                            ),
#                 )
# print(f'hyperpamameters:\n{hyp_param}')

# # forecast
# t = torch.linspace(1, y.shape[1], y.shape[1], dtype=torch.float32)

# model = LIM(y, hyp_param, verbose=False)

# forecast_time = 5
# yf = model.forecast(y[:, :-forecast_time, :], t[:forecast_time+1])
# print(y.shape, yf.shape)

# # optimals
# U, s, Vh = model.optimals(forecast_time=10, method='svd')
# print(U.shape, s.shape, Vh.shape)