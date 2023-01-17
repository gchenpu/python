#=============================================================
# Module for Linear Inverse Model (LIM) using PyTorch and Numpy
#=============================================================
import numpy as np
import torch
from torch import linalg as la

import matplotlib.pyplot as plt

#=============================================================
# class LIM()
# Linear Inverse Model(LIM) or POP analysis described in Penland (1989)
#=============================================================
class LIM():
    def __init__(self, X, hyp_param, verbose=False):
        """
        Input: X(years, days, x), where years and days are time (denoted by t), and x is space
        hyper_parm: `lag_time` is the lag time used to compute Ct
                    `eig_method` = 'eig2' or `eig_method` = 'pinv'

        The propagation matrix is calculated as
        `Gt = Ct @ C0_inv`.

        The eigenvector decomposition of the propagation matrix (using the function `eig_m`) is 
        `Gt @ vr = vr @ torch.diag(w)` or `Gt = vr @ torch.diag(w) @ vl.H`, \\
        where vl and vr are biorthogonal bases, `vr @ vl.H = I`, and `vl.H @ vr = I`.

        The projection of `X` in the eigenvector space is
        `pc(t, eig) = (vl.H @ X.H).H = X @ vl`, \\
        and the reconstruction in the physical space is
        `X(t, x) = (vr @ pc.H).H = pc @ vr.H`.        
        """

        self.C0 = cov_lag(X, lag_time=0)

        # SVD for U, s, V
        len_year, len_day, len_x = X.shape
        Xh_flat = X.reshape(-1, len_x).T
        U, s, Vh_flat = la.svd(Xh_flat, full_matrices=False)    # Xh_flat = U @ torch.diag(s) @ Vh_flat
        V = Vh_flat.T.reshape(len_year, len_day, len_x)     # len_x for the indices of SVDs

        r_min = 2   # minimum SVDs to consider
        rcond = torch.finfo(Xh_flat.dtype).eps * max(*Xh_flat.shape)
        r_max = len(s[s>rcond*s[0]])    # rank of Xh_flat
        s_prct = torch.zeros_like(s)
        for i in range(len(s)):
            s_prct[i] = torch.sum(s[:i+1]*s[:i+1])/torch.sum(s*s)*100
        # if verbose:
        #     print(f'Singular Values of X (rank={r_max}): \n {s[:r_max]}')
        #     print(f'Percentage of Variance: \n {s_prct[:r_max]}')

        r_opt = hyp_param['lim']['r_optimal']
        if r_opt is None:
            # solve r_opt that minimizes err + alpha*norm_r
            norm_r = torch.empty((r_max-r_min+1, ))
            err_r = torch.empty_like(norm_r)
            R2_r = torch.empty_like(norm_r)
            s_r = torch.empty_like(norm_r)
            for r in range(r_min, r_max+1):
                norm_r[r-r_min], err_r[r-r_min], R2_r[r-r_min], s_r[r-r_min] = self.LIM_r(U, s, V, r, X, hyp_param, verbose=verbose)

            rr = torch.arange(r_min, r_max+1)
            loss_r = err_r + norm_r*hyp_param['lim']['alpha']
            r_opt = torch.argmin(loss_r).item() + r_min
            if verbose:
                fig=plt.figure(figsize=(15,4.5))
                fig.add_subplot(1,3,1)
                plt.plot(rr, err_r, label=r'MSE($\hat{X}$, X)')
                plt.plot(rr, loss_r, label=r'MSE($\hat{X}$, X) + $\alpha$||G||$_1$')
                plt.plot(rr[r_opt-r_min], loss_r[r_opt-r_min], '-o')
                plt.xlabel('truncations (r)')
                plt.legend()

                fig.add_subplot(1,3,2)
                plt.plot(rr, R2_r)
                plt.plot(rr[r_opt-r_min], R2_r[r_opt-r_min], '-o')
                plt.xlabel('truncations (r)')
                plt.title(f'r$^2$')

                fig.add_subplot(1,3,3)
                plt.plot(rr, s_r)
                plt.plot(rr[r_opt-r_min], s_r[r_opt-r_min], '-o')
                plt.xlabel('truncations (r)')
                plt.title(f'Max Amplification')

        norm_opt, err_opt, R2_opt, s_opt = self.LIM_r(U, s, V, r_opt, X, hyp_param, verbose=verbose)
        if verbose:
            print(f'r_opt={r_opt}: % of var={s_prct[r_opt-1]:>8f}, norm={norm_opt:>8f}, err={err_opt:>8f}, R2={R2_opt:>8f}, s={s_opt:>8f}')
            print(f"POP e-folding timescales =\n {-1/self.b.real}")

    def LIM_r(self, U, s, V, r, X, hyp_param, verbose=False):
        """
        LIM for X truncated to the rth SVDs 
        """

        use_LIM_skills = True
        lag = hyp_param['lim']['lag_time']
        ord = hyp_param['lim']['ord']

        len_year, len_day, _ = V.shape
        Ct_hat = cov_lag(V[:, :, :r], lag_time=lag) * (len_day-lag) * len_year  # summation in time instead of time mean
        Ct_hat_s = torch.diag(s[:r]) @ Ct_hat.to(s.dtype) @ torch.diag(1.0/s[:r])
        self.w, self.vl_s, self.vr_s = eig_m(Ct_hat_s, eig_method=hyp_param['lim']['eig_method'], verbose=False)

        self.U = U[:, :r]
        self.vl = U[:, :r].to(self.vr_s.dtype) @ self.vl_s
        self.vr = U[:, :r].to(self.vr_s.dtype) @ self.vr_s
        self.b = torch.log(self.w)/lag
        self.B = (self.vr @ torch.diag(self.b) @ self.vl.conj().T).real
        # B_norm = la.vector_norm(self.B, ord=ord)**ord

        # noise covariance matrix and eigenvectors
        # Q is a symmetric real square matrix
        self.Q = - self.B @ self.C0 - self.C0 @ self.B.T    # matmul
        w_Q, vl_Q, vr_Q = eig_m(self.Q)
        self.w_Q, self.vl_Q, self.vr_Q = w_Q[:r].real, vl_Q[:, :r].real, vr_Q[:, :r].real

        Gt = (self.vr @ torch.diag(torch.exp(self.b * lag)) @ self.vl.conj().T).real
        Gt_norm = la.vector_norm(Gt, ord=ord)**ord

        t = torch.arange(lag+1)
        if use_LIM_skills:
            LIM_err, LIM_R2 = self.LIM_skills(t)
            err, R2 = LIM_err[-1], LIM_R2[-1]
        else:
            err, R2 = self.forecast_skills(X, t)

        U_r, s_r, V_r = self.optimals(t, method='svd')

        return Gt_norm, err, R2, s_r[-1]

    def forecast_offset(self, X2, t, slice_offset):
        """use `slice_offset` such that the initial condition `X2` has varying size
        """
        err = torch.zeros(len(t))     # err[0] = 0
        R2 = torch.ones_like(err)     # R2[0] = 1
        for idx in range(1, len(t)):
            if slice_offset-t[idx] >= 0:
                err[idx], R2[idx] = self.forecast_skills(X2[:, slice_offset-t[idx]:, :], t[[0, idx]])
            else:
                raise Exception('Error in `slice_offset`')

        return err, R2

    def LIM_skills(self, t):
        """Theoretical forecast skills of the LIM
        """
        len_t = len(t)
        err = torch.empty((len_t, ))
        R2 = torch.empty_like(err)
        for idx in range(len_t):
            Gt = (self.vr @ torch.diag(torch.exp(self.b * t[idx])) @ self.vl.conj().T).real
            err[idx] = 1.0 - torch.trace(Gt @ self.C0 @ Gt.T)/torch.trace(self.C0)
            R2[idx] = 1.0 - err[idx]

        return err, R2
    
    def forecast_skills(self, X, t):
        """forecast skills of the LIM for X at the time step t[-1]
        """
        Xf = self.forecast(X[:, :-t[-1], :], t[[0, -1]])    # t=0, -1

        # Anomaly correlation for t[-1]
        X1 = X[:, t[-1]:, :]
        X2 = Xf[:, -1, :, :]

        err = la.norm(X1 - X2)**2.0/X2.numel()
        X1X2 = np.trace(cov_lag(X1, lag_time=0, X2=X2))
        X1X1 = np.trace(cov_lag(X1, lag_time=0, X2=X1))
        X2X2 = np.trace(cov_lag(X2, lag_time=0, X2=X2))
        R2 = X1X2**2/X1X1/X2X2

        return err, R2

    def forecast(self, X, t):
        """
        X(years, batch, x): batch of initial conditions
        t: time lags to forecast; the first of t is t=0
        return y(years, t, batch, x): batch forecasts at t starting from X
        """

        if X.ndim == 2:
            X = X[None, :]  # 1st dim of `X` is size=1
        
        len_t = len(t)
        len_year, len_day, len_x = X.shape
        y = torch.empty((len_year, len_t, len_day, len_x))
        for idx in range(len_t):
            Gt = (self.vr @ torch.diag(torch.exp(self.b * t[idx])) @ self.vl.conj().T).real
            for n in range(len_year):
                y[n,idx,:,:] = (Gt @ torch.squeeze(X[n,:,:]).T).T
            
        return y

    def optimals(self, t, method='svd'):
        """
        solve for `optimals` of exp(b t) for a given `t`
        t: time lags to forecast
        `svd`: U for the pattern at `t`, Vh for optimals at t=0
        """

        len_t = len(t)
        if method == 'svd':
            U1 = torch.empty((len_t, len(self.vr)))
            s1 = torch.empty((len_t, ))
            V1 = torch.empty_like(U1)
            for idx in range(len_t):
                Gt = (self.vr @ torch.diag(torch.exp(self.b * t[idx])) @ self.vl.conj().T).real
                U, s, Vh = la.svd(Gt)
                U1[idx, :], s1[idx], V1[idx, :] = U[:, 0], s[0], Vh[0, :]

            return U1, s1, V1

        else:
            raise Exception('The method for finding optimals is not specified!')

#=============================================================
# tests of LIM class
#=============================================================
def LIM_test(y, t, hyp_param):
    model = LIM(y, hyp_param, verbose=True)

    forecast_time = 5
    yf = model.forecast(y[:, :-forecast_time, :], t[:forecast_time+1])
    print(f'y.shape={y.shape}, yf.shape={yf.shape}')

    U, s, V = model.optimals(forecast_time=t[:forecast_time+1], method='svd')
    print(f'U.shape={U.shape}, s.shape={s.shape}, V.shape={V.shape}')

# hyp_param = dict(lim = dict(lag_time = 5,
#                             r_optimal = None,
#                             eig_method = 'pinv',
#                             alpha = 1e-3,
#                             ord   = 1,
#                             forecast_time = 5,
#                             )
#                 )
# print(f'hyperpamameters:\n{hyp_param}')
# LIM_test(y, t, hyp_param)


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
    where ``vl`` and ``vr`` are biorthogonal bases, i.e., `vr @ vl.H = I`, and `vl.H @ vr = I`.

    eig_method options:
    eig2: solve vl and vr separately by calling `eig` twice
    pinv: solve vr by calling `eig`, and then solve vl by calling `pinv` with `inv_method`
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

    # factorizing the square matrix as ``a = vr @ torch.diag(w) @ vl.H``
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
def pinv(C0, Ct=None, inv_method='pinv'):
    """ Calculate  `Ct @ C0_inv` or `C0_inv` if Ct=None
        inv_method = 'pinv'; 'lstsq';
    """

    # C0_rank = la.matrix_rank(C0)
    # print(f'rank of inverted matrix ={C0_rank}')
    rcond = torch.finfo(C0.dtype).eps * max(C0.shape)
    if inv_method == 'pinv':
        C0_inv = la.pinv(C0, rcond=rcond)
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
def cov_lag(X, lag_time, X2=None):
    """ Input: X(years, days, x), optional: X2(years, days, x2), where years and days are time, and x and x2 are space
        First calculate the covariance in the dimension of `days` using `matmul`, \\
            and then take the average along the dimension of `years`
        return Ct(x, x2)
    """

    if X2 is None:
        X2 = X

    if X.ndim == 2:
        X = X[:, :, None]   # last dim of 'x' is size=1
    
    if X2.ndim == 2:
        X2 = X2[:, :, None]   # last dim of 'x' is size=1

    len_day = X.shape[1]
    X1 = torch.permute(X, (0, 2, 1))     # prepare for torch.matmul or @; torch.permute is equivalent np.transpose
    Ct = X1[:, :, lag_time:] @ X2[:, 0:len_day-lag_time, :] / (len_day-lag_time)
    
    return torch.squeeze(Ct.mean(axis=0))

#=============================================================
# tests of eig_m function
#=============================================================
def Gt_test(X, hyp_param, verbose=False):
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

def eig_m_test(X, hyp_param):
        Gt = Gt_test(X, hyp_param)
        w2, vl2, vr2 = eig_m(Gt, eig_method='eig2', inv_method='pinv', verbose=True)

# hyp_param = dict(lim = dict(lag_time = 5,
#                            ),
#                 )
# print(f'hyperpamameters:\n{hyp_param}')
# eig_m_test(y, hyp_param)

#=============================================================
# function rand_model()
# dX/dt = BX + F_s @ N(0,1)
# Q = vr_Q @ torch.diag(w_Q) @ vr_Q.T = F_s @ F_s.T * delta_t
#=============================================================
def rand_model(B, vr_Q, w_Q, len_t=1000, delta_t=1.0, verbose=False):
    """LIM forced by random noise
        X[t+1,:] = X[t,:] + (B @ X[t, :] + F_s @ N(0, 1)) * delta_t
        two-step process following Penland and Matrosova (1994)
    """
    torch.manual_seed(19680801)
    if delta_t > 1.0:
        raise Exception('`delta_t` must not be greater than 1.0')
    step = round(1.0/delta_t)
    delta_t_sqrt = delta_t**0.5

    len_x, r_opt = vr_Q.shape
    F_s = vr_Q @ torch.diag(w_Q**0.5) / delta_t_sqrt

    X = torch.empty((len_t+1, len_x))
    X[0, :] = F_s @ torch.randn(r_opt) * delta_t
    for t in np.arange(len_t):
        X[t+1,:] = X[t,:] + (B @ X[t,:] +  F_s @ torch.randn(r_opt)) * delta_t

    X_half = 0.5*(X[:-1,:] + X[1:,:])

    if verbose:
        print(f'F_s.shape={F_s.shape}, w_Q={w_Q}')

        fig = plt.figure(figsize=(7.5,6))
        ax1 = fig.add_subplot(2, 2, 1)
        plt.contourf(B)
        plt.title(r'B')
        cbar = plt.colorbar()

        ax1 = fig.add_subplot(2, 2, 3)
        plt.contourf(F_s @ F_s.T)
        plt.title(r'F$_s$F$_s^T$')
        cbar = plt.colorbar()

        ax1 = fig.add_subplot(2, 2, 2)
        plt.contourf(F_s)
        plt.title(r'F$_s$')
        # cbar = plt.colorbar()

        ax1 = fig.add_subplot(2, 2, 4)
        plt.plot(w_Q,'-o')
        plt.title('eigenvalues')

    return X_half[::step]

#=============================================================
# tests of rand_model function
#=============================================================
def rand_model_test():
    B = torch.tensor([[-0.06, 0], [0, -0.1]])
    C0 = torch.tensor([[0.5, 0], [0, 3.0]])
    Q =  - B @ C0 - C0 @ B.T
    w_Q, vl_Q, vr_Q = eig_m(Q)
    w_Q, vl_Q, vr_Q = w_Q.real, vl_Q.real, vr_Q.real

    y3 = rand_model(B, vr_Q, w_Q, len_t=10000, delta_t=0.5)
    print(f'y3={y3.shape}')

    hyp_param = dict(lim = dict(lag_time = 5,
                                r_optimal = None,
                                eig_method = 'pinv',
                                alpha = 1e-3,
                                ord   = 1,
                                )
                    )
    print(f'hyperpamameters:\n{hyp_param}')
    model3 = LIM(y3[None,:], hyp_param, verbose=False)

    print('-'*30)
    print(f' B={B},\n Q={Q},\n vr_Q={vr_Q},\n w_Q={w_Q},\n C0={C0}')
    print('-'*30)
    print(f' B={model3.B},\n Q={model3.Q},\n vr_Q={model3.vr_Q},\n w_Q={model3.w_Q},\n C0={model3.C0}')
