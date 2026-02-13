#=============================================================
# Module for Linear Inverse Model (LIM) using Numpy
#=============================================================
import numpy as np
from numpy import linalg as la

import matplotlib.pyplot as plt

#=============================================================
# class LIM(LIM_diag(LIM_core), LIM_forecast(LIM_core))
# Linear Inverse Model(LIM) described in Penland (1989)
#=============================================================
class LIM_core():
    def __init__(self, X, hyp_param, verbose=False):
        """
        Input: X(years, days, x), where years and days are time (denoted by t), and x is space
        hyper_parm: `lag_time` is the lag time used to compute Ct
                    `eig_method` = 'eig2' or `eig_method` = 'pinv'

        The propagation matrix is calculated as
        `Gt = Ct @ C0_inv`.

        The eigenvector decomposition of the propagation matrix (using the function `Myla.eig`) is 
        `Gt @ vr = vr @ np.diag(w)` or `Gt = vr @ np.diag(w) @ vl.H`, \\
        where vl and vr are biorthogonal bases, `vr @ vl.H = I`, and `vl.H @ vr = I`.

        The projection of `X` in the eigenvector space is
        `pc(t, eig) = (vl.H @ X.H).H = X @ vl`, \\
        and the reconstruction in the physical space is
        `X(t, x) = (vr @ pc.H).H = pc @ vr.H`.        
        """

        U_r, s, _ = Myla.svd(X)

        r_opt = hyp_param['lim']['r_optimal']
        self.r_opt = r_opt
        self.U_r = U_r[:, :r_opt]
        self.s_prct = np.sum(s[:r_opt]*s[:r_opt])/np.sum(s*s)*100

        self.LIM_r(self.get_r_from_x(X), hyp_param, verbose=verbose)

    def get_r_from_x(self, X):
        """ 
        from X = X(years, days, x) to Vs_r = Vs_r(years, days, r)
        """
        return X @ self.U_r
    
    def get_x_from_r(self, Vs_r):
        """ 
        from Vs_r = Vs_r(years, days, r) to  X = X(years, days, x)
        """
        return Vs_r @ self.U_r.T

    def LIM_r(self, Vs_r, hyp_param, verbose=False):
        """
        Input: Vs_r(years, days, r), where years and days are time
        """

        lag = hyp_param['lim']['lag_time']

        self.C0 = Myla.cov_lag(Vs_r, lag_time=0)
        Ct = Myla.cov_lag(Vs_r, lag_time=lag)
        Gt = Myla.pinv(self.C0, Ct, inv_method='pinv')

        self.w, self.vl_r, self.vr_r = Myla.eig(Gt, eig_method=hyp_param['lim']['eig_method'], verbose=False)
        # ensure w is np.complex64; np.log(self.w) for negative real value of `w` can result in `nan`
        self.w = self.w.astype(np.complex64)
        # print('w=', self.w)
        self.b = np.log(self.w)/lag
        # print('b=', self.b)

        self.vl_r = self.vl_r.astype(np.complex64)
        self.vr_r = self.vr_r.astype(np.complex64)
        self.B = (self.vr_r @ np.diag(self.b) @ self.vl_r.conj().T).real
        self.vl = self.get_x_from_r(self.vl_r.T).T
        self.vr = self.get_x_from_r(self.vr_r.T).T

        # noise covariance matrix and eigenvectors
        # Q is a symmetric real square matrix
        self.Q = - self.B @ self.C0 - self.C0 @ self.B.T    # matmul
        w_Q, vl_Q, vr_Q = Myla.eig(self.Q)
        self.w_Q, self.vl_Q, self.vr_Q = w_Q.real, vl_Q.real, vr_Q.real

        if verbose:
            timescale = [f'{-1/self.b[i].real:.4f}' for i in range(len(self.b))]
            print("POP e-folding timescales =" + ", ".join(timescale))

    def get_Gt_r(self, t):
        return (self.vr_r @ np.diag(np.exp(self.b * t)) @ self.vl_r.conj().T).real

    def get_Gt(self, t):
        return (self.vr @ np.diag(np.exp(self.b * t)) @ self.vl.conj().T).real

#=============================================================
class LIM_diag(LIM_core):
    def __init__(self, X, hyp_param, verbose=False):
        """
        optimals: svd of Gt \\
        mode: decomposion of X by eig modes
        """
        super().__init__(X, hyp_param, verbose)

    def optimals(self, t):
        """
        solve for `optimals` of exp(b t) for a given `t`
        t: time lags to forecast
        `svd`: U for the pattern at `t`, Vh for optimals at t=0
        """

        len_t = len(t)
        U1 = np.empty((len_t, len(self.vr_r)))
        s1 = np.empty((len_t, ))
        V1 = np.empty_like(U1)
        for idx in range(len_t):
            Gt = self.get_Gt_r(t[idx])
            U, s, Vh = la.svd(Gt)
            U1[idx, :], s1[idx], V1[idx, :] = U[:, 0], s[0], Vh[0, :]

        return self.get_x_from_r(U1), s1, self.get_x_from_r(V1)

    def mode(self, X):
        """
        X(years, days, x):  where years and days are time (denoted by t), and x is space
        return y(years, days, x, mode): filtered data by eig_mode
        """
        
        if isinstance(X, np.ma.MaskedArray):
            X = X.data

        if X.ndim == 2:
            X = X[None, :]  # 1st dim of `X` is size=1
        
        Vs_r = self.get_r_from_x(X)

        num_mode = len(self.w)
        y = np.empty((*X.shape, num_mode))
        for n in range(len(X)):
            for mm in range(num_mode): 
                Vs_r_mode = (self.vr_r[:, mm][:, None] @ self.vl_r[:, mm][:, None].conj().T).real @ np.squeeze(Vs_r[n,:,:]).T
                y[n,:,:,mm] = self.get_x_from_r(Vs_r_mode.T)
        
        return np.squeeze(y)

#=============================================================
class LIM_forecast(LIM_core):
    def __init__(self, X, hyp_param, verbose=False):
        """
        LIM_skills: theoretical skills \\
        forecast: forecasts of X for time t
        """
        super().__init__(X, hyp_param, verbose)

    def LIM_skills(self, t):
        """
        Theoretical forecast skills of the LIM
        """
        len_t = len(t)
        err = np.empty((len_t, ))
        R2 = np.empty_like(err)
        for idx in range(len_t):
            Gt = self.get_Gt_r(t[idx])
            err[idx] = 1.0 - np.trace(Gt @ self.C0 @ Gt.T)/np.trace(self.C0)
            R2[idx] = 1.0 - err[idx]

        return err, R2
    
    def forecast(self, X, t):
        """
        X(years, batch, x): batch of initial conditions
        t: time lags to forecast; the first of t is t=0
        return y(years, t, batch, x): batch forecasts at t starting from X
        """

        if isinstance(X, np.ma.MaskedArray):
            X = X.data

        if X.ndim == 2:
            X = X[None, :]  # 1st dim of `X` is size=1
        
        Vs_r = self.get_r_from_x(X)
        
        len_t = len(t)
        len_year, len_day, len_r = Vs_r.shape
        y = np.empty((len_year, len_t, len_day, len_r))
        for idx in range(len_t):
            Gt = self.get_Gt_r(t[idx])
            for n in range(len_year):
                y[n,idx,:,:] = (Gt @ np.squeeze(Vs_r[n,:,:]).T).T
            
        return self.get_x_from_r(y)
    
    def forecast_skills(self, X, t):
        """forecast skills of the LIM for X at the time step t[-1]
        """
        Xf = self.forecast(X[:, :-t[-1], :], t[[0, -1]])    # t=0, -1

        # Anomaly correlation for t[-1]
        X1 = X[:, t[-1]:, :]
        X2 = Xf[:, -1, :, :]

        err = la.norm(X1 - X2)**2.0/X2.size
        X1X2 = np.trace(Myla.cov_lag(X1, lag_time=0, X2=X2))
        X1X1 = np.trace(Myla.cov_lag(X1, lag_time=0, X2=X1))
        X2X2 = np.trace(Myla.cov_lag(X2, lag_time=0, X2=X2))
        R2 = X1X2**2/X1X1/X2X2

        return err, R2
    
    def forecast_offset(self, X2, t, slice_offset):
        """use `slice_offset` such that the initial condition `X2` has the same size for different forecast times
        """
        err = np.zeros(len(t))     # err[0] = 0
        R2 = np.ones_like(err)     # R2[0] = 1
        for idx in range(1, len(t)):
            if slice_offset-t[idx] >= 0:
                err[idx], R2[idx] = self.forecast_skills(X2[:, slice_offset-t[idx]:, :], t[[0, idx]])
            else:
                raise Exception('Error in `slice_offset`')

        return err, R2

#=============================================================
class LIM(LIM_diag, LIM_forecast):
    def __init__(self, X, hyp_param, verbose=False, use_LIM_skills=True):
        super().__init__(X, hyp_param, verbose)

        if verbose:
            lag = hyp_param['lim']['lag_time']
            ord = hyp_param['lim']['ord']

            B_norm = la.norm(self.B.flatten(), ord=ord)**ord
            Gt = self.get_Gt(lag)
            Gt_norm = la.norm(Gt.flatten(), ord=ord)**ord
            
            t = np.arange(lag+1)
            if use_LIM_skills:
                LIM_err, LIM_R2 = self.LIM_skills(t)
                err, R2 = LIM_err[-1], LIM_R2[-1]
            else:
                err, R2 = self.forecast_skills(X, t)

            U, s, V = self.optimals(t)

            print(f'r_opt={self.r_opt}: % of var={self.s_prct:>8f}, norm={Gt_norm:>8f}, err={err:>8f}, R2={R2:>8f}, s={s[-1]:>8f}')

#=============================================================
# tests of LIM class
#=============================================================
def LIM_test(y, t, hyp_param):
    model = LIM(y, hyp_param, verbose=True)

    # forecast_time = 5
    # yf = model.forecast(y[:, :-forecast_time, :], t[:forecast_time+1])
    # print(f'y.shape={y.shape}, yf.shape={yf.shape}')

    # U, s, V = model.optimals(t[:forecast_time+1], method='svd')
    # print(f'U.shape={U.shape}, s.shape={s.shape}, V.shape={V.shape}')

    y_mode = model.mode(y)

    # testing the modal decomposition between 1-2 and 3+
    fig = plt.figure(figsize=(8,7))
    k=0
    ax1 = fig.add_subplot(2, 2, 1)
    plt.plot(y[:,:,k], y_mode[:,:,k,:2].sum(axis=2), '.b')
    plt.plot(y[:,:,k], y_mode[:,:,k,2:].sum(axis=2), '.r')
    ax1 = fig.add_subplot(2, 2, 2)
    plt.plot(y[0,:,k], '-k', label='total')
    plt.plot(y_mode[0,:,k,:2].sum(axis=1), '-b', label='mode 1 & 2')
    plt.plot(y_mode[0,:,k,2:].sum(axis=1), '-r', label='mode 3+')
    plt.legend()

    k=-2
    ax1 = fig.add_subplot(2, 2, 3)
    plt.plot(y[:,:,k], y_mode[:,:,k,:2].sum(axis=2), '.b')
    plt.plot(y[:,:,k], y_mode[:,:,k,2:].sum(axis=2), '.r')
    ax1 = fig.add_subplot(2, 2, 4)
    plt.plot(y[0,:,k], '-k', label='total')
    plt.plot(y_mode[0,:,k,:2].sum(axis=1), '-b', label='mode 1 & 2')
    plt.plot(y_mode[0,:,k,2:].sum(axis=1), '-r', label='mode 3+')
    plt.legend()

# hyp_param = dict(lim = dict(lag_time = 5,
#                             r_optimal = 5,
#                             eig_method = 'pinv',
#                             ord   = 1,
#                             )
#                 )
# print(f'hyperpamameters:\n{hyp_param}')
# LIM_test(y, t, hyp_param)

#=============================================================
# function rand_model()
# dX = (B @ X + F_s @ N(0,1)) * delta_t
# Q = vr_Q @ np.diag(w_Q) @ vr_Q.T = F_s @ F_s.T * delta_t
#=============================================================
def rand_model(B, vr_Q, w_Q, len_day=1000, delta_t=0.2, verbose=False, get_x_from_r=None):
    """
    LIM forced by random noise
    X[t+1,:] = X[t,:] + (B @ X[t, :] + F_s @ N(0, 1)) * delta_t
    two-step process following Penland and Matrosova (1994)
    """
    rng = np.random.default_rng(19680801)
    if round(1.0/delta_t)*delta_t != 1.0:
        raise Exception('`delta_t` must be a fraction of 1.0')
    step = round(1.0/delta_t)
    len_t = len_day * step
    delta_t_sqrt = delta_t**0.5

    len_r, len_Q = vr_Q.shape
    F_s = vr_Q @ np.diag(w_Q**0.5) / delta_t_sqrt

    X = np.empty((len_t+1, len_r))
    X[0, :] = F_s @ rng.standard_normal(len_Q) * delta_t
    xi = np.empty((len_t, len_r))
    for t in np.arange(len_t):
        xi[t,:] = F_s @ rng.standard_normal(len_Q)
        X[t+1,:] = X[t,:] + B @ X[t,:] * delta_t + xi[t,:] * delta_t

    X_half = 0.5*(X[:-1,:] + X[1:,:])

    if verbose:
        Q = (xi.T @ X_half + X_half.T @ xi)/len(X_half)
        print(f'Q={Q}')
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

    if get_x_from_r:
        return get_x_from_r(X_half[::step])    # output once daily
    else:
        return X_half[::step]    # output once daily

#=============================================================
# tests of rand_model function
#=============================================================
def rand_model_test():
    # B = np.array([[-0.05, 0], [0, -0.1]])
    # C0 = np.array([[3.0, 0], [0, 0.5]])
    B = np.array([[-0.05, 0, 0], [0, -0.04, 0], [0, 0, -0.1]])
    C0 = np.array([[3.0, 0, 0],  [0, 1, 0],     [0, 0, 0.5]])
    Q =  - B @ C0 - C0 @ B.T
    w_Q, vl_Q, vr_Q = Myla.eig(Q)
    w_Q, vl_Q, vr_Q = w_Q.real, vl_Q.real, vr_Q.real

    y3 = rand_model(B, vr_Q, w_Q, len_day=100000, delta_t=0.25)
    print(f'y3={y3.shape}')
    
    hyp_param = dict(lim = dict(lag_time = 5,
                                r_optimal = 3,
                                eig_method = 'pinv',
                                ord = 1,
                                )
                    )
    print(f'hyperpamameters:\n{hyp_param}')
    model3 = LIM(y3[None,:], hyp_param, verbose=True)

    print('-'*30)
    print(f' B={B},\n Q={Q},\n vr_Q={vr_Q},\n w_Q={w_Q},\n C0={C0}')
    print('-'*30)
    print(f' B={model3.B},\n Q={model3.Q},\n vr_Q={model3.vr_Q},\n w_Q={model3.w_Q},\n C0={model3.C0}')

#=============================================================
# My linalg class
#=============================================================
class Myla():
    @staticmethod
    def eig(a, eig_method='eig2', inv_method='pinv', verbose=False):
        """
        modified version of eigenvector decomposition:
        ``w, vl, vr = eig(a)``

        Factorizes the square matrix `a` into the normalized eigenvector matrix ``vr``, its inverse ``vl.H``, and
        a 1-D array ``w`` of eigenvalues: \\
        ``a @ vr = vr @ np.diag(w)`` or ``a = vr @ np.diag(w) @ vl.H`` \\
        where ``vl`` and ``vr`` are biorthogonal bases, i.e., `vr @ vl.H = I`, and `vl.H @ vr = I`.

        eig_method options:
        eig2: solve vl and vr separately by calling `eig` twice
        pinv: solve vr by calling `eig`, and then solve vl by calling `pinv` with `inv_method`
        """

        def eig_sort(w, v, sort_by_w=True):
            """
            function to sort eigenvectors v by the decending order of eigenvalues w (default by real(w)), with the option to sort by w.congj()
            """
            w_np = w    # use the `sort` method in numpy; np.argsort not implemented in pytorch
            if sort_by_w:
                idx_w_sort_np = w_np.argsort()[::-1]
            else:
                idx_w_sort_np = w_np.conj().argsort()[::-1]    # sort by w.conj()
            idx_w_sort = idx_w_sort_np.copy()

            w_sort = np.zeros_like(w)
            v_sort = np.zeros_like(v)
            for idx, idx_sort in enumerate(idx_w_sort):
                w_sort[idx] = w[idx_sort]
                v_sort[:, idx] = v[:, idx_sort]

            return w_sort, v_sort

        def eig_lr(a, eig_method=None):
            """
            factorizing the square matrix as ``a = vr @ np.diag(w) @ vl.H``
            """
            # perform eigendecomposition
            if eig_method == 'eig2':
                w1, vr1 = la.eig(a)
                w, vr = eig_sort(w1, vr1)

                w2, vl2 = la.eig(a.conj().T)
                ww, vl = eig_sort(w2, vl2, sort_by_w=False)
        
            elif eig_method == 'pinv':
                w1, vr1 = la.eig(a)
                w, vr = eig_sort(w1, vr1)

                vl = Myla.pinv(vr, inv_method=inv_method).conj().T    # use pinv defined here 

            else:
                raise Exception('The method for eigenvector decomposition is not specified!')

            return w, vl, vr
   
        def check_eigs(w, vl, vr, a=None):
            """
            To ensure the biorthogonality of the eigenvectors from eig()
            vr is normalized, but vl is not normalized in the modified version 
            """
            vrH_vr = vr.conj().T @ vr
            print(f"diagonal(vrH_vr)=\n{np.diagonal(vrH_vr).real}")

            vlH_c = vl.conj().T @ vr @ vl.conj().T
            print(f"norm(vlH - vlH @ vr @vlH) = {la.norm(vl.conj().T-vlH_c)}")

            vr_c = vr @ vl.conj().T @ vr
            print(f"norm(vr - vr @ vlH @vr) = {la.norm(vr-vr_c)}")

            if a is not None:
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

        w, vl, vr = eig_lr(a, eig_method=eig_method)

        # form a biorthogonal system
        # note the transpose operator (.T) in python differs from the operator (') in matlab for a complex
        vlH_vr = vl.conj().T @ vr
        for idx in range(len(w)):
            vl[:, idx] = vl[:, idx]/(vlH_vr[idx, idx].conj())

        if verbose:
            check_eigs(w, vl, vr, a)
            
        return w, vl, vr
        
    @staticmethod
    def pinv(C0, Ct=None, inv_method='pinv'):
        """ Calculate  `Ct @ C0_inv` or `C0_inv` if Ct=None
            inv_method = 'pinv'; 'lstsq';
        """

        # C0_rank = la.matrix_rank(C0)
        # print(f'rank of inverted matrix ={C0_rank}')
        rcond = np.finfo(C0.dtype).eps * max(C0.shape)
        if inv_method == 'pinv':
            C0_inv = la.pinv(C0, rcond=rcond)
            if Ct is None:
                Gt = C0_inv
            else:
                Gt = Ct @ C0_inv

        elif inv_method == 'lstsq':
            if Ct is None:
                Gt = la.lstsq(C0.T, np.eye(len(C0), dtype=C0.dtype), rcond=rcond)[0].T
            else:
                Gt = la.lstsq(C0.T, Ct.T, rcond=rcond)[0].T
                
        else:
            raise Exception('The method for matrix inversion is not specified!')

        return Gt

    @staticmethod
    def svd(X, verbose=False, full_matrices=False):
        """
        Input: X(years, days, x) \\
        svd: X = V @ np.diag(s) @ U.T \\
        """

        if isinstance(X, np.ma.MaskedArray):
            X = X.data

        # SVD for U, s, V
        len_year, len_day, len_x = X.shape
        Xh_flat = X.reshape(-1, len_x).T
        U, s, Vh_flat = la.svd(Xh_flat, full_matrices)    # Xh_flat = U @ np.diag(s) @ Vh_flat
        V = Vh_flat.T.reshape(len_year, len_day, len_x)     # len_x for the indices of SVDs

        rcond = np.finfo(Xh_flat.dtype).eps * max(*Xh_flat.shape)
        r_max = len(s[s>rcond*s[0]])    # rank of Xh_flat
        if verbose:
            s_prct = [f'{np.sum(s[:i+1]*s[:i+1])/np.sum(s*s)*100:.2f}' for i in range(r_max)]
            print(f'SVD: % of var=' + ", ".join(s_prct))

        return U, s, V

    @staticmethod
    def cov_lag(X, lag_time, X2=None):
        """ Input: X(years, days, x), optional: X2(years, days, x2), where years and days are time, and x and x2 are space
            First calculate the covariance in the dimension of `days` using `matmul`, \\
                and then take the average along the dimension of `years`
            return Ct(x, x2)
        """
        
        if isinstance(X, np.ma.MaskedArray):
            X = X.data

        if X2 is None:
            X2 = X
        elif isinstance(X2, np.ma.MaskedArray):
            X2 = X2.data

        if X.ndim == 2:
            X = X[:, :, None]   # last dim of 'x' is size=1
        
        if X2.ndim == 2:
            X2 = X2[:, :, None]   # last dim of 'x' is size=1

        len_day = X.shape[1]
        X1 = np.transpose(X, (0, 2, 1))     # prepare for np.matmul or @; Note that @ does not seem to work for N-dim MaskedArray with N>=2

        Ct = X1[:, :, lag_time:] @ X2[:, 0:len_day-lag_time, :] / (len_day-lag_time)
    
        return np.squeeze(Ct.mean(axis=0))
        
#=============================================================
# tests of My linalg class
#=============================================================
def Gt_test(X, hyp_param, verbose=False):
    """ Input: X(years, days, x), where years and days are time, and x is space
        The propagation matrix is calculated as
        `Gt = Ct @ C0_inv`
    """
    lag = hyp_param['lim']['lag_time']
    C0 = Myla.cov_lag(X, lag_time=0)
    Ct = Myla.cov_lag(X, lag_time=lag)
    Gt = Myla.pinv(C0, Ct, inv_method='pinv')

    if verbose:
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(1, 2, 1)
        plt.contourf(Gt, cmap='jet')
        plt.title('Gt')
        plt.colorbar()

    return Gt

def eig_m_test(X, hyp_param):
        Gt = Gt_test(X, hyp_param)
        w2, vl2, vr2 = Myla.eig(Gt, eig_method='eig2', inv_method='pinv', verbose=True)

# hyp_param = dict(lim = dict(lag_time = 5,
#                            ),
#                 )
# print(f'hyperpamameters:\n{hyp_param}')

# # test of `cov_lag` and `pinv`
# Gt_test(y, hyp_param, verbose=True)

# # test of `eig_m`
# eig_m_test(y, hyp_param)
