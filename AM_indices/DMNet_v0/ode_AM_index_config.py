#=============================================================
# module for neural ode of Annular mode
#=============================================================
import math
import numpy as np

import torch
from torch import nn

import matplotlib.pyplot as plt

#=============================================================
# get red noise data and plot
#=============================================================
def get_red_noise(gamma=[0.06], data_size=1000):
    np.random.seed(19680801)
    #eps = np.random.uniform(-1, 1, data_size)
    eps = np.random.randn(data_size)
        
    y = np.zeros_like(eps)
    if len(gamma) == 1:    # AR-1
        for i in range(1, data_size):
            y[i] = (1-gamma[0])*y[i-1] + eps[i-1]
    elif len(gamma) == 2:    # AR-2
        for i in range(2, data_size):
            y[i] = (1-gamma[0])*y[i-1] + gamma[1]*y[i-2] + eps[i-1]

    t = torch.linspace(1, len(y), len(y), dtype=torch.float32)
    true_y = torch.from_numpy(y.astype(np.float32))[:,None]
        
    return  t, true_y


def plot_red_noise(t, y, lag_time=30, gamma=None, model=None):
    from ode_nn_mod import OdeModel
    
    lags = np.linspace(-lag_time, lag_time, 2*lag_time+1)
    
    u = np.squeeze(y.detach().numpy())
    data_size = len(u)
    Cu = np.array([np.cov(u[0:data_size-lag], u[lag:])[0, 1] for lag in np.arange(0, lag_time+1)])
    Cov_u = np.hstack((Cu[-1::-1], Cu[1:]))
    
    fig1 = plt.figure(figsize=(12,5))
    ax1 = fig1.add_subplot(1,2,1)
    ax1.plot(lags, Cov_u/Cov_u[lag_time], '-b', label='true C(u)')
    if gamma is not None and len(gamma)==1:
        ax1.plot(lags, np.exp(-abs(lags)*gamma), '--k', label='exp(-gamma|t|)')
    
    if model is not None:
        with torch.no_grad():
            if isinstance(model, OdeModel):
                #call OdeModel method directly
                #print('\n', model.func)
                u2= np.squeeze(model(y[:data_size-lag_time-1], t[:lag_time+1]).detach().numpy())
            else:
                func = model
                if hyp_param['ode']['adjoint']:
                    from torchdiffeq import odeint_adjoint as odeint
                else:
                    from torchdiffeq import odeint
                #integrate func using odeint
                #print('\n', func)
                u2 = np.squeeze(odeint(func, y[:data_size-lag_time-1], t[:lag_time+1]).detach().numpy())
        
        Cu2 = np.array([np.cov(u2[0,:].T, u2[lag,:].T)[0, 1] for lag in np.arange(0, lag_time+1)])
        Cov_u2 = np.hstack((Cu2[-1::-1], Cu2[1:]))
        ax1.plot(lags, Cov_u2/Cov_u2[lag_time], '--r', label='predicted C(u)')

        ax2 = fig1.add_subplot(1,2,2)
        Cu3 = np.array([np.corrcoef(u[lag:data_size-lag_time-1+lag], u2[lag,:].T)[0, 1] for lag in np.arange(0, lag_time+1)])
        Cor_u3 = np.hstack((Cu3[-1::-1], Cu3[1:]))
        ax2.plot(lags, Cov_u/Cov_u[lag_time], '-b', label='true C(u)')
        ax2.plot(lags, Cor_u3, '--r', label='corr(true, predicted)')
        ax2.set_xlim(-lag_time, lag_time)
        ax2.legend()

    ax1.set_xlim(-lag_time, lag_time)
    ax1.legend()
