#=============================================================
# shared module for neural ode
#=============================================================
import math
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt

#=============================================================
# class OdeDataset(Dataset)
#=============================================================
class OdeDataset(Dataset):
    def __init__(self, X, batch_time):
        """
        class OdeDataset(Dataset): inherits from torch.utils.data.Dataset  
        X(t, x): input data, where t is time, and x is space \\
        batch_time: for the data sampled at t=idx, return y=X(idx:idx+batch_time, x) is time series with length=batch_time. \\
        Note that batch_time>=1, and python slice go from 0 through stop-1
        """

        self.X = X
        self.batch_time = batch_time

    def __len__(self):
        return len(self.X) - self.batch_time + 1

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.X[idx:idx+self.batch_time]
        return X, y


#=============================================================
# class OdeModel(nn.Module)
# integrate dy/dt = f(t, y), where f(t, y) is specified by `func`
#=============================================================
class OdeModel(nn.Module):

    def __init__(self, func, ode_param, t_batch=None, verbose=False):
        """
        integrate dy/dt = f(t, y), where f(t, y) is specified by func

        func: f(t, X) with X(t, x) and time t, and is an instance of nn.Module for the adjoint method
        ode_param: parameters passing to call odeint
        t_batch: t_batch is specified only for batch training so that t is not needed during training. size(t_batch)=batch_time
        """

        super(OdeModel, self).__init__()
        self.func = func
        self.ode_param = ode_param
        self.t_batch = t_batch
    
        if verbose:
            for name, param in self.named_parameters():
                print(f'params: {name}, {param}')

    def forward(self, X, t=None):    
        """
        X(batch_size, x): input data, where a batch of data are randomly sampled, and x is space \\
        t=None for batch training (using t_batch instead), otherwise it should be specified
        return y(batch_size, batch_time, x), where batch_time is the length of integration in time
        """

        if self.ode_param['adjoint']:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        if t is not None:    
            # t is specified
            y = odeint(self.func, X, t, method=self.ode_param['method'], options=self.ode_param['options'])
            return y
        else: 
            # use t_batch provided in __init__
            if self.t_batch is None:
                raise Exception('t is not defined!')
            else:
                y = odeint(self.func, X, self.t_batch, method=self.ode_param['method'], options=self.ode_param['options'])
                # 'permute_arg' switches the 0th dim (batch_time: integration in time) of 'y' with its 1st dim (batch_size: number of samples) for output
                permute_arg = [1, 0] + [i for i in range(2, y.ndim)]
                return y.permute(tuple(permute_arg))

            
#=============================================================
# class OdeTrainer()
# training and testing
#=============================================================
class OdeTrainer():
    
    def __init__(self, loss_fn, optimizer, train_param):
        """
        hyp_param: a list of hyper-parameters
        """

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_param = train_param
        self.save = {}    # log of loss and model parameters

    def fit(self, model, train_loader, test_loader, vis_arg=None):
        """
        Training and testing:
        model: an instance of class OdeModel
        train_loader: training instance of class OdeDataset
        test_loader: testing instance of class OdeDataset
        """

        if self.train_param['lr_scheduler']['method']=='StepLR':
            scheduler = StepLR(self.optimizer, 
                                    step_size=self.train_param['lr_scheduler']['StepLR']['step_size'], 
                                    gamma=self.train_param['lr_scheduler']['StepLR']['gamma']
                                   )

        for t in range(self.train_param['epochs']):
            if self.train_param['lr_scheduler']['method'] is None:
                print(f"Epoch {t+1}\n-------------------------------")
            else:
                print(f"Epoch {t+1}(LR={scheduler.get_last_lr()})\n-------------------------------")                

            self.train_loop(train_loader, model, self.loss_fn, self.optimizer)
            self.test_loop(test_loader, model, self.loss_fn)
            
            # # visualization at the end of each epoch
            # if vis_arg is not None:
            #     t, true_y, plot = vis_arg
            #     with torch.no_grad():
            #         plot(t, true_y, model=model)

            # StepLR
            if self.train_param['lr_scheduler']['method']=='StepLR':
                scheduler.step()
                
        print("#####  Training Completed!  #####")

    def train_loop(self, dataloader, model, loss_fn, optimizer):
        """
        Training:
        update the model parameters by backpropagation for each (mini)batch
        """

        size = len(dataloader.dataset)  # number of batches * batch_size
        
        if self.train_param['First_batch_only']:
            first_batch = next(iter(dataloader))
            dataloader2 = [first_batch] * len(dataloader)    # try to overfit the first batch only
        else:
            dataloader2 = dataloader
            
        for batch, (X, y) in enumerate(dataloader2):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            if self.train_param['max_norm'] is not None:
                clip_grad_norm_(model.parameters(), self.train_param['max_norm'])    # Clips gradient by max_norm is not None
            optimizer.step()

            self.log('train_loss', loss.item())
            self.log_param(model)
            
            if batch % self.train_param['output_freq'] == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
    def test_loop(self, dataloader, model, loss_fn):
        # if len(dataloader) == 0:
        if dataloader is None:
            print('test_loader is empty!')
            self.log('test_loss', [])
            return

        size = len(dataloader.dataset)
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                loss = loss_fn(pred, y)
                test_loss += loss.item()
                
                self.log('test_loss', loss.item())
        
        test_loss /= len(dataloader)    # divided by number of batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    def log(self, key, value, append_value=True):
        if (key in self.save) and append_value:
            self.save[key].append(value)
        else:
            self.save[key] = [value]

    def log_param(self, model, beta=0.9):
        with torch.no_grad():
            pmean_sum = 0
            gmean_sum = 0
            grad_norm = 0
            for name, param in model.named_parameters():
                # save the running means of param and their gradients
                if (name+".pmean" in self.save):
                    pmean = beta*self.save[name+".pmean"][0] + (1.0-beta)*param
                    gmean = beta*self.save[name+".gmean"][0] ** 2.0 + (1.0-beta)*param.grad ** 2.0
                else:
                    pmean = (1.0-beta)*param
                    gmean = (1.0-beta)*param.grad ** 2.0
                self.log(name+".pmean", pmean, append_value=False)
                self.log(name+".gmean", gmean ** 0.5, append_value=False)
                pmean_sum += torch.sum(torch.abs(pmean)).item()
                gmean_sum += torch.sum(gmean).item()
                #print(f'{name}:{param}')

                # save the norm of gradient
                param_norm = torch.linalg.norm(param.grad)
                grad_norm += param_norm.item() ** 2.0

            self.log("pmean", pmean_sum)
            self.log("gmean", gmean_sum ** 0.5)
            self.log("grad_norm", grad_norm ** 0.5)
        
    def plot_loss(self, log_scale=False):
        train_epoch = np.linspace(0, self.train_param['epochs'], len(self.save['train_loss']))
        test_epoch  = np.linspace(0, self.train_param['epochs'], len(self.save['test_loss']))

        fig1 = plt.figure(figsize=(12,5))
        ax1 = fig1.add_subplot(1,2,1)
        ax1.plot(train_epoch, self.save['train_loss'], '-r', label='train_loss')
        ax1.plot(test_epoch,  self.save['test_loss'],  '-b', label='test_loss')
        legend = ax1.legend(loc='upper right', fontsize="x-large")
        ax1.set_title('loss')
        ax1.set_xlabel('epoch #')
        if log_scale:
            ax1.set_yscale('log')

        ax1 = fig1.add_subplot(1,2,2)
        ax1.plot(train_epoch, self.save['grad_norm'], '-r', label='norm')
        ax1.plot(train_epoch, self.save['pmean'], '-k', label='|pmean|')
        ax1.plot(train_epoch, self.save['gmean'], '-b', label='|gmean|')
        legend = ax1.legend(loc='upper right', fontsize="x-large")
        ax1.set_title('param statistics')
        ax1.set_xlabel('epoch #')
        if log_scale:
            ax1.set_yscale('log')

#=============================================================
# class NeuralODE()
# train the neural network `func` using the input dataset `X`
# use OdeDataset, OdeModel, and OdeTrainer classes
#=============================================================
class NeuralODE():
    def __init__(self, X, t, hyp_param, verbose=False):
        """
        train the neural network `func` using the input dataset `X`
        X(t, x): input data, where t is time, and x is space
        t: time
        hyp_param: hyper-parameters for the neural network
        """

        print('Loading data ......')
        dataset = OdeDataset(X, hyp_param['data']['batch_time'])
        train_loader, test_loader = get_dataset(dataset, hyp_param['data'], verbose=verbose)
    
        print(f"\nInitializing model ......")
        func = get_func(hyp_param['func'], verbose=verbose)    # func: neural network
        # t_batch used for batch training without specifying 't'
        self.model = OdeModel(func, hyp_param['ode'], t_batch=t[:hyp_param['data']['batch_time']], verbose=verbose)

        print(f"\nTraining model ......")
        loss_fn = get_loss_fn(hyp_param['loss_fn'], verbose=verbose)
        optimizer = get_optimizer(self.model, hyp_param['optimizer'], verbose=verbose)

        self.trainer = OdeTrainer(loss_fn, optimizer, hyp_param['train'])
        #trainer.fit(self.model, self.train_loader, self.test_loader, vis_arg=(t, true_y, plot))
        self.trainer.fit(self.model, train_loader, test_loader)

def get_dataset(dataset, data_param, verbose=False):
        # split dataset into train and test datasets
        test_size = math.floor(len(dataset)*data_param['test_frac'])
        data_train, data_test = random_split(dataset, [len(dataset)-test_size, test_size],
                                     generator=torch.Generator().manual_seed(42))
        
        if verbose:
            print(f'total dataset size = {len(dataset)}, train size = {len(data_train)}, test size = {len(data_test)}')

        # loading datasets
        if data_param['First_batch_only']:
            train_loader = DataLoader(data_train, batch_size=data_param['batch_size'])
        else:
            train_loader = DataLoader(data_train, batch_size=data_param['batch_size'], shuffle=True)
            # shuffle set to True to have the data reshuffled at every epoch
        
        if len(data_test) == 0:
            print('data_test is empty!')
            test_loader = None
        else:
            test_loader = DataLoader(data_test, batch_size=data_param['batch_size'], shuffle=True)
        if verbose:
            print(f'batch size = {data_param["batch_size"]}, train batch # = {len(train_loader)}, test batch # = {len(test_loader)}')

            X0, y0 = next(iter(train_loader))
            print('Sample of dataset:')
            print(f'X0.shape = {X0.shape} for (batch_size, ...) \ny0.shape = {y0.shape} for (batch_size, batch_time, ...)')

        return train_loader, test_loader


def get_func(func_param, verbose=False):
        # func
        n_input, n_hidden, n_output = func_param["size"]
        if func_param['method'] == 'Linear':
            func = LinearFunc(n_input, n_output)
        elif func_param['method'] == 'DpTanh':
            func = DpTanhFunc(n_input, n_hidden, n_output)
        elif func_param['method'] == 'Tanh':
            func = TanhFunc(n_input, n_hidden, n_output)
        elif func_param['method'] == 'Softplus':
            func = SoftplusFunc(n_input, n_hidden, n_output)
        elif func_param['method'] == 'LeakyReLu':
            func = LeakyReLuFunc(n_input, n_hidden, n_output)
        else:
            raise Exception('func is not recognized!')
        
        if verbose:
            print(f'func: {func}')

        return func

def get_loss_fn(loss_param, verbose=False):
        if loss_param['method'] == 'MSELoss':
            loss_fn = nn.MSELoss()
        elif loss_param['method'] == 'L1Loss':
            loss_fn = nn.L1Loss()
        else:
            raise Exception('loss_fn is not recognized!')

        if verbose:
            print(f"loss_fn = {loss_param}")

        return loss_fn

def get_optimizer(model, opt_param, verbose=False):
        if opt_param['method'] == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=opt_param['learning_rate'])
        elif opt_param['method'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=opt_param['learning_rate'])
        elif opt_param['method'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=opt_param['learning_rate'], 
                        momentum=opt_param['SGD']['momentum'], weight_decay=opt_param['SGD']['weight_decay'])
        else:
            raise Exception('optimizer is not recognized!')

        if verbose:
            print(f"optimizer = {opt_param}")

        return optimizer

#=============================================================
# nn.Module class used to define func for f(t, y)
#=============================================================
class DpTanhFunc(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(DpTanhFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class TanhFunc(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(TanhFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
    
class SoftplusFunc(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(SoftplusFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Softplus(),
            nn.Linear(n_hidden, n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
    
class LeakyReLuFunc(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(LeakyReLuFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_output),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
    
class LinearFunc(nn.Module):

    def __init__(self, n_input, n_output):
        super(LinearFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, n_output)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
