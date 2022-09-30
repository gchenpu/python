#=============================================================
# hyperparameters for the dynamical modes of Annular mode
#=============================================================

# default values for the hyperparameters
hyp_param = dict(lim = dict(method = 'DMD',
                            lag_time = 5,
                            # eval_time = 20,
                            # forecast_time = 40,
                            LIM = dict( r_optimal = None,
                                       ),
                            DMD = dict( r_alpha = 1e-3,
                                        r_ord   = 1,
                                        r_optimal = None,
                                        r_forecast = 5,
                                        eig_method = 'pinv',
                                       )
                            ),

                ode = dict(adjoint = False, # False 
                           method = 'midpoint',
                           options = dict(step_size=0.2),    # step_size for fixed grid ODE solver in unit of t
                           ),
                                       
                data = dict(batch_size = 50,    # 100,    # 20, 200, # size of mini-batch
                            batch_time = 5,      # 10,    # the length in 't' integrated by odeint (cf. lag_time in LIM)
                            test_frac = 0.1,     # 0.1,    # fraction of 'dataset' used for test data
                            ),
                
                train = dict(epochs = 40,   # 40, 20  # number of epochs
                             output_freq = 100,    # output per the number of batches for each epoch
                             lr_scheduler = dict(method = None, # 'StepLR',    # None, 'StepLR', ...
                                                 # if `method = StepLR`, 
                                                 # the learning rate decays by gamma every step_size epochs
                                                 StepLR = dict(step_size=5, gamma=0.5),
                                                ),
                             max_norm = None,    # maximum norm used to clip the norms of model parameters
                             r_alpha = 1e-3,
                             r_ord   = 1,
                             First_batch_only = False, #False,    # overfit the first batch only for the purpose of debugging
                            ),     

                # L1 loss is more robust to outliers, but its derivatives are not continuous
                # L2 loss is sensitive to outliers, but gives a more stable and closed form solution
                loss_fn = dict(method = 'MSELoss', # 'MSELoss', 'L1Loss'
                              ),

                # https://ruder.io/optimizing-gradient-descent/
                # RMSprop is an adaptive learning rate algorithm
                # Adam adds bias-correction and momentum to RMSprop.
                optimizer = dict(method = 'SGD',    # 'RMSprop',     # 'RMSprop', 'Adam', 'SGD'
                                learning_rate = 1e-3,    # 1e-3,    # default LR
                                RMSprop = dict(),
                                Adam = dict(),
                                SGD = dict(momentum=0, weight_decay=0),
                                ),

                func = dict(method = 'DM', # 'Tanh',
                            size = 5,
                           ),
                
                )

# print(f'hyperpamameters:\n{hyp_param}')