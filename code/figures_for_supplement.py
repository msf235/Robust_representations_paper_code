import time
import numpy as np
import plots
import itertools
from multiprocessing import Process, Lock
from pathlib import Path
import seaborn as sns
# from matplotlib import pyplot as plt

plots.folder_root = "figs_revision"
plots.LEGEND = True  # Set this to False to get remove legends for plots
subdir_prefix = Path('')

chaos_color_0 = np.array([0, 160, 160])/255
chaos_color_1 = np.array([233, 38, 41])/255
chaos_colors = [chaos_color_0, chaos_color_1]
chaos_palette = sns.color_palette(chaos_colors)

CLEAR_PREVIOUS_RUNS = False
# CLEAR_PREVIOUS_RUNS = True # Delete saved parameters from previous runs

if CLEAR_PREVIOUS_RUNS:
    import shutil
    shutil.rmtree('../data/output')

# % Shallow network tests
base_params = dict(N=200,
                   num_epochs=40,
                   num_train_samples_per_epoch=800,
                   num_test_samples_per_epoch=100,
                   X_clusters=60,
                   X_dim=200,
                   num_classes=2,
                   n_lag=0,
                   # g_radius=1,
                   g_radius=20,
                   wout_scale=1.,
                   clust_sig=.02,
                   input_scale=1.0,
                   input_style='hypercube',
                   model_seed=1,
                   n_hold=1,
                   n_out=1,
                   use_biases=False,  # use_biases=True,
                   loss='mse',
                   optimizer='sgd',
                   momentum=0,
                   dt=.01,
                   learning_rate=1e-3,
                   # learning_patience=100,
                   scheduler='onecyclelr_4e4',
                   learning_patience=20,
                   scheduler_factor=10,
                   # scheduler='plateau',
                   # learning_patience=5,
                   # scheduler_factor=.5,
                   # scheduler='cyclic',
                   # learning_patience=20,
                   # scheduler_factor=.05,
                   batch_size=1,
                   train_output_weights=True,
                   freeze_input=False,
                   network='vanilla_rnn',
                   # network='feedforward',
                   Win='orthog',
                   l2_regularization=0,
                   patience_before_stopping=6000,
                   hid_nonlin='linear',
                   saves_per_epoch=1,
                   rerun=False,
                   # rerun=True,
                   )
# %% Probing batch sizes and freezing training weights
params_shallow_1 = {
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    'optimizer': ['sgd', 'rmsprop'],
    'loss': ['mse'],
    'hid_nonlin': ['linear'],
    'l2_regularization': [0],
    # 'l2_regularization': [0, 10],
    'g_radius': [1]
    }
params_shallow_2 = {
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'optimizer': ['sgd', 'rmsprop'],
    'loss': ['mse_scalar'],
    'hid_nonlin': ['linear'],
    'l2_regularization': [0],
    # 'l2_regularization': [0, 10],
    'g_radius': [1]
    }
keys_shallow = list(params_shallow_1.keys())
keys_shallow_abbrev = ['lr', 'opt', 'loss', 'nonlin', 'l2', 'g']
ps_list_shallow = list(itertools.product(*params_shallow_1.values())) \
    + list(itertools.product(*params_shallow_2.values()))

def run_shallow_1(param_set, train_params, i0, multiprocess_lock=None):
    # time.sleep(i0)
    plots.USE_ERRORBARS = False
    print(multiprocess_lock)
    train_params = train_params.copy()
    for i0, key in enumerate(keys_shallow):
        train_params[key] = param_set[i0]
    train_params['network'] = 'feedforward'
    n_lag = 0
    train_params['n_lag'] = n_lag
    full_batch_size = train_params['num_train_samples_per_epoch']
    tps_11 = train_params.copy()
    tps_11['batch_size'] = 1
    # tps_11['num_epochs'] = 200
    # tps_11['saves_per_epoch'] = 1/20
    tps_11['num_epochs'] = 100
    tps_11['saves_per_epoch'] = [0]*100
    for k in range(0, 10):
        tps_11['saves_per_epoch'][k] = 2
    for k in range(11, 100, 10):
        tps_11['saves_per_epoch'][k] = 1
    # tps_11['num_epochs'] = 100
    # tps_11['saves_per_epoch'] = 1/10
    if train_params['scheduler'] in ('onecyclelr_4e4', 'onecyclelr'):
        tps_11['learning_patience'] = 20
        tps_11['scheduler_factor'] = 10
        tps_11['patience_before_stopping'] = tps_11['num_epochs']
    tps_12 = train_params.copy()
    tps_12['batch_size'] = full_batch_size
    tps_12['num_epochs'] = 1000
    tps_12['saves_per_epoch'] = 1/100
    if train_params['scheduler'] in ('onecyclelr_4e4', 'onecyclelr'):
        tps_12['learning_patience'] = 100
        tps_12['scheduler_factor'] = 10
        tps_12['patience_before_stopping'] = tps_12['num_epochs']
    tps_21 = tps_11.copy()
    tps_21['train_output_weights'] = False
    tps_22 = tps_12.copy()
    tps_22['train_output_weights'] = False
    figname = ''.join(key + '_' + str(val) + '_' for key, val in
                      zip(keys_shallow_abbrev, param_set))
    figname = figname[:-1]

    subdir_prefix2 = Path('{}/'.format(tps_11['network']))
    subdir_suffix = Path('nlag_{}_g_{}_l2_{}/'.format(n_lag, tps_11['g_radius'],
                                                      train_params[
                                                          'l2_regularization']))
    plot_ps = ([tps_11, tps_12], [tps_21, tps_22], [0, 1, 2], 'batch_size',
               'train_output_weights', figname)
    subdir = subdir_prefix2/'dim_over_training'/subdir_suffix
    plots.dim_through_training(*plot_ps, subdir=subdir_prefix/subdir,
                               multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'dim_over_layers'/subdir_suffix
    plots.dim_over_layers(*plot_ps, subdir=subdir_prefix/subdir,
                          multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'orth_compression_through_training'/subdir_suffix
    plots.orth_compression_through_training(*plot_ps,
                                            subdir=subdir_prefix/subdir,
                                            multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'orth_compression_through_training_input_sep' \
                            ''/subdir_suffix
    plots.orth_compression_through_training_input_sep(*plot_ps,
                                                      subdir=subdir_prefix/subdir,
                                                      multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'acc_over_training'/subdir_suffix
    plots.acc_over_training(*plot_ps, subdir=subdir_prefix/subdir,
                            multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'loss_over_training'/subdir_suffix
    plots.loss_over_training(*plot_ps, subdir=subdir_prefix/subdir,
                             multiprocess_lock=multiprocess_lock)

# %% Probing noise of readout weights through training
params_readout_noise_1 = params_shallow_1.copy()
params_readout_noise_2 = params_shallow_2.copy()
params_readout_noise_1['l2_regularization'] = [0]
params_readout_noise_2['l2_regularization'] = [0]
keys_readout_noise = list(params_readout_noise_1.keys())
keys_readout_noise_abbrev = ['lr', 'opt', 'loss', 'nonlin', 'l2', 'g']
ps_list_readout_noise = list(itertools.product(*params_readout_noise_1.values())) \
                        + list(itertools.product(*params_readout_noise_2.values()))

def run_readout_noise(param_set, train_params, i0, multiprocess_lock=None):
    # time.sleep(i0)
    plots.USE_ERRORBARS = False
    print(multiprocess_lock)
    train_params = train_params.copy()
    for i0, key in enumerate(keys_readout_noise):
        train_params[key] = param_set[i0]
    train_params['network'] = 'feedforward'
    n_lag = 0
    train_params['n_lag'] = n_lag
    full_batch_size = train_params['num_train_samples_per_epoch']
    tps_11 = train_params.copy()
    tps_11['batch_size'] = 1
    # tps_11['num_epochs'] = 200
    # tps_11['saves_per_epoch'] = 1/20
    tps_11['num_epochs'] = 100
    tps_11['saves_per_epoch'] = 20
    if train_params['scheduler'] in ('onecyclelr_4e4', 'onecyclelr'):
        tps_11['learning_patience'] = 20
        tps_11['scheduler_factor'] = 10
        tps_11['patience_before_stopping'] = tps_11['num_epochs']
    figname = ''.join(key + '_' + str(val) + '_' for key, val in
                      zip(keys_readout_noise_abbrev, param_set))
    figname = figname[:-1]

    subdir_prefix2 = Path('{}/'.format(tps_11['network']))
    subdir_suffix = Path('nlag_{}_g_{}_l2_{}/'.format(n_lag, tps_11['g_radius'],
                                                      train_params[
                                                          'l2_regularization']))
    subdir = subdir_prefix2/'weight_var_through_training'/subdir_suffix
    plots.weight_var_through_training([tps_11], None, [0, 1, 2],
                                      hue_key='batch_size',
                                      style_key='train_output_weights',
                                      figname=figname,
                                      subdir=subdir_prefix/subdir,
                                      multiprocess_lock=multiprocess_lock)

# % Recurrent network experiments
# %% batch sizes
params_recurrent_1 = {
    # 'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'optimizer': ['sgd', 'rmsprop'],
    'loss': ['mse', 'mse_scalar'],
    'hid_nonlin': ['tanh', 'linear'],
    'l2_regularization': [0, 50],
    'g_radius': [20]
    }
params_recurrent_2 = {
    'learning_rate': [1e-6, 1e-5, 1e-4],
    'optimizer': ['sgd', 'rmsprop'],
    'loss': ['cce'],
    'hid_nonlin': ['tanh', 'linear'],
    'l2_regularization': [0],
    'g_radius': [20]
    }
keys_deep = list(params_recurrent_1.keys())
keys_abbrev = ['lr', 'opt', 'loss', 'nonlin', 'l2', 'g']
ps_list_recurrent = list(itertools.product(*params_recurrent_1.values())) + \
                    list(itertools.product(*params_recurrent_2.values()))

def run_recurrent(param_set, train_params, i0, multiprocess_lock=None):
    plots.USE_ERRORBARS = False
    # time.sleep(i0)
    print(multiprocess_lock)
    train_params = train_params.copy()
    for i0, key in enumerate(keys_deep):
        train_params[key] = param_set[i0]
    # if train_params['loss'] == 'mse_scalar':
    #     train_params['rerun'] = True
    train_params['n_lag'] = 10
    lr = train_params['learning_rate']
    optimizer = train_params['optimizer']
    n_lag = train_params['n_lag']
    figname = ''.join(
        key + '_' + str(val) + '_' for key, val in zip(keys_abbrev, param_set))
    figname = figname[:-1]
    full_batch_size = train_params['num_train_samples_per_epoch']
    tps_11 = train_params.copy()
    tps_11['batch_size'] = 1
    tps_11['num_epochs'] = 200
    tps_11['saves_per_epoch'] = 1/20
    # tps_11['num_epochs'] = 100
    # tps_11['saves_per_epoch'] = [0]*100
    # for k in range(0, 10):
    #     tps_11['saves_per_epoch'][k] = 2
    # for k in range(11, 100, 10):
    #     tps_11['saves_per_epoch'][k] = 1
    if train_params['scheduler'] in ('onecyclelr_4e4', 'onecyclelr'):
        tps_11['learning_patience'] = 20
        tps_11['scheduler_factor'] = 10
        tps_11['patience_before_stopping'] = tps_11['num_epochs']
    tps_11['patience_before_stopping'] = tps_11['num_epochs']
    tps_12 = train_params.copy()
    tps_12['batch_size'] = full_batch_size
    tps_12['num_epochs'] = 1000
    tps_12['saves_per_epoch'] = 1/100
    if train_params['scheduler'] in ('onecyclelr_4e4', 'onecyclelr'):
        tps_12['learning_patience'] = 20
        tps_12['scheduler_factor'] = 10
        tps_12['patience_before_stopping'] = tps_12['num_epochs']
    # figname = "lr_{}_opt_{}_l2_{}".format(lr, optimizer,
    #                                       train_params['l2_regularization'])

    subdir_prefix2 = Path('{}'.format(train_params['network']))
    subdir_suffix = Path('nlag_{}_g_{}_l2_{}'.format(n_lag, tps_11['g_radius'],
                                                     train_params[
                                                         'l2_regularization']))
    plot_ps = ([tps_11, tps_12], None, [0, 1], 'batch_size', None, figname)
    subdir = subdir_prefix2/'dim_over_training'/subdir_suffix
    plots.dim_through_training(*plot_ps, subdir=subdir_prefix/subdir,
                               multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'dim_over_layers'/subdir_suffix
    plots.dim_over_layers(*plot_ps, subdir=subdir_prefix/subdir,
                          multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'orth_compression_through_training'/subdir_suffix
    plots.orth_compression_through_training(*plot_ps,
                                            subdir=subdir_prefix/subdir,
                                            multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'orth_compression_through_training_input_sep' \
                            ''/subdir_suffix
    plots.orth_compression_through_training_input_sep(*plot_ps,
                                                      subdir=subdir_prefix/subdir,
                                                      multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'acc_over_training'/subdir_suffix
    plots.acc_over_training(*plot_ps, subdir=subdir_prefix/subdir,
                            multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'loss_over_training'/subdir_suffix
    plots.loss_over_training(*plot_ps, subdir=subdir_prefix/subdir,
                             multiprocess_lock=multiprocess_lock)

params_readout_noise_deep_1 = params_recurrent_1.copy()
params_readout_noise_deep_1['l2_regularization'] = [0]
params_readout_noise_deep_2 = params_recurrent_2.copy()
params_readout_noise_deep_2['l2_regularization'] = [0]
keys_readout_noise_deep = list(params_recurrent_1.keys())
keys_readout_noise_deep_abbrev = ['lr', 'opt', 'loss', 'nonlin', 'l2', 'g']
ps_list_readout_noise_recurrent = list(
    itertools.product(*params_readout_noise_deep_1.values())) + \
                                  list(itertools.product(
                               *params_readout_noise_deep_2.values()))
# pvals_readout_noise_deep = ps_vals_deep

def run_readout_noise_recurrent(param_set, train_params, i0, multiprocess_lock=None):
    # time.sleep(i0)
    plots.USE_ERRORBARS = False
    print(multiprocess_lock)
    train_params = train_params.copy()
    for i0, key in enumerate(keys_readout_noise_deep):
        train_params[key] = param_set[i0]
    train_params['n_lag'] = 10
    n_lag = train_params['n_lag']
    tps_11 = train_params.copy()
    tps_11['batch_size'] = 1
    # tps_11['num_epochs'] = 200
    # tps_11['saves_per_epoch'] = 1/20
    tps_11['num_epochs'] = 100
    tps_11['saves_per_epoch'] = 20
    if train_params['scheduler'] in ('onecyclelr_4e4', 'onecyclelr'):
        tps_11['learning_patience'] = 20
        tps_11['scheduler_factor'] = 10
        tps_11['patience_before_stopping'] = tps_11['num_epochs']
    figname = ''.join(key + '_' + str(val) + '_' for key, val in
                      zip(keys_readout_noise_deep_abbrev, param_set))
    figname = figname[:-1]

    subdir_prefix2 = Path('{}/'.format(tps_11['network']))
    subdir_suffix = Path('nlag_{}_g_{}_l2_{}/'.format(n_lag, tps_11['g_radius'],
                                                      train_params[
                                                          'l2_regularization']))
    subdir = subdir_prefix2/'weight_var_through_training'/subdir_suffix
    plots.weight_var_through_training([tps_11], None, [0, 1],
                                      hue_key='batch_size',
                                      style_key='train_output_weights',
                                      figname=figname,
                                      subdir=subdir_prefix/subdir,
                                      multiprocess_lock=multiprocess_lock)
    subdir = subdir_prefix2/'weight_product_var_through_training'/subdir_suffix
    plots.weight_var_through_training([tps_11], None, [0, 1],
                                      hue_key='batch_size',
                                      style_key='train_output_weights',
                                      figname=figname,
                                      subdir=subdir_prefix/subdir,
                                      weight_type='product',
                                      multiprocess_lock=multiprocess_lock)
# %% Low-d input experiments
low_d_params = dict(N=200, num_epochs=80, num_train_samples_per_epoch=800,
                    X_clusters=60, X_dim=2, num_classes=2, n_lag=10,
                    g_radius=20, clust_sig=.02, input_scale=1, n_hold=1,
                    n_out=1, loss='cce', optimizer='rmsprop', dt=.01,
                    momentum=0, learning_rate=1e-3, batch_size=10,
                    freeze_input=False, network='vanilla_rnn', Win='orthog',
                    patience_before_stopping=6000, hid_nonlin='tanh',
                    model_seed=0, rerun=False)

params_lowd = {
    'X_clusters': [40, 60, 120],
    'X_dim': [2, 4, 10],
    'learning_rate': [1e-4, 1e-3],
    'n_lag': [6, 10, 14],
    'N': [200, 300],
    'loss': ['cce', 'mse'],
    # 'l2_regularization': [0, 10, 50],
    }
# params_lowd = {
#     'X_clusters': [60],
#     'X_dim': [2, 4],
#     'learning_rate': [1e-4, 1e-3, 5e-3],
#     'n_lag': [14],
#     'N': [200],
#     'loss': ['mse']
#     }
keys_lowd = list(params_lowd.keys())
keys_abbrev_lowd = ['clust', 'xdim', 'lr', 'nlag', 'N', 'loss']
ps_list_lowd = list(itertools.product(*params_lowd.values()))

def run_lowd(param_set, train_params, i0, multiprocess_lock=None):
    plots.USE_ERRORBARS = True
    # time.sleep(i0)
    tps = low_d_params.copy()
    # tps['table_path'] = 'output_lowd/output_table.csv'
    subdir_prefix = Path('Win_orth_lowd/')
    for i0, key in enumerate(keys_lowd):
        tps[key] = param_set[i0]
    tps_11 = tps.copy()
    tps_11['g_radius'] = 20
    tps_21 = tps_11.copy()
    tps_21['num_epochs'] = 0
    tps_12 = tps.copy()
    tps_12['g_radius'] = 250
    tps_22 = tps_12.copy()
    tps_22['num_epochs'] = 0

    figname = ''.join(key + '_' + str(val) + '_' for key, val in
                      zip(keys_abbrev_lowd, param_set))
    figname = figname[:-1]
    # print(figname)
    # figname = "lr_{}_clust_{}_layers_{}".format(lr, x_cluster, n_lag)
    subdir = subdir_prefix/'dim_over_layers'
    plots.dim_over_layers([tps_11, tps_12], [tps_21, tps_22], seeds=[0, 1, 2],
                          hue_key='g_radius', style_key='num_epochs',
                          figname=figname + '_g_{}'.format(tps_11['g_radius']),
                          subdir=subdir, multiprocess_lock=multiprocess_lock,
                          style_order=[80, 0], palette=chaos_palette)
    plots.dim_over_layers([tps_11, tps_12], [tps_21, tps_22], seeds=[0, 1, 2],
                          hue_key='g_radius', style_key='num_epochs',
                          figname=figname + '_g_{}'.format(tps_11['g_radius']),
                          subdir=subdir, multiprocess_lock=multiprocess_lock,
                          style_order=[80, 0], palette=chaos_palette)
    subdir = subdir_prefix/'clust_holdout_over_layers'
    plots.clust_holdout_over_layers([tps_11, tps_12], [tps_21, tps_22], seeds=[0, 1, 2],
                                    hue_key='g_radius', style_key='num_epochs',
                                    figname=figname + '_g_{}'.format(tps_11['g_radius']),
                                    subdir=subdir, multiprocess_lock=multiprocess_lock,
                                    style_order=[80, 0], palette=chaos_palette)
    # subdir = subdir_prefix/'ashok_compression_metric'
    # plots.ashok_compression_metric([tps_12], [tps_21, tps_22], seeds=[0, 1, 2],
    #                                style_key='num_epochs',
    #                                figname=figname + '_g_{}'.format(
    #                                    tps_12['g_radius']), subdir=subdir,
    #                                multiprocess_lock=multiprocess_lock,
    #                                style_order=[80, 0])
    plots.USE_ERRORBARS = False

# %% Low-d chaos experiments (Figures S
low_d_chaos_params = dict(N=200, num_epochs=80, num_train_samples_per_epoch=800,
                          X_clusters=60, X_dim=2, num_classes=2, n_lag=10,
                          g_radius=20, clust_sig=.02, input_scale=1, n_hold=1,
                          n_out=1, loss='cce', optimizer='rmsprop', dt=.01,
                          momentum=0, learning_rate=1e-3, batch_size=10,
                          freeze_input=False, network='vanilla_rnn',
                          Win='orthog',
                          patience_before_stopping=6000, hid_nonlin='tanh',
                          model_seed=0, rerun=False)

params_lowd_chaos = {
    'X_clusters': [60],
    'X_dim': [2],
    'learning_rate': [1e-3, 1e-4],
    'n_lag': [10]
    }
keys_lowd_chaos = list(params_lowd_chaos.keys())
keys_abbrev_lowd_chaos = ['clust', 'xdim', 'lr', 'nlag']
ps_list_lowd_chaos = list(itertools.product(*params_lowd_chaos.values()))

def run_lowd_chaos(param_set, train_params, i0, multiprocess_lock=None):
    plots.USE_ERRORBARS = True
    # time.sleep(i0)
    tps = train_params.copy()
    subdir_prefix = Path('Win_orth_lowd_chaos/')
    for i0, key in enumerate(keys_lowd_chaos):
        tps[key] = param_set[i0]

    tp_list_0 = []
    tp_list_1 = []
    for g in range(20, 261, 40):
        tp = tps.copy()
        tp['g_radius'] = g
        tp_list_1.append(tp)
        tp = tp.copy()
        tp['num_epochs'] = 0
        tp_list_0.append(tp)

    figname = ''.join(key + '_' + str(val) + '_' for key, val in
                      zip(keys_abbrev_lowd_chaos, param_set))
    figname = figname[:-1]
    seeds = list(range(5))
    subdir = subdir_prefix/'dim_over_layers'
    plots.dim_over_layers(tp_list_0, None, seeds=seeds,
                          hue_key='g_radius', style_key='num_epochs',
                          figname=figname + '_before',
                          subdir=subdir, use_error_bars=True,
                          multiprocess_lock=multiprocess_lock,
                          palette='viridis')
    plots.dim_over_layers(tp_list_1, None, seeds=seeds,
                          hue_key='g_radius', style_key='num_epochs',
                          figname=figname,
                          use_error_bars=True,
                          subdir=subdir, multiprocess_lock=multiprocess_lock,
                          palette='viridis')
    plots.USE_ERRORBARS = False

high_d_input_edge_of_chaos_params = dict(N=200,
                                         num_epochs=40,
                                         num_train_samples_per_epoch=800,
                                         X_clusters=60,
                                         X_dim=200,
                                         num_classes=2,
                                         n_lag=10,
                                         g_radius=20,
                                         clust_sig=.02,
                                         input_scale=1,
                                         n_hold=1,
                                         n_out=1,
                                         loss='cce',
                                         optimizer='rmsprop',
                                         dt=.01,
                                         momentum=0,
                                         learning_rate=1e-3,
                                         batch_size=10,
                                         freeze_input=False,
                                         train_output_weights=True,
                                         network='vanilla_rnn',
                                         Win='orthog',
                                         patience_before_stopping=6000,
                                         hid_nonlin='tanh',
                                         model_seed=0,
                                         rerun=False)

# %% high-d input experiments with different hyperparameters
params_rnn_high_d = {
    'loss': ['cce', 'mse', 'mse_scalar'],
    'optimizer': ['sgd', 'rmsprop'],
    'learning_rate': [1e-3, 1e-4],
    'n_lag': [6, 10, 14],
    'N': [200, 300]
    }
ps_list_rnn_high_d_vals = list(itertools.product(*params_rnn_high_d.values()))
params_rnn_high_d_keys = params_rnn_high_d.keys()
params_rnn_high_d_keys_abbrev = ['loss', 'opt', 'lr', 'n_lag', 'N']
def run_rnn_high_d_input(param_set, multiprocess_lock=None):
    plots.USE_ERRORBARS = False
    subdir_prefix2 = Path('vanilla_rnn')
    tps = high_d_input_edge_of_chaos_params.copy()
    for i0, key in enumerate(params_rnn_high_d_keys):
        tps[key] = param_set[i0]
    seeds = list(range(5))
    tps_11 = tps.copy()
    tps_11['g_radius'] = 20
    tps_12 = tps.copy()
    tps_12['g_radius'] = 250
    figname = ''.join(key + '_' + str(val) + '_' for key, val in
                      zip(params_rnn_high_d_keys_abbrev, param_set))
    figname = figname[:-1]
    subdir = subdir_prefix2/'dim_high_d_experiments'
    plots.dim_over_layers([tps_11, tps_12], None, seeds, 'g_radius', None,
                          figname,
                          subdir=subdir_prefix/subdir,
                          multiprocess_lock=multiprocess_lock,
                          palette=chaos_palette)

# %% Training RNN on high-d data and freezing output weights
# high_d_input_strongly_chaotic_params['g_radius'] = 250
params_rnn_freeze = {
    'loss': ['cce', 'mse', 'mse_scalar'],
    'optimizer': ['sgd', 'rmsprop'],
    'learning_rate': [1e-3, 1e-4]
    }
ps_list_rnn_freeze_vals = list(itertools.product(*params_rnn_freeze.values()))
params_rnn_freeze_keys = params_rnn_freeze.keys()
params_rnn_freeze_keys_abbrev = ['loss', 'opt', 'lr']
def run_rnn_freeze_output(param_set, multiprocess_lock=None):
    plots.USE_ERRORBARS = True
    subdir_prefix2 = Path('vanilla_rnn')
    tps = high_d_input_edge_of_chaos_params.copy()
    for i0, key in enumerate(params_rnn_freeze_keys):
        tps[key] = param_set[i0]
    seeds = list(range(5))
    tps_11 = tps.copy()
    tps_11['train_output_weights'] = True
    tps_12 = tps.copy()
    tps_12['train_output_weights'] = False
    figname = ''.join(key + '_' + str(val) + '_' for key, val in
                      zip(params_rnn_freeze_keys_abbrev, param_set))
    figname = figname[:-1]
    plot_ps = (
        [tps_11, tps_12], None, seeds, 'train_output_weights', None, figname)
    subdir = subdir_prefix2/'dim_freeze_output_weights'
    plots.dim_through_training(*plot_ps, subdir=subdir_prefix/subdir,
                               multiprocess_lock=multiprocess_lock)

# %% Training RNN with noisy units
temp1 = {
    'loss': ['cce', 'mse', 'mse_scalar'],
    'optimizer': ['sgd', 'rmsprop'],
    'learning_rate': [1e-3, 1e-4],
    'dropout_p': [0, 0.05, 0.1, 0.2, 0.5],
    'unit_injected_noise': [0]
    }
temp2 = {
    'loss': ['cce', 'mse', 'mse_scalar'],
    'optimizer': ['sgd', 'rmsprop'],
    'learning_rate': [1e-3, 1e-4],
    'dropout_p': [0],
    'unit_injected_noise': [0.05, 0.1, 0.2, 0.5]
    }
# params_rnn_noisy_units =
ps_list_rnn_noisy_units_vals = list(itertools.product(*temp1.values())) \
                               + list(itertools.product(*temp2.values()))
params_rnn_noisy_units_keys = temp1.keys()
params_rnn_noisy_units_keys_abbrev = ['loss', 'opt', 'lr', 'dropout', 'noise']
def run_rnn_noisy_units(param_set, multiprocess_lock=None):
    plots.USE_ERRORBARS = False
    subdir_prefix2 = Path('vanilla_rnn')
    tps_high_d = high_d_input_edge_of_chaos_params.copy()
    for i0, key in enumerate(params_rnn_noisy_units_keys):
        tps_high_d[key] = param_set[i0]
    seeds = list(range(5))
    tps_11_high_d = tps_high_d.copy()
    tps_11_high_d['g_radius'] = 20
    tps_12_high_d = tps_high_d.copy()
    tps_12_high_d['g_radius'] = 250
    figname = ''.join(key + '_' + str(val) + '_' for key, val in
                      zip(params_rnn_noisy_units_keys_abbrev, param_set))
    figname = figname[:-1]
    subdir = subdir_prefix2/'dim_noisy_units'
    plots.dim_over_layers([tps_11_high_d, tps_12_high_d], None, seeds,
                          'g_radius', None, figname + '_X_dim_200',
                          subdir=subdir_prefix/subdir,
                          multiprocess_lock=multiprocess_lock,
                          palette=chaos_palette)

    tps_low_d = low_d_params.copy()
    for i0, key in enumerate(params_rnn_noisy_units_keys):
        tps_low_d[key] = param_set[i0]
    tps_11_low_d = tps_low_d.copy()
    tps_11_low_d['g_radius'] = 20
    tps_12_low_d = tps_low_d.copy()
    tps_12_low_d['g_radius'] = 250
    plots.dim_over_layers([tps_11_low_d, tps_12_low_d], None, seeds,
                          'g_radius', None, figname + '_X_dim_2',
                          subdir=subdir_prefix/subdir,
                          multiprocess_lock=multiprocess_lock,
                          palette=chaos_palette)

    tps_low_d_2neurons_11 = tps_low_d.copy()
    tps_low_d_2neurons_11['Win'] = 'diagonal_first_two'
    tps_low_d_2neurons_11['g_radius'] = 20
    tps_low_d_2neurons_12 = tps_low_d.copy()
    tps_low_d_2neurons_12['Win'] = 'diagonal_first_two'
    tps_low_d_2neurons_12['g_radius'] = 250
    plots.dim_over_layers([tps_low_d_2neurons_11, tps_low_d_2neurons_12], None,
                          seeds, 'g_radius', None,
                          figname + '_X_dim_2_ident',
                          subdir=subdir_prefix/subdir,
                          multiprocess_lock=multiprocess_lock,
                          palette=chaos_palette)

# %% Run things here
if __name__ == '__main__':
    lock = Lock()

    # Serial run
    [run_shallow_1(p, base_params, k) for k, p in enumerate(
    ps_list_shallow)]  #
    # [run_readout_noise(p, base_params, k) for k, p in
    #  enumerate(pvals_readout_noise)]
    # [run_deep_1(p, base_params, k) for k, p in enumerate(ps_vals_deep)]
    # [run_readout_noise_deep(p, base_params, k) for k, p in
    #  enumerate(pvals_readout_noise_deep)]
    # [run_lowd(p, low_d_params, k) for k, p in enumerate(params_lowd_loop)]
    # [run_lowd_chaos(p, low_d_chaos_params, k) for k, p in
    #  enumerate(params_lowd_chaos_loop)]
    # [run_rnn_high_d_input(p) for k, p in enumerate(params_rnn_high_d_vals)]
    # [run_rnn_freeze_output(p) for p in params_rnn_freeze_vals]
    # [run_rnn_noisy_units_output(p) for p in params_rnn_noisy_units_vals]

    # print("Setting up multiprocess.")
    # processes = []
    # processes += [Process( # Plots for Figure S5
    #     target=run_rnn_freeze_output,
    #     args=(p, lock)
    #     ) for p in ps_list_rnn_freeze_vals]
    # processes += [Process( # Plots for Figures S6 and S7
    #     target=run_shallow_1, args=(p, base_params, i0, lock)) for i0, p in
    #     enumerate(ps_list_shallow)]
    # processes += [Process( # Plots for Figures S6 and S7
    #     target=run_readout_noise, args=(p, base_params, i0, lock)
    #     ) for i0, p in enumerate(ps_list_readout_noise)]
    # processes += [Process( # Plots for Figures S8-S11
    #     target=run_recurrent, args=(p, base_params, i0, lock)) for i0, p in
    #     enumerate(ps_list_recurrent)]
    # processes += [Process( # Plots for Figures S8-S11
    #     target=run_readout_noise_recurrent, args=(p, base_params, i0, lock)
    #     ) for i0, p in enumerate(ps_list_readout_noise_recurrent)]
    # processes += [Process( # Plots for Figure S13
    #     target=run_rnn_high_d_input,
    #     args=(p, lock)
    #     ) for p in ps_list_rnn_high_d_vals]
    # processes += [Process( # Plots for Figures S14--S16, S17, and S22
    #     target=run_lowd, args=(p, low_d_params, i0, lock)
    #     ) for i0, p in enumerate(ps_list_lowd)]
    # processes += [Process( # Plots for Figure S18
    #     target=run_lowd_chaos, args=(p, low_d_chaos_params, i0, lock)
    #     ) for i0, p in enumerate(ps_list_lowd_chaos)]
    # processes += [Process( # Plots for Figure S18--S21
    #     target=run_rnn_noisy_units,
    #     args=(p, lock)
    #     ) for p in ps_list_rnn_noisy_units_vals]
    # print("Starting", len(processes), "processes")
    # [process.start() for process in processes]
    # print("Joining processes.")
    # [process.join() for process in processes]
