# %% Script for generating all of the figures used in the manuscript
# "Robust representations learned in RNNs through implicit balance of expansion and compression"

import plots

# See initialize_and_train.initialize_and_train for a description of these parameters.
high_d_input_edge_of_chaos_params = dict(N=200,
                                         num_epochs=40,
                                         num_train_samples_per_epoch=1250,
                                         X_clusters=60,
                                         X_dim=200,
                                         num_classes=2,
                                         n_lag=9,
                                         g_radius=5,
                                         clust_sig=.02,
                                         n_hold=1,
                                         n_out=1,
                                         loss='cce',
                                         optimizer='rmsprop',
                                         dt=.01,
                                         learning_rate=1e-4,
                                         batch_size=10,
                                         freeze_input=False,
                                         network='vanilla_rnn',
                                         Win='orthog',
                                         patience_before_stopping=6000,
                                         hid_nonlin='tanh',
                                         model_seed=0,
                                         rerun=False)

high_d_input_strongly_chaotic_params = high_d_input_edge_of_chaos_params.copy()
high_d_input_strongly_chaotic_params['g_radius'] = 250
low_d_input_edge_of_chaos_params = high_d_input_edge_of_chaos_params.copy()
low_d_input_edge_of_chaos_params['X_dim'] = 2
low_d_input_strongly_chaotic_params = high_d_input_edge_of_chaos_params.copy()
low_d_input_strongly_chaotic_params['g_radius'] = 250
low_d_input_strongly_chaotic_params['X_dim'] = 2

# seeds = range(30)
seeds = range(3)

# %% Figure 1b (top)
plots.lyaps([0], high_d_input_edge_of_chaos_params, [0, 5, 15, 35, 40], figname="lyaps")

# %% Figure 1b (bottom)
plots.lyaps([0], high_d_input_edge_of_chaos_params, [0, 5, 15, 35, 40], figname="lyaps")

# %% Figure 1c (top)
plots.snapshots_through_time(high_d_input_edge_of_chaos_params)

# %% Figure 1c (bottom)
plots.snapshots_through_time(high_d_input_strongly_chaotic_params)

# %% Figure 1d
hue_dictionary={'g_radius': [5,250]}
hue_target = ('g_radius', None)
acc_and_loss_params = high_d_input_edge_of_chaos_params.copy()
acc_and_loss_params['num_epochs'] = 10
acc_and_loss_params['num_train_samples_per_epoch'] = 200
plots.acc_and_loss_over_training(acc_and_loss_params, seeds, hue_dictionary, hue_target)

# %% Figure 1e
plots.dim_over_layers(seeds, [5, 250], high_d_input_edge_of_chaos_params)

# %% Figures 1f and 1g
plots.clust_holdout_over_layers(seeds, [5, 250], high_d_input_edge_of_chaos_params)


# %% Figure 2b (top)
plots.lyaps([0], low_d_input_edge_of_chaos_params, [0, 5, 15, 35, 40], figname="lyaps")

# %% Figure 2b (bottom)
plots.lyaps([0], low_d_input_edge_of_chaos_params, [0, 5, 15, 35, 40], figname="lyaps")

# %% Figure 2c (top)
plots.snapshots_through_time(low_d_input_edge_of_chaos_params)

# %% Figure 2c (bottom)
plots.snapshots_through_time(low_d_input_strongly_chaotic_params)

# %% Figure 2d
hue_dictionary={'g_radius': [5,250]}
hue_target = ('g_radius', None)
acc_and_loss_params = low_d_input_edge_of_chaos_params.copy()
acc_and_loss_params['num_epochs'] = 10
acc_and_loss_params['num_train_samples_per_epoch'] = 200
plots.acc_and_loss_over_training(acc_and_loss_params, seeds, hue_dictionary, hue_target)

# %% Figure 2e
plots.dim_over_layers(seeds, [5, 250], low_d_input_edge_of_chaos_params)

# %% Figures 2f and 2g
plots.clust_holdout_over_layers(seeds, [5, 250], low_d_input_edge_of_chaos_params)
