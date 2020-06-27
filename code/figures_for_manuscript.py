# %% Script for generating all of the figures used in the manuscript
# "Robust representations learned in RNNs through implicit balance of expansion and compression"

# FAST_RUN = False # Takes about 40 minutes
FAST_RUN = True  # Takes about 8 minutes

CLEAR_PREVIOUS_RUNS = False
# CLEAR_PREVIOUS_RUNS = True # Delete saved weights from previous runs

import plots

if FAST_RUN:
    seeds = [0]
else:
    seeds = list(range(5))

if CLEAR_PREVIOUS_RUNS:
    import shutil

    shutil.rmtree('../data/output')

# See initialize_and_train.initialize_and_train for a description of these parameters.
high_d_input_edge_of_chaos_params = dict(N=200,
                                         num_epochs=150,
                                         num_train_samples_per_epoch=800,
                                         X_clusters=60,
                                         X_dim=200,
                                         num_classes=2,
                                         n_lag=11,
                                         g_radius=5,
                                         clust_sig=.02,
                                         input_scale=1,
                                         n_hold=1,
                                         n_out=1,
                                         loss='cce',
                                         optimizer='rmsprop',
                                         momentum=0,
                                         dt=.01,
                                         learning_rate=1e-3,
                                         batch_size=10,
                                         freeze_input=False,
                                         network='vanilla_rnn',
                                         Win='orthog',  # todo: refactor code so this can be defined here.
                                         Wrec_rand_proportion=.2,
                                         patience_before_stopping=6000,
                                         hid_nonlin='tanh',
                                         model_seed=0,
                                         rerun=True)

high_d_input_strongly_chaotic_params = high_d_input_edge_of_chaos_params.copy()
high_d_input_strongly_chaotic_params['g_radius'] = 250
low_d_input_edge_of_chaos_params = high_d_input_edge_of_chaos_params.copy()
low_d_input_edge_of_chaos_params['X_dim'] = 2
low_d_input_edge_of_chaos_params['num_epochs'] = 150
low_d_input_strongly_chaotic_params = high_d_input_edge_of_chaos_params.copy()
low_d_input_strongly_chaotic_params['g_radius'] = 250
low_d_input_strongly_chaotic_params['X_dim'] = 2
low_d_input_strongly_chaotic_params['num_epochs'] = 150

# # %% Figure 1b (top)
# plots.lyaps([0], high_d_input_edge_of_chaos_params, [0, 5, 15, 35, 40], figname="lyaps")
# high_d_input_edge_of_chaos_params['num_epochs'] = 1
# plots.lyaps([0], high_d_input_edge_of_chaos_params, [0, 1], figname="lyaps")
#
# # %% Figure 1b (bottom)
# plots.lyaps([0], high_d_input_strongly_chaotic_params, [0, 5, 15, 35, 40], figname="lyaps")
#
# # %% Figure 1c (top)
# plots.snapshots_through_time(high_d_input_edge_of_chaos_params)
#
# # %% Figure 1c (bottom)
# plots.snapshots_through_time(high_d_input_strongly_chaotic_params)
#
# # %% Figure 1d
hue_dictionary = {'g_radius': [5, 250]}
hue_target = ('g_radius', None)
acc_and_loss_params = high_d_input_edge_of_chaos_params.copy()
acc_and_loss_params['num_epochs'] = 15
acc_and_loss_params['num_train_samples_per_epoch'] = 400
epoch = list(range(16))
# plots.acc_and_loss_over_training(acc_and_loss_params, seeds, hue_dictionary, hue_target, epoch_list=epoch,
#                                  epoch_plot=epoch)
#
# # %% Figure 1e
# plots.dim_over_layers(seeds, [5, 250], high_d_input_edge_of_chaos_params)
#
# # %% Figures 1f and 1g
# plots.clust_holdout_over_layers(seeds, [5, 250], high_d_input_edge_of_chaos_params)


# %% Figure 2b (top)
# plots.lyaps([0], low_d_input_edge_of_chaos_params, [0, 5, 15, 35, 40], figname="lyaps")

# %% Figure 2b (bottom)
# plots.lyaps([0], low_d_input_strongly_chaotic_params, [0, 5, 15, 35, 40], figname="lyaps")

# %% Figure 2c (top)
# plots.snapshots_through_time(low_d_input_edge_of_chaos_params)

# %% Figure 2c (bottom)
# plots.snapshots_through_time(low_d_input_strongly_chaotic_params)

# %% Figure 2d
hue_dictionary = {'g_radius': [5, 250]}
hue_target = ('g_radius', None)
acc_and_loss_params = low_d_input_edge_of_chaos_params.copy()
acc_and_loss_params['num_train_samples_per_epoch'] = 800
# epochs = list(range(1))
epochs = list(range(0, 81, 4))
# plots.acc_and_loss_over_training(acc_and_loss_params, seeds, hue_dictionary, hue_target, epoch_list=epochs,
#                                  epoch_plot=epochs)

# %% Figure 2e
plots.dim_over_layers(seeds, [5, 250], low_d_input_edge_of_chaos_params)

# %% Figures 2f and 2g
# plots.clust_holdout_over_layers(seeds, [5, 250], low_d_input_edge_of_chaos_params)

# %%
low_d_2n_input_edge_of_chaos_params = low_d_input_edge_of_chaos_params.copy()
low_d_2n_input_edge_of_chaos_params['Win'] = 'diagonal_first_two'
low_d_2n_input_strongly_chaotic_params = low_d_input_strongly_chaotic_params.copy()
low_d_2n_input_strongly_chaotic_params['Win'] = 'diagonal_first_two'
# %% Figure 3b (top)
# plots.lyaps([0], low_d_2n_input_edge_of_chaos_params, [0, 5, 15, 35, 40], figname="lyaps")

# %% Figure 3b (bottom)
# plots.lyaps([0], low_d_2n_input_strongly_chaotic_params, [0, 5, 15, 35, 40], figname="lyaps")

# %% Figure 3c (top)
# plots.snapshots_through_time(low_d_2n_input_edge_of_chaos_params)

# %% Figure 3c (bottom)
# plots.snapshots_through_time(low_d_2n_input_strongly_chaotic_params)

# %% Figure 3d
hue_dictionary = {'g_radius': [5, 250]}
hue_target = ('g_radius', None)
acc_and_loss_params = low_d_2n_input_edge_of_chaos_params.copy()
acc_and_loss_params['num_train_samples_per_epoch'] = 1250
epochs = list(range(41))
# plots.acc_and_loss_over_training(acc_and_loss_params, seeds, hue_dictionary, hue_target, epoch_list=epochs,
#                                  epoch_plot=epochs)

# %% Figure 3e
# plots.dim_over_layers(seeds, [5, 250], low_d_2n_input_edge_of_chaos_params)

# %% Figures 3f and 3g
# plots.clust_holdout_over_layers(seeds, [5, 250], low_d_2n_input_edge_of_chaos_params)
