# %% Script for generating all of the figures used in the manuscript
# "Robust representations learned in RNNs through implicit balance of expansion and compression"

# %% Setting up the options and parameters

# FAST_RUN = False # Takes about 30 minutes
FAST_RUN = True  # Takes about 10 minutes

CLEAR_PREVIOUS_RUNS = False
# CLEAR_PREVIOUS_RUNS = True # Delete saved weights from previous runs

import plots

if FAST_RUN:
    seeds = [1]
else:
    seeds = list(range(5))

if CLEAR_PREVIOUS_RUNS:
    import shutil
    shutil.rmtree('../data/output')

# See initialize_and_train.initialize_and_train for a description of these parameters.
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
low_d_input_edge_of_chaos_params['num_epochs'] = 80
low_d_input_strongly_chaotic_params = high_d_input_edge_of_chaos_params.copy()
low_d_input_strongly_chaotic_params['g_radius'] = 250
low_d_input_strongly_chaotic_params['X_dim'] = 2
low_d_input_strongly_chaotic_params['num_epochs'] = 80

lyap_epochs_edge_of_chaos = [0, 1, 5, 10, 150]
lyap_epochs_strongly_chaotic = [0, 1, 5, 10, 150]
# %% Calls to plotting functions

# # %% Figure 1b (top)
g = high_d_input_edge_of_chaos_params['g_radius']
plots.lyaps([0], high_d_input_edge_of_chaos_params, lyap_epochs_edge_of_chaos,
            figname="fig_1b_top_g_{}_lyaps".format(g))
#
# # %% Figure 1b (bottom)
g = high_d_input_strongly_chaotic_params['g_radius']
plots.lyaps([0], high_d_input_strongly_chaotic_params, lyap_epochs_strongly_chaotic,
            figname="fig_1b_bottom_g_{}_lyaps".format(g))
#
# # %% Figure 1c (top)
g = high_d_input_edge_of_chaos_params['g_radius']
plots.snapshots_through_time(high_d_input_edge_of_chaos_params, subdir_name='fig_1c_top_snaps_g_{}'.format(g))
#
# # %% Figure 1c (bottom)
g = high_d_input_strongly_chaotic_params['g_radius']
plots.snapshots_through_time(high_d_input_strongly_chaotic_params, subdir_name='fig_1c_bottom_snaps_g_{}'.format(g))

# %% Figure 1d
hue_dictionary = {'g_radius': [20, 250]}
acc_and_loss_params = high_d_input_edge_of_chaos_params.copy()
acc_and_loss_params['num_epochs'] = 10
acc_and_loss_params['num_train_samples_per_epoch'] = 400
epochs = list(range(acc_and_loss_params['num_epochs']+1))
figname = 'fig_1d_acc_and_loss_over_training'
plots.acc_and_loss_over_training(acc_and_loss_params, seeds, epochs=epochs, hue_dictionary=hue_dictionary,
                                 figname=figname)

# %% Figure 1e
plots.dim_over_layers(seeds, [20, 250], high_d_input_edge_of_chaos_params,
                      figname="fig_1e_dim_over_time")
#
# # %% Figures 1f and 1g
plots.clust_holdout_over_layers(seeds, [20, 250], high_d_input_edge_of_chaos_params,
                                figname="fig_1f_1g_clust_holdout_over_time")


# %% Figure 2b (top)
g = low_d_input_edge_of_chaos_params['g_radius']
plots.lyaps([0], low_d_input_edge_of_chaos_params, lyap_epochs_edge_of_chaos, figname="fig_2b_top_lyaps_g_{}".format(g))

# %% Figure 2b (bottom)
g = low_d_input_strongly_chaotic_params['g_radius']
plots.lyaps([0], low_d_input_strongly_chaotic_params, lyap_epochs_strongly_chaotic,
            figname="fig_2b_bottom_lyaps_g_{}".format(g))

# %% Figure 2c (top)
g = low_d_input_edge_of_chaos_params['g_radius']
plots.snapshots_through_time(low_d_input_edge_of_chaos_params, subdir_name='fig_2c_top_snaps_g_{}'.format(g))

# %% Figure 2c (bottom)
g = low_d_input_strongly_chaotic_params['g_radius']
plots.snapshots_through_time(low_d_input_strongly_chaotic_params, subdir_name='fig_2c_bottom_snaps_g_{}'.format(g))

# %% Figure 2d
hue_dictionary = {'g_radius': [20, 250]}
acc_and_loss_params = low_d_input_edge_of_chaos_params.copy()
epochs = list(range(acc_and_loss_params['num_epochs']+1))
figname = 'fig_2d_acc_and_loss_over_training'
plots.acc_and_loss_over_training(acc_and_loss_params, seeds, epochs=epochs, hue_dictionary=hue_dictionary,
                                 figname=figname)

# %% Figure 2e
plots.dim_over_layers(seeds, [20, 250], low_d_input_edge_of_chaos_params,
                      figname="fig_2e_dim_over_time")

# %% Figures 2f and 2g
plots.clust_holdout_over_layers(seeds, [20, 250], low_d_input_edge_of_chaos_params,
                                figname="fig_2f_2g_clust_holdout_over_time")

# %%
low_d_2n_input_edge_of_chaos_params = low_d_input_edge_of_chaos_params.copy()
low_d_2n_input_edge_of_chaos_params['Win'] = 'diagonal_first_two'
low_d_2n_input_strongly_chaotic_params = low_d_input_strongly_chaotic_params.copy()
low_d_2n_input_strongly_chaotic_params['Win'] = 'diagonal_first_two'
# %% Figure 3b (top)
g = low_d_2n_input_edge_of_chaos_params['g_radius']
plots.lyaps([0], low_d_2n_input_edge_of_chaos_params, lyap_epochs_edge_of_chaos, figname="fig_3b_top_g_{}_lyaps".format(g))

# %% Figure 3b (bottom)
g = low_d_2n_input_strongly_chaotic_params['g_radius']
plots.lyaps([0], low_d_2n_input_strongly_chaotic_params, lyap_epochs_strongly_chaotic,
            figname="fig_3b_bottom_g_{}_lyaps".format(g))

# %% Figure 3c (top)
g = low_d_2n_input_edge_of_chaos_params['g_radius']
plots.snapshots_through_time(low_d_2n_input_edge_of_chaos_params,
                             subdir_name='fig_3c_top_snaps_g_{}'.format(g))

# %% Figure 3c (bottom)
g = low_d_2n_input_strongly_chaotic_params['g_radius']
plots.snapshots_through_time(low_d_2n_input_strongly_chaotic_params,
                             subdir_name='fig_3c_bottom_snaps_g_{}'.format(g))

# %% Figure 3d
hue_dictionary = {'g_radius': [20, 250]}
acc_and_loss_params = low_d_2n_input_edge_of_chaos_params.copy()
epochs = list(range(acc_and_loss_params['num_epochs']+1))
plots.acc_and_loss_over_training(acc_and_loss_params, seeds, epochs=epochs, hue_dictionary=hue_dictionary,
                                 figname='fig_3d_acc_and_loss_over_training')

# %% Figure 3e
plots.dim_over_layers(seeds, [20, 250], low_d_2n_input_edge_of_chaos_params,
                      figname="fig_3e_dim_over_time")

# %% Figures 3f and 3g
plots.clust_holdout_over_layers(seeds, [20, 250], low_d_2n_input_edge_of_chaos_params,
                                figname="fig_3f_3g_clust_holdout_over_time")
