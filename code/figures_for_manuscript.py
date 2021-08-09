# %% Script for generating all of the figures used in the manuscript
# "Robust representations learned in RNNs through implicit balance of
# expansion and compression"

# %% Setting up the options and parameters

# FAST_RUN = False # Takes about 30 minutes
FAST_RUN = True  # Takes about 10 minutes

CLEAR_PREVIOUS_RUNS = False
# CLEAR_PREVIOUS_RUNS = True # Delete saved weights from previous runs

import plots
import seaborn as sns


if FAST_RUN:
    seeds = [1]
    # seeds = [1,2,3]
    # seeds = [1, 2]
else:
    seeds = list(range(5))

if CLEAR_PREVIOUS_RUNS:
    import shutil


    shutil.rmtree('../data/output')

# See initialize_and_train.initialize_and_train for a description of these
# parameters.
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

chaos_color_0 = [0, 160/255, 160/255]
chaos_color_1 = [233/255, 38/255, 41/255]
chaos_colors = [chaos_color_0, chaos_color_1]
chaos_palette = sns.color_palette(chaos_colors)


# %% Helper function
def split_params_chaos(params):
    params = params.copy()
    params_edge_of_chaos = params.copy()
    params_edge_of_chaos['g_radius'] = 20
    params_strongly_chaotic = params.copy()
    params_strongly_chaotic['g_radius'] = 250
    return params_edge_of_chaos, params_strongly_chaotic


def split_params_training(params):
    params = params.copy()
    params_before_training = params.copy()
    params_before_training['num_epochs'] = 0
    return params_before_training, params


def split_params_chaos_and_training(params):
    params = params.copy()
    eoc, str_ch = split_params_chaos(params)
    eoc_before = split_params_training(eoc)[0]
    str_ch_before = split_params_training(str_ch)[0]
    return (eoc, str_ch), (eoc_before, str_ch_before)


# %% Calls to plotting functions

# # %% Figure 2b (top)
g = high_d_input_edge_of_chaos_params['g_radius']
plots.lyaps([0], high_d_input_edge_of_chaos_params, lyap_epochs_edge_of_chaos,
            figname="fig_2b_top_g_{}_lyaps".format(g),
            subdir="fig2")
#
# # %% Figure 2b (bottom)
g = high_d_input_strongly_chaotic_params['g_radius']
plots.lyaps([0], high_d_input_strongly_chaotic_params,
            lyap_epochs_strongly_chaotic,
            figname="fig_2b_bottom_g_{}_lyaps".format(g),
            subdir="fig2")

# %% Figure 2c (top)
g = high_d_input_edge_of_chaos_params['g_radius']
plots.snapshots_through_time(high_d_input_edge_of_chaos_params,
                             subdir='fig2/fig_2c_top_snaps_g_{}'.format(g))
#
# # %% Figure 2c (bottom)
g = high_d_input_strongly_chaotic_params['g_radius']
plots.snapshots_through_time(high_d_input_strongly_chaotic_params,
                             subdir='fig2/fig_2c_bottom_snaps_g_{}'.format(g))

# %% Figure 2d
hue_dictionary = {'g_radius': [20, 250]}
acc_params = high_d_input_edge_of_chaos_params.copy()
acc_params['num_epochs'] = 10
acc_params['num_train_samples_per_epoch'] = 400
chaos_params = split_params_chaos(acc_params)
epochs = list(range(acc_params['num_epochs'] + 1))
figname = 'fig_2d_acc_over_training'
plots.acc_over_training(chaos_params, None, seeds,
                        hue_key='g_radius', figname=figname, subdir='fig2',
                        palette=chaos_palette)

# %% Figure 2e
fig2e_params = high_d_input_edge_of_chaos_params.copy()
chaos_params, chaos_params_epoch_0 = split_params_chaos_and_training(
    fig2e_params)
plots.dim_over_layers(chaos_params, chaos_params_epoch_0, seeds,
                      hue_key='g_radius', style_key='num_epochs',
                      figname="fig_2e_dim_over_time",
                      subdir="fig2",
                      style_order=[fig2e_params['num_epochs'], 0],
                      palette=chaos_palette)
#
# # %% Figures 2f and 2g
figname="fig_2f_2g_clust_holdout_over_time"
plots.clust_holdout_over_layers(chaos_params, chaos_params_epoch_0, seeds,
                                hue_key='g_radius', style_key='num_epochs',
                                figname=figname, subdir="fig2",
                                style_order=[fig2e_params['num_epochs'], 0],
                                palette=chaos_palette)

# %% Figure 3b (top)
g = low_d_input_edge_of_chaos_params['g_radius']
plots.lyaps([0], low_d_input_edge_of_chaos_params, lyap_epochs_edge_of_chaos,
            figname="fig_3b_top_lyaps_g_{}".format(g))

# %% Figure 3b (bottom)
g = low_d_input_strongly_chaotic_params['g_radius']
plots.lyaps([0], low_d_input_strongly_chaotic_params,
            lyap_epochs_strongly_chaotic,
            figname="fig_3b_bottom_lyaps_g_{}".format(g))

# %% Figure 3c (top)
g = low_d_input_edge_of_chaos_params['g_radius']
plots.snapshots_through_time(low_d_input_edge_of_chaos_params,
                             subdir='fig3/fig_3c_top_snaps_g_{}'.format(g))

# %% Figure 3c (bottom)
g = low_d_input_strongly_chaotic_params['g_radius']
plots.snapshots_through_time(low_d_input_strongly_chaotic_params,
                             subdir='fig3/fig_3c_bottom_snaps_g_{}'.format(g))

# %% Figure 2d
acc_params = low_d_input_edge_of_chaos_params.copy()
chaos_params = split_params_chaos(acc_params)
figname = 'fig_3d_acc_over_training'
plots.acc_over_training(chaos_params, None, seeds,
                        hue_key='g_radius',
                        figname=figname, subdir='fig3',
                        palette=chaos_palette)

# %% Figure 2e
fig3e_params = low_d_input_edge_of_chaos_params.copy()
chaos_params, chaos_params_epoch_0 = split_params_chaos_and_training(
    fig3e_params)
plots.dim_over_layers(chaos_params, chaos_params_epoch_0, seeds,
                      hue_key='g_radius', style_key='num_epochs',
                      figname="fig_3e_dim_over_time",
                      subdir='fig3',
                      style_order=[fig3e_params['num_epochs'], 0],
                      palette=chaos_palette)

# %% Figures 3f and 3g
plots.clust_holdout_over_layers(chaos_params, chaos_params_epoch_0, seeds,
                                hue_key='g_radius', style_key='num_epochs',
                                figname="fig_3f_3g_clust_holdout_over_time",
                                subdir='fig3',
                                style_order=[fig3e_params['num_epochs'], 0],
                                palette=chaos_palette)

# %%
low_d_2n_input_edge_of_chaos_params = low_d_input_edge_of_chaos_params.copy()
low_d_2n_input_edge_of_chaos_params['Win'] = 'diagonal_first_two'
low_d_2n_input_strongly_chaotic_params = \
    low_d_input_strongly_chaotic_params.copy()
low_d_2n_input_strongly_chaotic_params['Win'] = 'diagonal_first_two'
# %% Figure 4b (top)
g = low_d_2n_input_edge_of_chaos_params['g_radius']
plots.lyaps([0], low_d_2n_input_edge_of_chaos_params, lyap_epochs_edge_of_chaos,
            figname="fig_4b_top_g_{}_lyaps".format(g), subdir="fig4")

# %% Figure 4b (bottom)
g = low_d_2n_input_strongly_chaotic_params['g_radius']
plots.lyaps([0], low_d_2n_input_strongly_chaotic_params,
            lyap_epochs_strongly_chaotic,
            figname="fig_4b_bottom_g_{}_lyaps".format(g), subdir="fig4")

# %% Figure 4c (top)
g = low_d_2n_input_edge_of_chaos_params['g_radius']
plots.snapshots_through_time(low_d_2n_input_edge_of_chaos_params,
                             subdir='fig4/fig_4c_top_snaps_g_{}'.format(g))

# %% Figure 4c (bottom)
g = low_d_2n_input_strongly_chaotic_params['g_radius']
plots.snapshots_through_time(low_d_2n_input_strongly_chaotic_params,
                             subdir='fig4/fig_4c_bottom_snaps_g_{}'.format(g))

# %% Figure 4d
acc_params = low_d_2n_input_edge_of_chaos_params.copy()
chaos_params = split_params_chaos(acc_params)
plots.acc_over_training(chaos_params, None, seeds, hue_key='g_radius',
                        figname='fig_4d_acc_over_training',
                        subdir='fig4', palette=chaos_palette)

# %% Figure 4e
fig4e_params = low_d_2n_input_edge_of_chaos_params.copy()
chaos_params, chaos_params_epoch_0 = split_params_chaos_and_training(
    fig4e_params)
plots.dim_over_layers(chaos_params, chaos_params_epoch_0, seeds=seeds,
                      hue_key='g_radius', style_key='num_epochs',
                      figname="fig_4e_dim_over_time", subdir='fig4',
                      style_order=[fig4e_params['num_epochs'], 0],
                      palette=chaos_palette)

# %% Figures 4f and 4g
figname = "fig_4f_4g_clust_holdout_over_time"
plots.clust_holdout_over_layers(chaos_params, chaos_params_epoch_0, seeds=seeds,
                                hue_key='g_radius', style_key='num_epochs',
                                figname="fig_4e_dim_over_time", subdir='fig4',
                                style_order=[fig4e_params['num_epochs'], 0],
                                palette=chaos_palette)

# %% Figure 5a
fig5a_params = high_d_input_edge_of_chaos_params.copy()
fig5a_params['loss'] = 'mse'
# A higher learning rate is needed to get the strongly chaotic network's
# dimensionality to decrease.
fig5a_params['learning_rate'] = 5e-3
fig5a_params['num_epochs'] = 100
chaos_params, chaos_params_epoch_0 = split_params_chaos_and_training(
    fig5a_params)
plots.dim_over_layers(chaos_params, chaos_params_epoch_0, seeds=seeds,
                      hue_key='g_radius', style_key='num_epochs',
                      figname="fig_5a", subdir='fig5', style_order=[100, 0],
                      palette=chaos_palette)

# %% Figure 5b
fig5b_params = low_d_input_edge_of_chaos_params.copy()
fig5b_params['loss'] = 'mse'
# A higher learning rate is needed to get the strongly chaotic network's
# dimensionality to decrease.
fig5b_params['learning_rate'] = 5e-3
fig5b_params['num_epochs'] = 100
chaos_params, chaos_params_epoch_0 = split_params_chaos_and_training(
    fig5b_params)
plots.dim_over_layers(chaos_params, chaos_params_epoch_0, seeds=seeds,
                      hue_key='g_radius', style_key='num_epochs',
                      figname="fig_5b", subdir='fig5', style_order=[100, 0],
                      palette=chaos_palette)
