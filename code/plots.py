import copy
import pickle as pkl
from scipy import stats as st
import torch
from sklearn import svm
from sklearn import linear_model
import pandas as pd
import seaborn as sns
import warnings
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import juggle_axes
import lyap
import model_loader_utils as loader
import initialize_and_train
import utils

## Functions for computing means and error bars for the plots. 95% confidence intervals and means are currently
# implemented in this code. The commented out code is for using a gamma distribution to compute these, but uses a
# custom version of seaborn plotting library to plot.

USE_ERRORBARS = True

# def ci_acc(vals):
#     median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1., shift=-.0001,
#                                       reflect=True)
#     return bounds[1], bounds[0]
#
ci_acc = 70
# def est_acc(vals):
#     median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1., shift=-.0001,
#                                       reflect=True)
#     return median
est_acc = "mean"
# est_acc = "median"

def ci_dim(vals):
    median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1)
    return bounds[1], bounds[0]
# ci_dim = 95

def est_dim(vals):
    median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1)
    return median
# est_dim = "mean"

def point_replace(a_string):
    a_string = str(a_string)
    return a_string.replace(".", "p")

def get_color(x, cmap=plt.cm.plasma):
    """Get normalized color assignments based on input data x and colormap cmap."""
    mag = torch.max(x) - torch.min(x)
    x_norm = (x.float() - torch.min(x))/mag
    return cmap(x_norm)

def median_and_bound(samples, perc_bound, dist_type='gamma', loc=0., shift=0, reflect=False):
    """Get median and probability mass intervals for a gamma distribution fit of samples."""
    samples = np.array(samples)

    def do_reflect(x, center):
        return -1*(x - center) + center

    if dist_type == 'gamma':
        if np.sum(samples[0] == samples) == len(samples):
            median = samples[0]
            interval = [samples[0], samples[0]]
            return median, interval

        if reflect:
            samples_reflected = do_reflect(samples, loc)
            shape_ps, loc_fit, scale = st.gamma.fit(samples_reflected, floc=loc + shift)
            median_reflected = st.gamma.median(shape_ps, loc=loc, scale=scale)
            interval_reflected = np.array(st.gamma.interval(perc_bound, shape_ps, loc=loc, scale=scale))
            median = do_reflect(median_reflected, loc)
            interval = do_reflect(interval_reflected, loc)
        else:
            shape_ps, loc, scale = st.gamma.fit(samples, floc=loc + shift)
            median = st.gamma.median(shape_ps, loc=loc, scale=scale)
            interval = np.array(st.gamma.interval(perc_bound, shape_ps, loc=loc, scale=scale))
    else:
        raise ValueError("Distribution option (dist_type) not recognized.")

    return median, interval

## Set parameters for figure aesthetics
plt.rcParams['font.size'] = 6
plt.rcParams['font.size'] = 6
plt.rcParams['lines.markersize'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.titlesize'] = 8

# Colormaps
class_style = 'color'
cols11 = np.array([90, 100, 170])/255
cols12 = np.array([37, 50, 120])/255
cols21 = np.array([250, 171, 62])/255
cols22 = np.array([156, 110, 35])/255
cmap_activation_pnts = mcolors.ListedColormap([cols11, cols21])
cmap_activation_pnts_edge = mcolors.ListedColormap([cols12, cols22])

rasterized = False
dpi = 800
ext = 'pdf'

# Default figure size
figsize = (1.5, 1.2)
ax_pos = (0, 0, 1, 1)

def make_fig(figsize=figsize, ax_pos=ax_pos):
    """Create figure."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_pos)
    return fig, ax

def out_fig(fig, name, train_params, subfolder='', show=False, save=True, axis_type=0, name_order=0, data=None):
    """ Save figure."""
    folder = '../results/figs/Win_{}/'.format(train_params['Win'])
    os.makedirs('../results/figs/', exist_ok=True)
    os.makedirs(folder, exist_ok=True)
    nonlinearity = train_params['hid_nonlin']
    loss = train_params['loss']
    X_dim = train_params['X_dim']
    ax = fig.axes[0]
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    ax.set_rasterized(rasterized)
    if axis_type == 1:
        ax.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off
    elif axis_type == 2:
        ax.axis('off')
    if name_order == 0:
        fig_name = folder + subfolder + point_replace(
            '{}_Xdim_{}_{}_{}'.format(name, X_dim, nonlinearity, loss))
    else:
        fig_name = folder + subfolder + point_replace('Xdim_{}_{}_{}_{}'.format(X_dim, name, nonlinearity, loss))
    if save:
        os.makedirs(folder + subfolder, exist_ok=True)
        fig_file = fig_name + '.' + ext
        print(f"Saving figure to {fig_file}")
        fig.savefig(fig_file, dpi=dpi, transparent=True, bbox_inches='tight')

    if show:
        fig.tight_layout()
        fig.show()

    if data is not None:
        os.makedirs(folder + subfolder + 'data/', exist_ok=True)
        with open(folder + subfolder + 'data/Xdim_{}_{}_data'.format(X_dim, name), 'wb') as fid:
            pkl.dump(data, fid, protocol=4)

def snapshots_through_time(train_params, subdir_name="snaps/"):
    """
    Plot PCA snapshots of the representation through time.

    Parameters
    ----------
    train_params : dict
        Dictionary of training parameters that specify the model and dataset to use for training.

    """
    X_dim = train_params['X_dim']
    FEEDFORWARD = train_params['network'] == 'feedforward'
    SUBFOLDER = train_params['network'] + '/'

    num_pnts_dim_red = 800
    num_plot = 600

    train_params_loc = copy.deepcopy(train_params)

    model, params, run_dir = initialize_and_train.initialize_and_train(**train_params_loc)

    class_datasets = params['datasets']
    class_datasets['train'].max_samples = num_pnts_dim_red
    torch.manual_seed(train_params_loc['model_seed'])
    X, Y = class_datasets['train'][:]

    if FEEDFORWARD:
        T = 10
        y = Y
        X0 = X
    else:
        T = 30
        X = utils.extend_input(X, T + 2)
        X0 = X[:, 0]
        y = Y[:, -1]
    loader.load_model_from_epoch_and_dir(model, run_dir, 0, 0)
    hid_0 = [X0]
    hid_0 += model.get_post_activations(X)[:-1]
    loader.load_model_from_epoch_and_dir(model, run_dir, -1, 0)
    hid = [X0]
    hid += model.get_post_activations(X)[:-1]

    coloring = get_color(y, cmap_activation_pnts)[:num_plot]
    edge_coloring = get_color(y, cmap_activation_pnts_edge)[:num_plot]

    ## Now get principal components (pcs) and align them from time point to time point
    pcs = []
    p_track = 0
    norm = np.linalg.norm
    for i1 in range(len(hid)):
        pc = utils.get_pcs_covariance(hid[i1], [0, 1])
        if i1 > 0:

            # Check for the best alignment
            pc_flip_x = pc.clone()
            pc_flip_x[:, 0] = -pc_flip_x[:, 0]
            pc_flip_y = pc.clone()
            pc_flip_y[:, 1] = -pc_flip_y[:, 1]
            pc_flip_both = pc.clone()
            pc_flip_both[:, 0] = -pc_flip_both[:, 0]
            pc_flip_both[:, 1] = -pc_flip_both[:, 1]
            difference0 = norm(p_track - pc)
            difference1 = norm(p_track - pc_flip_x)
            difference2 = norm(p_track - pc_flip_y)
            difference3 = norm(p_track - pc_flip_both)
            amin = np.argmin([difference0, difference1, difference2, difference3])
            if amin == 1:
                pc[:, 0] = -pc[:, 0]
            elif amin == 2:
                pc[:, 1] = -pc[:, 1]
            elif amin == 3:
                pc[:, 0] = -pc[:, 0]
                pc[:, 1] = -pc[:, 1]
        p_track = pc.clone()
        pcs.append(pc[:num_plot])

    def take_snap(i0, scat, dim=2, border=False):
        hid_pcs_plot = pcs[i0][:, :dim].numpy()

        xm = np.min(hid_pcs_plot[:, 0])
        xM = np.max(hid_pcs_plot[:, 0])
        ym = np.min(hid_pcs_plot[:, 1])
        yM = np.max(hid_pcs_plot[:, 1])
        xc = (xm + xM)/2
        yc = (ym + yM)/2
        hid_pcs_plot[:, 0] = hid_pcs_plot[:, 0] - xc
        hid_pcs_plot[:, 1] = hid_pcs_plot[:, 1] - yc
        if class_style == 'shape':
            scat[0].set_offsets(hid_pcs_plot)
        else:
            if dim == 3:
                scat._offsets3d = juggle_axes(*hid_pcs_plot[:, :dim].T, 'z')
            else:
                scat.set_offsets(hid_pcs_plot)

        xm = np.min(hid_pcs_plot[:, 0])
        xM = np.max(hid_pcs_plot[:, 0])
        ym = np.min(hid_pcs_plot[:, 1])
        yM = np.max(hid_pcs_plot[:, 1])
        max_extent = max(xM - xm, yM - ym)
        max_extent_arg = xM - xm > yM - ym
        if dim == 2:
            x_factor = .4
            if max_extent_arg:
                ax.set_xlim([xm - x_factor*max_extent, xM + x_factor*max_extent])
                ax.set_ylim([xm - .1*max_extent, xM + .1*max_extent])
            else:
                ax.set_xlim([ym - x_factor*max_extent, yM + x_factor*max_extent])
                ax.set_ylim([ym - .1*max_extent, yM + .1*max_extent])
        else:
            if max_extent_arg:
                ax.set_xlim([xm - .1*max_extent, xM + .1*max_extent])
                ax.set_ylim([xm - .1*max_extent, xM + .1*max_extent])
                ax.set_zlim([xm - .1*max_extent, xM + .1*max_extent])
            else:
                ax.set_xlim([ym - .1*max_extent, yM + .1*max_extent])
                ax.set_ylim([ym - .1*max_extent, yM + .1*max_extent])
                ax.set_zlim([ym - .1*max_extent, yM + .1*max_extent])

        if dim == 3:
            out_fig(fig, "snapshot_{}".format(i0-1), train_params_loc,
                    subfolder=SUBFOLDER + subdir_name + '/', axis_type=2,
                    name_order=1)
        else:
            out_fig(fig, "snapshot_{}".format(i0-1), train_params_loc,
                    subfolder=SUBFOLDER + subdir_name + '/', axis_type=2,
                    name_order=1)
        return scat,

    dim = 2
    hid_pcs_plot = pcs[0]
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
    else:
        fig, ax = make_fig()
    ax.grid(False)

    scat = ax.scatter(*hid_pcs_plot[:num_plot, :dim].T, c=coloring,
                      edgecolors=edge_coloring, s=10, linewidths=.65)

    if FEEDFORWARD:
        snap_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    else:
        snap_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 21, 26, 31])
    for i0 in snap_idx:
        take_snap(i0, scat, dim=dim, border=False)

def acc_and_loss_over_training(train_params, seeds, epochs, hue_dictionary=None, hue_legend_key=None, figname=None,
                               **plot_kwargs):
    """
    Plot accuracy over training. hue_dictionary is an optional dictionary with a single entry that allows you to
    specify a parameter for which to plot multiple lines with different hues on the same plot.

    Parameters
    ----------
    train_params : dict
        Dictionary of training parameters that specify the model and dataset to use for training.
    seeds : List[int]
        List of random number seeds to use for generating instantiations of the model and dataset. Variation over
        these seeds is used to plot error bars.
    epochs : List[int]
        Epochs at which to plot the accuracy.
    hue_dictionary : Optional[dict]
        Dictionary specifying values that will result in a different line with a different hue. As an example, if
        hue_dictionary = {key: [value_1, value_2]}, then key is a parameter that will be set to value_1 and then
        value_2. Setting key to value_1 results in the plotting of a line with one hue, and setting key to value_2
        results in the plotting of a line with another hue, so that the final plot has two lines, each with a
        different hue. key should be a key in train_params.
    hue_legend_key : Optional[str]
        The key in hue_dictionary to use for the legend of the plot. If hue_str=None, this is set to the first key of
        hue_dictionary.
    figname : Optional[str]
        Name of the figure to save.

    """
    if figname is None:
        figname = 'acc_and_loss_over_training'
    if hue_legend_key is None:
        hue_legend_key = list(hue_dictionary.keys())[0]

    def generate_data_table(train_params, seeds, hue_dictionary, hue_legend_key, epochs):
        train_params_loc = copy.copy(train_params)
        NUM_SAMPLES = 1000
        spe = train_params_loc['num_train_samples_per_epoch']
        FEEDFORWARD = train_params_loc['network'] == 'feedforward'

        if hue_dictionary is None:
            hue_dictionary_None = True
            hue_keys = []
            loss_and_acc_table = pd.DataFrame(columns=['seed', 'num_training_samples', 'accuracy'])
            num_hues = 1
        else:
            hue_dictionary_None = False
            hue_keys = list(hue_dictionary.keys())
            loss_and_acc_table = pd.DataFrame(columns=['seed', 'num_training_samples', 'accuracy', hue_legend_key])
            num_hues = len(hue_dictionary[hue_keys[0]])

        for hue_idx in range(num_hues):
            for key in hue_keys:
                train_params_loc[key] = hue_dictionary[key][hue_idx]

            for i1, seed in enumerate(seeds):
                train_params_loc['model_seed'] = seed
                torch.manual_seed(seed)
                model, params, run_dir = initialize_and_train.initialize_and_train(**train_params_loc)

                testset = params['datasets']['train']
                X = []
                Y = []
                for k in range(NUM_SAMPLES):
                    x, y = testset[k]
                    X.append(x)
                    if FEEDFORWARD:
                        Y.append(y)
                    else:
                        Y.append(y[-1])
                X = torch.stack(X)
                Y = torch.stack(Y)
                accs = []
                with torch.no_grad():
                    for epoch in epochs:
                        for save in [0]:
                            loader.load_model_from_epoch_and_dir(model, run_dir, epoch, save)
                            out = model(X).detach()
                            if not FEEDFORWARD:
                                out = out[:, -1]
                            out_cat = torch.argmax(out, dim=1)
                            acc = torch.mean((out_cat == Y).type(torch.float)).item()
                            accs.append(acc)
                    if not hue_dictionary_None:
                        d = {'seed': seed, 'num_training_samples': np.array(epochs)*spe, 'accuracy': accs,
                             hue_legend_key: hue_dictionary[hue_legend_key][hue_idx]}
                    else:
                        d = {'seed': seed, 'num_training_samples': np.array(epochs)*spe, 'accuracy': accs}

                    df = pd.DataFrame(d)
                    loss_and_acc_table = loss_and_acc_table.append(df)

        loss_and_acc_table['seed'] = loss_and_acc_table['seed'].astype('category')
        if not hue_dictionary_None:
            loss_and_acc_table[hue_legend_key] = loss_and_acc_table[hue_legend_key].astype('category')
        return loss_and_acc_table

    loss_and_acc_table = generate_data_table(train_params, seeds, hue_dictionary, hue_legend_key, epochs)

    fig, ax = utils.make_fig(figsize)
    if USE_ERRORBARS:
        g = sns.lineplot(ax=ax, x='num_training_samples', y='accuracy', data=loss_and_acc_table, hue=hue_legend_key,
                         estimator=est_acc, ci=ci_acc, **plot_kwargs)
    else:
        g = sns.lineplot(ax=ax, x='num_training_samples', y='accuracy', data=loss_and_acc_table, hue=hue_legend_key,
                         estimator=None, units='seed', **plot_kwargs)
    if g.legend_ is not None:
        g.legend_.remove()
    out_fig(fig, figname, train_params, subfolder=train_params['network'] + '/acc_and_loss_over_training/',
            data=loss_and_acc_table)

def _cluster_holdout_test_acc_stat_fun(h, y, clust_identity, classifier_type='logistic_regression', num_repeats=5,
                                       train_ratio=0.8, seed=11):
    np.random.seed(seed)
    num_clusts = np.max(clust_identity) + 1
    num_clusts_train = int(round(num_clusts*train_ratio))
    num_samples = h.shape[0]
    test_accs = np.zeros(num_repeats)
    train_accs = np.zeros(num_repeats)

    for i0 in range(num_repeats):
        permutation = np.random.permutation(np.arange(len(clust_identity)))
        perm_inv = np.argsort(permutation)
        clust_identity_shuffled = clust_identity[permutation]
        train_idx = clust_identity_shuffled <= num_clusts_train
        test_idx = clust_identity_shuffled > num_clusts_train
        hid_train = h[train_idx[perm_inv]]
        y_train = y[train_idx[perm_inv]]
        y_test = y[test_idx[perm_inv]]
        hid_test = h[test_idx[perm_inv]]
        if classifier_type == 'svm':
            classifier = svm.LinearSVC(random_state=3*i0 + 1)
        else:
            classifier = linear_model.LogisticRegression(random_state=3*i0 + 1, solver='lbfgs')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(hid_train, y_train)
        train_accs[i0] = classifier.score(hid_train, y_train)
        test_accs[i0] = classifier.score(hid_test, y_test)

    return train_accs, test_accs

def clust_holdout_over_layers(seeds, gs, train_params, figname="clust_holdout_over_layers",
                              **plot_kwargs):
    """
    Logistic regression training and testing error on the representation through the layers. Compares networks trained
    with different choices of g_radius (specified by input parameter gs).

    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of the model and dataset. Variation over
        these seeds is used to plot error bars.
    gs : List[float]
        Values of g_radius to iterate over.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset to use for training. Value of g_radius
        is overwritten by values in gs.
    figname : str
        Name of the figure to save.
    """
    if not hasattr(gs, '__len__'):
        gs = [gs]
    layer_label = 'layer'

    def generate_data_table(seeds, gs, train_params):
        layer_label = 'layer'
        clust_acc_table = pd.DataFrame(columns=['seed', 'g_radius', 'training', layer_label, 'LR training', 'LR testing'])

        train_params_loc = copy.deepcopy(train_params)

        for i0, seed in enumerate(seeds):
            for i1, g in enumerate(gs):

                train_params_loc['g_radius'] = g
                train_params_loc['model_seed'] = seed

                num_pnts_dim_red = 500

                model, params, run_dir = initialize_and_train.initialize_and_train(**train_params_loc)

                class_datasets = params['datasets']
                num_train_samples = len(class_datasets['train'])
                class_datasets['train'].max_samples = num_pnts_dim_red
                torch.manual_seed(params['model_seed'])
                X, Y = class_datasets['train'][:]
                if train_params_loc['network'] == 'feedforward':
                    X0 = X
                else:
                    X0 = X[:, 0]

                for epoch, epoch_label in zip([0, -1], ['before', 'after']):
                    loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
                    hid = [X0]
                    hid += model.get_post_activations(X)[:-1]
                    if len(Y.shape) > 1:
                        Y = Y[:, -1]
                    cluster_identity = class_datasets['train'].cluster_identity
                    ds = []
                    for lay, h in enumerate(hid):
                        stat = _cluster_holdout_test_acc_stat_fun(h.numpy(), Y.numpy(), cluster_identity)
                        ds.extend([{'seed': seed, 'g_radius': g, 'training': epoch_label, layer_label: lay, 'LR training':
                            stat[0][k], 'LR testing': stat[1][k]} for k in range(len(stat[0]))])

                    clust_acc_table = clust_acc_table.append(pd.DataFrame(ds), ignore_index=True)

        clust_acc_table['seed'] = clust_acc_table['seed'].astype('category')
        clust_acc_table['g_radius'] = clust_acc_table['g_radius'].astype('category')
        clust_acc_table['training'] = clust_acc_table['training'].astype('category')
        return clust_acc_table

    clust_acc_table = generate_data_table(seeds, gs, train_params)
    layers = set(clust_acc_table[layer_label])

    for stage in ['LR training', 'LR testing']:
        if stage == 'LR training':
            clust_acc_table_stage = clust_acc_table.drop(columns=['LR testing'])
        else:
            clust_acc_table_stage = clust_acc_table.drop(columns=['LR training'])
        fig, ax = make_fig((1.5, 1.2))
        if USE_ERRORBARS:
            table = clust_acc_table_stage.copy()
            table = table[table['training'] == 'after']
            g = sns.lineplot(ax=ax, x=layer_label, y=stage, data=table, estimator=est_acc,
                             ci=ci_acc, hue='g_radius', **plot_kwargs)
            # table = table.drop(columns=['training'])
            # table = table.drop(columns=['g_radius'])
            # table = table.rename(columns={'layer': 'x_values', 'LR testing': 'y_values'})
            # table.to_csv('seaborn_demo.csv')
            # g = sns.lineplot(ax=ax, x=layer_label, y=stage, data=clust_acc_table_stage, estimator=est_acc,
            #                  ci=ci_acc, style='training', style_order=['after', 'before'], hue='g_radius')
        else:
            table = clust_acc_table_stage.copy()
            table = table[table['training'] == 'after']
            g = sns.lineplot(ax=ax, x=layer_label, y=stage, data=table, estimator=None,
                             units='seed', hue='g_radius', **plot_kwargs)
        if g.legend_ is not None:
            g.legend_.remove()
        ax.set_ylim([-.01, 1.01])
        ax.set_xticks(range(len(layers)))
        out_fig(fig, figname + '_' + stage, train_params, subfolder=train_params['network'] +
                                                                    '/clust_holdout_over_layers/',
                show=False, save=True, axis_type=0, name_order=0, data=clust_acc_table)

    plt.close('all')

def dim_over_layers(seeds, gs, train_params, figname="dim_over_layers", T=0):
    """
    Effective dimension measured over layers (or timepoints if looking at an RNN) of the network, before and after
    training.

    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of the model and dataset. Variation over
        these seeds is used to plot error bars.
    gs : List[float]
        Values of g_radius to iterate over.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset to use for training. Value of g_radius
        is overwritten by values in gs.
    figname : str
        Name of the figure to save.
    T : int
        Final timepoint to plot (if looking at an RNN). If 0, disregard this parameter.
    """
    if not hasattr(gs, '__len__'):
        gs = [gs]
    stat_key = 'dim'
    layer_label = 'layer'

    train_params_loc = copy.deepcopy(train_params)
    dim_table = pd.DataFrame(columns=['seed', 'g_radius', 'training', layer_label, stat_key])

    for i0, seed in enumerate(seeds):
        for i1, g in enumerate(gs):

            train_params_loc['g_radius'] = g
            train_params_loc['model_seed'] = seed

            num_pnts_dim_red = 500

            model, params, run_dir = initialize_and_train.initialize_and_train(**train_params_loc)

            class_datasets = params['datasets']
            class_datasets['train'].max_samples = num_pnts_dim_red
            torch.manual_seed(params['model_seed'])
            X, Y = class_datasets['train'][:]
            if T > 0:
                X = utils.extend_input(X, T)
                X0 = X[:, 0]
            elif train_params_loc['network'] != 'feedforward':
                X0 = X[:, 0]
            else:
                X0 = X

            for epoch, epoch_label in zip([0, -1], ['before', 'after']):
                loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
                hid = [X0]
                hid += model.get_post_activations(X)[:-1]
                if len(Y.shape) > 1:
                    Y = Y[:, -1]
                stats = [utils.get_effdim(h, False).item() for h in hid]
                ds = {'seed': seed, 'g_radius': g, 'training': epoch_label, layer_label: list(range(len(hid))),
                      stat_key: stats}
                dim_table = dim_table.append(pd.DataFrame(ds), ignore_index=True)

    dim_table['seed'] = dim_table['seed'].astype('category')
    dim_table['g_radius'] = dim_table['g_radius'].astype('category')
    dim_table['training'] = dim_table['training'].astype('category')
    layers = set(dim_table[layer_label])

    fig, ax = make_fig((1.5, 1.2))
    if USE_ERRORBARS:
        g = sns.lineplot(ax=ax, x=layer_label, y=stat_key, data=dim_table, estimator=est_dim,
                         ci=ci_dim, style='training', style_order=['after', 'before'], hue='g_radius')
    else:
        g = sns.lineplot(ax=ax, x=layer_label, y=stat_key, data=dim_table, estimator=None,
                         style='training', style_order=['after', 'before'], hue='g_radius',
                         units='seed')
    if g.legend_ is not None:
        g.legend_.remove()
    ax.set_xticks(range(len(layers)))
    out_fig(fig, figname, train_params_loc, subfolder=train_params_loc['network'] + '/dim_over_layer/',
            show=False, save=True, axis_type=0, data=dim_table)

    plt.close('all')

def lyaps(seeds, train_params, epochs, figname="lyaps"):
    """
    Lyapunov exponent plots.

    Parameters
    ----------
    seeds : List[int]
        List of random number seeds to use for generating instantiations of the model and dataset. Variation over
        these seeds is used to plot error bars.
    train_params : dict
        Dictionary of training parameters that specify the model and dataset to use for training. Value of g_radius
        is overwritten by values in gs.
    epochs : List[int]
        Epochs at which to plot the accuracy.
    figname : str
        Name of the figure to save.
    """
    ICs = 'random'

    train_params_loc = copy.deepcopy(train_params)
    lyap_table = pd.DataFrame(columns=['seed', 'epoch', 'lyap', 'lyap_num', 'sem', 'chaoticity'])
    k_LE = 10

    for i0, seed in enumerate(seeds):
        train_params_loc['model_seed'] = seed

        model, params, run_dir = initialize_and_train.initialize_and_train(**train_params_loc)

        torch.manual_seed(train_params_loc['model_seed'])

        for epoch in epochs:
            loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
            Wrec = model.Wrec.detach().numpy()
            Win = model.Win.detach().numpy()
            brec = model.brec.detach().numpy()

            if isinstance(ICs, str):
                if ICs == 'random':
                    ICs_data = None
                else:
                    ICs_data = None
            else:
                ICs_data = ICs

            LEs, sem, trajs = lyap.getSpectrum(Wrec, brec, Win, x=0, k_LE=k_LE, max_iters=1000,
                                               max_ICs=10, ICs=ICs_data, tol=2e-3, verbose=True)
            LEs = np.sort(LEs)[::-1]
            chaoticity = np.sum(LEs[:3]/np.arange(1, len(LEs[:3]) + 1))
            d = [{'seed': seed, 'epoch': epoch, 'lyap': LEs[k], 'lyap_num': k, 'sem': sem,
                  'chaoticity': chaoticity} for k in range(len(LEs))]
            lyap_table = lyap_table.append(d, ignore_index=True)

    lyap_table['seed'] = lyap_table['seed'].astype('category')
    lyap_table['epoch'] = lyap_table['epoch'].astype('category')

    fig, ax = make_fig((2, 1.2))
    lyap_table_plot = lyap_table.drop(columns=['sem', 'chaoticity'])
    g = sns.pointplot(ax=ax, x='lyap_num', y='lyap', data=lyap_table_plot, style='training', hue='epoch', ci=95,
                      scale=0.5)
    ax.set_xticks(sorted(list(set(lyap_table['lyap_num']))))
    ax.axhline(y=0, color='black', linestyle='--')
    out_fig(fig, figname, train_params_loc, subfolder=train_params_loc['network'] + '/lyaps/',
            show=False, save=True, axis_type=0, data=lyap_table)
    #
