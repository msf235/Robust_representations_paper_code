import copy
import pickle as pkl
from pathlib import Path

from scipy import stats as st
import torch
from sklearn import svm
from sklearn import linear_model
import pandas as pd
import matplotlib.ticker
from matplotlib import animation, ticker
import seaborn as sns
import subprocess
import warnings
# warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import juggle_axes

import lyap
import model_loader_utils as loader
import initialize_and_train
import utils
# from joblib import Parallel, delayed, Memory


# memory = Memory(location='../joblib_cache', verbose=2)
# memory.clear()

# def ci_acc(vals):
#     median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1., shift=-.0001,
#                                       reflect=True)
#     return bounds[1], bounds[0]

ci_acc = 95
# def est_acc(vals):
#     median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1., shift=-.0001,
#                                       reflect=True)
#     return median
est_acc = "mean"

# def ci_dim(vals):
#     median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1)
#     return bounds[1], bounds[0]
ci_dim = 95

# def est_dim(vals):
#     median, bounds = median_and_bound(vals, perc_bound=0.75, loc=1)
#     return median
est_dim = "mean"

def point_replace(a_string):
    a_string = str(a_string)
    return a_string.replace(".", "p")

def get_color(hidden, cmap=plt.cm.plasma):
    mag = torch.max(hidden) - torch.min(hidden)
    # hid_norm = (hidden - np.min(hidden)) / (mag - 1)
    hid_norm = (hidden.float() - torch.min(hidden)) / mag

    return cmap(hid_norm)


def median_and_bound(samples, perc_bound, dist_type='gamma', loc=0., shift=0, reflect=False, show_fit=False):
    samples = np.array(samples)

    def do_reflect(x, center):
        return -1 * (x - center) + center

    if dist_type == 'gamma':
        # gam = st.gamma
        # a1 = 2
        # scale = 200
        # xx = np.linspace(0,1500)
        # yy = gam.pdf(xx, a1, loc=0, scale=scale)
        # median_true = st.gamma.median(a1, loc=0, scale=scale)
        #
        # samples = gam.rvs(a1, loc=0, scale=scale, size=40)
        if np.sum(samples[0] == samples) == len(samples):
            median = samples[0]
            interval = [samples[0], samples[0]]
            return median, interval

        if reflect:
            # reflect_point = loc + shift
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

        if np.isnan(np.sum(median)) or np.isnan(np.sum(interval)):
            print

        if show_fit:
            fig, ax = plt.subplots()
            ax.hist(samples, density=True)
            xx = np.linspace(np.min(samples), np.max(samples))
            if reflect:
                yy_hat = st.gamma.pdf(do_reflect(xx, loc), shape_ps, loc=loc, scale=scale)
                ax.plot(xx, yy_hat)
                ax.axvline(x=median)
                ax.axvline(x=interval[0], linestyle='--')
                ax.axvline(x=interval[1], linestyle='--')
                plt.show()
                print
            else:
                yy_hat = st.gamma.pdf(xx, shape_ps, loc=loc, scale=scale)
                ax.plot(xx, yy_hat)
                ax.axvline(x=median)
                ax.axvline(x=interval[0], linestyle='--')
                ax.axvline(x=interval[1], linestyle='--')
                plt.show()
            print

            # plt.plot(xx, yy)
            # plt.plot(xx, yy_hat, '--')
            # plt.axvline(x=median)
            # plt.axvline(x=median_true, color='red')
            # plt.axvline(x=interval[0], linestyle='--')
            # plt.axvline(x=interval[1], linestyle='--')
            # plt.show()
            # plt.close()

    return median, interval

plt.rcParams['font.size'] = 6
plt.rcParams['font.size'] = 6
plt.rcParams['lines.markersize'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
# plt.rcParams['text.usetex'] = True
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rcParams['axes.titlesize'] = 8

class_style = 'color'

cols11 = np.array([90, 100, 170]) / 255
cols12 = np.array([37, 50, 120]) / 255
cols21 = np.array([250, 171, 62]) / 255
cols22 = np.array([156, 110, 35]) / 255
cmap_activation_pnts = mcolors.ListedColormap([cols11, cols21])
cmap_activation_pnts_edge = mcolors.ListedColormap([cols12, cols22])

rasterized = False
dpi = 800

ext = 'pdf'
# ext = 'png'
create_svg = False

# figsize = (2, 1.6)
figsize = (1.5, 1.2)
figsize_small = (1, 0.8)
# figsize_long = (1.2, 1.2/1.5)
# figsize_smaller = (.55, 0.5)
figsize_smaller = (.8, 0.6)
figsize_smallest = (.3, 0.2)
# figsize_long = (.8, 0.4)
figsize_long = (1, 0.5)
# figsize_smallest = (.4, 0.25)
ax_pos = (0, 0, 1, 1)


def make_fig(figsize=figsize, ax_pos=ax_pos):
    """

    Args:
        figsize ():
        ax_pos ():

    Returns:

    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_pos)
    return fig, ax


def out_fig(fig, name, train_params, subfolder='', show=False, save=True, axis_type=0, name_order=0, data=None):
    """

    Args:
        fig ():
        name ():
        show ():
        save ():
        axis_type (int): 0: leave axes alone, 1: Borders but no ticks, 2: Axes off

    Returns:

    """
    # fig.tight_layout()
    # fig.axes[0].ticklabel_format(style='sci',scilimits=(-2,2),axis='both')
    # fig.tight_layout()
    folder = 'figs/Win_{}/'.format(train_params['Win'])
    os.makedirs('figs/', exist_ok=True)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder + 'snaps/', exist_ok=True)
    os.makedirs(folder + 'snaps/no_border', exist_ok=True)
    os.makedirs(folder + 'snaps/border', exist_ok=True)
    os.makedirs(folder + 'snaps_3d/no_border', exist_ok=True)
    os.makedirs(folder + 'snaps_3d/border', exist_ok=True)
    g = train_params['g_radius']
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
        fig_name = point_replace(
            folder + subfolder + '{}_g_{}_Xdim_{}_{}_{}'.format(name, g, X_dim, nonlinearity, loss))
    else:
        fig_name = point_replace(folder + subfolder + 'g_{}_Xdim_{}_{}'.format(g, X_dim, name, nonlinearity, loss))
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
        with open(folder + subfolder + 'data/g_{}_Xdim_{}_{}_data'.format(g, X_dim, name), 'wb') as fid:
            pkl.dump(data, fid, protocol=4)

def activity_visualization(train_params):
    # a = classification_dep.ClassificationAnalysis(architecture='noisy_recurrent')
    # init_params = dict(architecture='recurrent')
    g_str = '_' + str(train_params['g_radius'])
    X_dim = train_params['X_dim']
    FEEDFORWARD = train_params['network'] == 'feedforward'
    SUBFOLDER = train_params['network'] + '/' + 'activity_visualization/' + train_params['hid_nonlin'] + '/'

    num_pnts_dim_red = 800
    # num_pnts_dim_red = 40000
    num_plot = 600

    accv = []
    val_accv = []
    lossvs = []
    val_lossvs = []

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
        # loader.load_model_from_epoch_and_dir(model, run_dir, 0, 0)
        # hid_0 = model.get_post_activations(X)[:-1]
        # loader.load_model_from_epoch_and_dir(model, run_dir, -1, 0)
        # hid = model.get_post_activations(X)[:-1]
    else:
        T = 30
        X = utils.extend_input(X, T + 2)
        X0 = X[:, 0]
        y = Y[:, -1]
        # loader.load_model_from_epoch_and_dir(model, run_dir, 0, 0)
        # hid_0 = model.get_post_activations(X).transpose(0, 1)
        # # out = model(X)[:, -1]
        # hid_0 = [h for h in hid_0]
        # # hid_0.append(out)
        # loader.load_model_from_epoch_and_dir(model, run_dir, -1, 0)
        # hid = model.get_post_activations(X).transpose(0, 1)
        # # out = model(X)[:, -1]
        # hid = [h for h in hid]
        # # hid.append(out)
    loader.load_model_from_epoch_and_dir(model, run_dir, 0, 0)
    hid_0 = [X0]
    hid_0 += model.get_post_activations(X)[:-1]
    loader.load_model_from_epoch_and_dir(model, run_dir, -1, 0)
    hid = [X0]
    hid += model.get_post_activations(X)[:-1]

    # loader.load_model_from_epoch_and_dir(model, run_dir, 0, 0)
    # hid_0 = model.get_post_activations(X)
    # loader.load_model_from_epoch_and_dir(model, run_dir, -1, 0)
    # hid = model.get_post_activations(X)

    # out = hid_bef_aft = na.model_loader_utils.get_activity(model, run_dir, X, [0, -1], [-1], return_as_Tensor=True)
    # out_load = na.model_loader_utils.activity_loader(model, run_dir, X, 1)
    # out = out_load[-1].transpose(0,1)
    # o = out[:, 10, 0]

    # %% Visualize
    coloring = get_color(y, cmap_activation_pnts)[:num_plot]
    edge_coloring = get_color(y, cmap_activation_pnts_edge)[:num_plot]
    traj_size = (.6, .35)

    # Input
    if X_dim > 2:
        X_pcs = utils.get_pcs_covariance(X0[:num_pnts_dim_red], [0, 1, 2])
    else:
        X_pcs = utils.get_pcs_covariance(X0[:num_pnts_dim_red], [0, 1])

    fig, ax = make_fig()
    if class_style == 'shape':
        X_pcs_plot = X_pcs[:num_plot]
        X_0 = X_pcs_plot[y[:num_plot] == 0]
        X_1 = X_pcs_plot[y[:num_plot] == 1]
        ax.scatter(X_0[:, 0], X_0[:, 1], c=coloring, edgecolors=edge_coloring, s=10, linewidths=.4, marker='o')
        ax.scatter(X_1[:, 0], X_1[:, 1], c=coloring, edgecolors=edge_coloring, s=10, linewidths=.4, marker='^')
    else:
        ax.scatter(X_pcs[:num_plot, 0], X_pcs[:num_plot, 1], c=coloring,
                   edgecolors=edge_coloring, s=10, linewidths=.4)

    # out_params = train_params_loc
    out_fig(fig, "input_pcs", train_params_loc, SUBFOLDER, axis_type=1)

    # fig, ax = make_fig()
    if X_dim > 2:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        if class_style == 'shape':
            X_pcs_plot = X_pcs[:num_plot]
            X_0 = X_pcs_plot[y[:num_plot] == 0]
            X_1 = X_pcs_plot[y[:num_plot] == 1]
            ax.scatter(X_0[:, 0], X_0[:, 1], X_0[:, 2], c=inp_col, edgecolors=inp_col_edge, s=10, linewidths=.4,
                       marker='o')
            ax.scatter(X_1[:, 0], X_1[:, 1], X_1[:, 2], c=inp_col, edgecolors=inp_col_edge, s=10, linewidths=.4,
                       marker='^')
        else:
            ax.scatter(X_pcs[:num_plot, 0], X_pcs[:num_plot, 1], X_pcs[:num_plot, 2], c=coloring,
                       edgecolors=edge_coloring, s=10, linewidths=.4)
        ax.grid(False)
        out_fig(fig, "input_pcs_3d", train_params_loc, SUBFOLDER, axis_type=0)

    fig, ax = make_fig()
    ax.scatter(X0[:num_plot, 0], X0[:num_plot, 1], c=coloring,
               edgecolors=edge_coloring, s=10, linewidths=.4)
    out_fig(fig, "input_first2", train_params_loc, SUBFOLDER, axis_type=1)
    #
    #
    fig, ax = make_fig(figsize=figsize_long)
    h0_temporal = [h[0, :10].numpy() for h in hid_0[1:]]
    ax.plot(h0_temporal, linewidth=0.7)
    if train_params['hid_nonlin'] == 'tanh':
        ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-0.02, T + .02])
    ax.set_xlabel(r"$t$", labelpad=-5)
    ax.set_ylabel(r"$h$", labelpad=-2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(30))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_locator(plt.FixedLocator([-1, 1]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([-1, 0, 1]))
    out_fig(fig, "hid_before_train", train_params_loc, SUBFOLDER)
    #
    # plt.style.use('ggplot')
    # fig, ax = make_fig(figsize=traj_size)
    # fig, ax = make_fig(figsize=figsize_small)
    fig, ax = make_fig(figsize=figsize_long)
    # plt.rc('axes', linewidth=2.3)
    h_temporal = [h[0, :10].numpy() for h in hid[1:]]
    # ax.plot(hid_0[0, :, :10], linewidth=0.7)
    ax.plot(h_temporal, linewidth=0.7)
    # ax.plot(hid[0, :, :10], linewidth=0.7)
    if train_params['hid_nonlin'] == 'tanh':
        ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-0.02, 30.02])
    ax.set_xlabel(r"$t$", labelpad=-5)
    ax.set_ylabel(r"$h$", labelpad=-2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(30))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_locator(plt.FixedLocator([-1, 1]))
    ax.yaxis.set_minor_locator(plt.FixedLocator([-1, 0, 1]))
    # ax.tick_params(axis='both', which='major', width=1, length=3)
    # ax.tick_params(axis='both', which='minor', width=.7, length=2)
    out_fig(fig, "hid_after_train", train_params_loc, SUBFOLDER)

    # fig, ax = make_fig(figsize=traj_size)
    # ax.plot(hid_0[0, :, :10])
    # # ax.plot(hid_0[0, :, :])
    # ax.set_ylim([-1.1, 1.1])
    # ax.set_xlim([-0.02, 30.02])
    # ax.set_xlabel(r"$t$", labelpad=-5)
    # ax.set_ylabel(r"$h$", labelpad=-2)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(30))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    # out_fig(fig, "hid_before_train", g, X_dim)
    #
    # # plt.style.use('ggplot')
    # fig, ax = make_fig(figsize=traj_size)
    # # plt.rc('axes', linewidth=2.3)
    # ax.plot(hid[0, :, :10])
    # # ax.plot(hid[0, :, :])
    # ax.set_ylim([-1.1, 1.1])
    # ax.set_xlim([-0.02, 30.02])
    # ax.set_xlabel(r"$t$", labelpad=-5)
    # ax.set_ylabel(r"$h$", labelpad=-2)
    # # ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(30))
    # # ax.xaxis.set_minor_locator(plt.MultipleLocator(2.5))
    # # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    # # ax.tick_params(axis='both', which='major', width=1, length=3)
    # # ax.tick_params(axis='both', which='minor', width=.7, length=2)
    # out_fig(fig, "hid_after_train", g, X_dim)

    fig, ax = make_fig(figsize=traj_size)
    # ax.hist(hid_0[:, 0, :].flatten())
    ax.hist(hid_0[0].flatten())
    out_fig(fig, "hid0_t0_hist", train_params_loc, SUBFOLDER)

    fig, ax = make_fig(figsize=traj_size)
    ax.hist(hid_0[-1].flatten())
    out_fig(fig, "hid0_T_hist", train_params_loc, SUBFOLDER)

    # pcs = dim_tools.get_pcs_stefan(hid, [0, 1])
    # pcs = np.zeros(hid.shape[:-1] + (3,))
    pcs = []
    p_track = 0
    # y0 = y == 0
    # y1 = y == 1
    norm = np.linalg.norm
    for i1 in range(len(hid)):
        # pc = dim_tools.get_pcs_stefan(hid[:, i1], [0, 1, 2])
        pc = utils.get_pcs_covariance(hid[i1], [0, 1])
        if i1 > 0:
            # pc_old = pc.copy()
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
        # pcs[:, i1] = pc
        pcs.append(pc[:num_plot])
    # pcs = pcs[:num_plot]
    plt.close('all')

    # if gs[0] == 20:
    #     pcs[:,:,0] = -pcs[:,:,0]
    # p_track = pcs[0]

    # fig, ax = make_fig()  # This causes an error with the movie
    # fig, ax = plt.subplots()

    # y = y[:num_plot]

    def take_snap(i0, scat, dim=2, border=False):
        # hid_pcs_plot = pcs[y==1, i0, :dim]
        hid_pcs_plot = pcs[i0][:, :dim].numpy()
        # hid_pcs_plot = hid_pcs_plot[:, :2] - np.mean(hid_pcs_plot[:, :2], axis=0)
        xm = np.min(hid_pcs_plot[:, 0])
        xM = np.max(hid_pcs_plot[:, 0])
        ym = np.min(hid_pcs_plot[:, 1])
        yM = np.max(hid_pcs_plot[:, 1])
        xc = (xm + xM) / 2
        yc = (ym + yM) / 2
        hid_pcs_plot[:, 0] = hid_pcs_plot[:, 0] - xc
        hid_pcs_plot[:, 1] = hid_pcs_plot[:, 1] - yc
        if class_style == 'shape':
            # scat[0].set_offsets(hid_pcs_plot[y==0])
            # scat[1].set_offsets(hid_pcs_plot[y==1])
            scat[0].set_offsets(hid_pcs_plot)
        else:
            if dim == 3:
                scat._offsets3d = juggle_axes(*hid_pcs_plot[:, :dim].T, 'z')
            else:
                scat.set_offsets(hid_pcs_plot)
        # if dim == 3:
        #     scat[0]._offsets3d = juggle_axes(*hid_pcs_plot[:, :dim].T, 'z')
        # else:
        #     scat.set_offsets(hid_pcs_plot)

        xm = np.min(hid_pcs_plot[:, 0])
        xM = np.max(hid_pcs_plot[:, 0])
        ym = np.min(hid_pcs_plot[:, 1])
        yM = np.max(hid_pcs_plot[:, 1])
        max_extent = max(xM - xm, yM - ym)
        max_extent_arg = xM - xm > yM - ym
        if dim == 2:
            x_factor = .4
            if max_extent_arg:
                ax.set_xlim([xm - x_factor * max_extent, xM + x_factor * max_extent])
                ax.set_ylim([xm - .1 * max_extent, xM + .1 * max_extent])
            else:
                ax.set_xlim([ym - x_factor * max_extent, yM + x_factor * max_extent])
                ax.set_ylim([ym - .1 * max_extent, yM + .1 * max_extent])
        else:
            if max_extent_arg:
                ax.set_xlim([xm - .1 * max_extent, xM + .1 * max_extent])
                ax.set_ylim([xm - .1 * max_extent, xM + .1 * max_extent])
                ax.set_zlim([xm - .1 * max_extent, xM + .1 * max_extent])
            else:
                ax.set_xlim([ym - .1 * max_extent, yM + .1 * max_extent])
                ax.set_ylim([ym - .1 * max_extent, yM + .1 * max_extent])
                ax.set_zlim([ym - .1 * max_extent, yM + .1 * max_extent])
            # ax.set_xlim([-10, 10])
            # ax.set_ylim([-10, 10])
            # ax.set_zlim([-10, 10])

        if dim == 3:
            if border:
                out_fig(fig, "snapshot_{}".format(i0), train_params_loc,
                        subfolder=SUBFOLDER + "snaps_3d/border/", axis_type=0,
                        name_order=1)
            else:
                out_fig(fig, "snapshot_{}".format(i0), train_params_loc,
                        subfolder=SUBFOLDER + "snaps_3d/no_border/", axis_type=2,
                        name_order=1)
            # out_fig(fig, "snapshot_{}".format(i0), g, X_dim, folder=folder + "snaps/no_border/", axis_type=0,
            #         name_order=1)
        else:
            if border:
                out_fig(fig, "snapshot_{}".format(i0), train_params_loc,
                        subfolder=SUBFOLDER + "snaps/border/", axis_type=0,
                        name_order=1)
            else:
                out_fig(fig, "snapshot_{}".format(i0), train_params_loc,
                        subfolder=SUBFOLDER + "snaps/no_border/", axis_type=2,
                        name_order=1)
            # out_fig(fig, "snapshot_{}".format(i0), g, X_dim, folder=folder + "snaps/no_border/", axis_type=0,
            #         name_order=1)
        return scat,

    dim = 2
    hid_pcs_plot = pcs[0]
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        # ax.xaxis._axinfo['juggled'] = (0, 1, 0)
    else:
        fig, ax = make_fig()
    ax.grid(False)

    scat = ax.scatter(*hid_pcs_plot[:num_plot, :dim].T, c=coloring,
                      edgecolors=edge_coloring, s=10, linewidths=.65)

    if FEEDFORWARD:
        snap_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    else:
        snap_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30])
    # snap_idx = np.array([0, 1, 2, 3])
    for i0 in snap_idx:
        take_snap(i0, scat, dim=dim, border=False)
    # plt.close('all')
    # anim = animation.FuncAnimation(fig, take_snap, interval=200, frames=n_points - 1, blit=True)
    # anim = animation.FuncAnimation(fig, take_snap, frames=8, interval=200)
    # anim.save('/Users/matt/something.mp4')
    # plt.show()
    # anim.save("figs/pdf/snaps/video_{}.mp4".format(g))
    # fname = point_replace("/Users/matt/video_{}".format(g))
    # fname = point_replace("video_{}".format(g))
    # anim.save(fname + ".mp4")
    # anim.save("something.gif")
    # plt.show()


def snapshots_through_time(train_params):
    # a = classification_dep.ClassificationAnalysis(architecture='noisy_recurrent')
    # init_params = dict(architecture='recurrent')
    X_dim = train_params['X_dim']
    FEEDFORWARD = train_params['network'] == 'feedforward'
    SUBFOLDER = train_params['network'] + '/' + 'activity_visualization/' + train_params['hid_nonlin'] + '/'

    num_pnts_dim_red = 800
    # num_pnts_dim_red = 40000
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
        # pc = dim_tools.get_pcs_stefan(hid[:, i1], [0, 1, 2])
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
        xc = (xm + xM) / 2
        yc = (ym + yM) / 2
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
                ax.set_xlim([xm - x_factor * max_extent, xM + x_factor * max_extent])
                ax.set_ylim([xm - .1 * max_extent, xM + .1 * max_extent])
            else:
                ax.set_xlim([ym - x_factor * max_extent, yM + x_factor * max_extent])
                ax.set_ylim([ym - .1 * max_extent, yM + .1 * max_extent])
        else:
            if max_extent_arg:
                ax.set_xlim([xm - .1 * max_extent, xM + .1 * max_extent])
                ax.set_ylim([xm - .1 * max_extent, xM + .1 * max_extent])
                ax.set_zlim([xm - .1 * max_extent, xM + .1 * max_extent])
            else:
                ax.set_xlim([ym - .1 * max_extent, yM + .1 * max_extent])
                ax.set_ylim([ym - .1 * max_extent, yM + .1 * max_extent])
                ax.set_zlim([ym - .1 * max_extent, yM + .1 * max_extent])

        if dim == 3:
            if border:
                out_fig(fig, "snapshot_{}".format(i0), train_params_loc,
                        subfolder=SUBFOLDER + "snaps_3d/border/", axis_type=0,
                        name_order=1)
            else:
                out_fig(fig, "snapshot_{}".format(i0), train_params_loc,
                        subfolder=SUBFOLDER + "snaps_3d/no_border/", axis_type=2,
                        name_order=1)
        else:
            if border:
                out_fig(fig, "snapshot_{}".format(i0), train_params_loc,
                        subfolder=SUBFOLDER + "snaps/border/", axis_type=0,
                        name_order=1)
            else:
                out_fig(fig, "snapshot_{}".format(i0), train_params_loc,
                        subfolder=SUBFOLDER + "snaps/no_border/", axis_type=2,
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
        snap_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    else:
        snap_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30])
    for i0 in snap_idx:
        take_snap(i0, scat, dim=dim, border=False)

def acc_and_loss_over_training(train_params, seeds, hue_dictionary=None, hue_target=(None, None), epoch_list=None,
                               epoch_plot=None, figname=None):
    """
    Plot accuracy over training. hue_dictionary is an optional dictionary with a single entry that allows you to
    specify a parameter for which to plot multiple lines with different hues on the same plot.
    """
    if figname is None:
        figname = 'acc_and_loss_over_training'
    if epoch_list is None:
        epoch_list_None = True
    else:
        epoch_list_None = False
    if epoch_plot is None:
        epoch_plot_None = True
    else:
        epoch_plot_None = False


    # @memory.cache()
    def memoized_core(train_params, seeds, hue_dictionary, hue_target, epoch_list, epoch_plot):
        train_params_loc = copy.copy(train_params)
        spe = train_params_loc['num_train_samples_per_epoch']
        pretrain = 'pretrain_params' in train_params_loc.keys() and train_params_loc['pretrain_params'] is not None
        NUM_SAMPLES = 100
        # device = 'cuda'
        device = 'cpu'
        spe = train_params_loc['num_train_samples_per_epoch']
        FEEDFORWARD = train_params_loc['network'] == 'feedforward'
        if pretrain:
            num_epochs_pretrain = train_params_loc['pretrain_params']['num_epochs']
            start_epoch_train = num_epochs_pretrain
        else:
            num_epochs_pretrain = 0
            start_epoch_train = 0
        num_epochs_total = num_epochs_pretrain + train_params_loc['num_epochs']

        if hue_target[0] is not None:
            hue_target_str = hue_target[0]
            hue_target_idx = 0
            param_target = train_params_loc
            # hue_dict_split = [hue_dictionary]
        elif hue_target[1] is not None:
            hue_target_str = hue_target[1] + '_pretrain'
            hue_target_idx = 1
            # hue_dict_split = [hue_target, hue_target['pretrain']]
        elif hue_dictionary is None:
            hue_target_str = None
        else:
            raise AttributeError('')

        if hue_dictionary is None:
            hue_dictionary_None = True
            hue_keys = []
            hue_pt_keys = []  # pretrain keys
            loss_and_acc_table = pd.DataFrame(columns=['seed', 'num_training_samples', 'accuracy'])
            num_hues = 1
        else:
            hue_dictionary_None = False
            if 'pretrain_params' in hue_dictionary:
                hue_keys = list(hue_dictionary.keys())
                hue_keys.remove('pretrain_params')
                hue_pt_keys = list(hue_dictionary['pretrain_params'].keys())
            else:
                hue_keys = list(hue_dictionary.keys())
                hue_pt_keys = list()
            loss_and_acc_table = pd.DataFrame(columns=['seed', 'num_training_samples', 'accuracy', hue_target_str])
            if hue_target_idx == 0:
                num_hues = len(hue_dictionary[hue_target[hue_target_idx]])
            elif hue_target_idx == 1:
                num_hues = len(hue_dictionary['pretrain_params'][hue_target[hue_target_idx]])
            else:
                raise AttributeError('')

        for hue_idx in range(num_hues):
            for key in hue_keys:
                train_params_loc[key] = hue_dictionary[key][hue_idx]
            for key in hue_pt_keys:
                train_params_loc['pretrain_params'][key] = hue_dictionary['pretrain_params'][key][hue_idx]

            for i1, seed in enumerate(seeds):
                train_params_loc['model_seed'] = seed
                torch.manual_seed(seed)
                model, params, run_dir = initialize_and_train.initialize_and_train(load_prev_model=False,
                                                                                   **train_params_loc)
                model.to(device)
                if pretrain:
                    testset_pretrain = params['datasets_pretrain']['train']
                    X_pretrain = []
                    Y_pretrain = []
                    for k in range(NUM_SAMPLES):
                        x, y = testset_pretrain[k]
                        X_pretrain.append(x)
                        if FEEDFORWARD:
                            Y_pretrain.append(y)
                        else:
                            Y_pretrain.append(y[-1])
                    X_pretrain = torch.stack(X_pretrain)
                    Y_pretrain = torch.stack(Y_pretrain)

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
                epochs, saves = loader.get_epochs_and_saves(run_dir)
                frac_epochs = utils.saves_to_partial_epochs(epochs, saves)
                if epoch_list_None:
                    epoch_list = [frac_epochs] * num_hues
                # checkpoints = loader.get_check_nums(run_dir)
                accs = []
                with torch.no_grad():
                    # for k, epoch in enumerate(epoch_list[hue_idx]):
                    # Todo: set up to work with pretrain
                    # for epoch in range(0, num_epochs_pretrain):
                    #     # for save in saves[epoch]:
                    #     for save in [0]:
                    #         print(epoch, save)
                    #         loader.load_model_from_epoch_and_dir(model, run_dir, epoch, save)
                    #         print(model.Wrec.detach()[:4, :4])
                    #         out = model(X_pretrain).detach()
                    #         if not FEEDFORWARD:
                    #             out = out[:, -1]
                    #         out_cat = torch.argmax(out, dim=1)
                    #         acc = torch.mean((out_cat == Y_pretrain).type(torch.float)).item()
                    #         print(acc)
                    #         accs.append(acc)
                    for epoch in epoch_list:
                        # for save in saves[epoch]:
                        for save in [0]:
                            print(epoch, save)
                            loader.load_model_from_epoch_and_dir(model, run_dir, epoch, save)
                            out = model(X).detach()
                            if not FEEDFORWARD:
                                out = out[:, -1]
                            out_cat = torch.argmax(out, dim=1)
                            acc = torch.mean((out_cat == Y).type(torch.float)).item()
                            accs.append(acc)
                    if epoch_plot_None:
                        epoch_plot = epoch_list[hue_idx]
                    if not hue_dictionary_None:
                        if hue_target_idx == 0:
                            hue_target_val = train_params_loc[hue_target[hue_target_idx]]
                        else:
                            hue_target_val = train_params_loc['pretrain_params'][hue_target[hue_target_idx]]
                        d = {'seed': seed, 'num_training_samples': np.array(epoch_plot)*spe, 'accuracy': accs,
                             hue_target_str: hue_target_val}
                    else:
                        d = {'seed': seed, 'num_training_samples': np.array(epoch_plot)*spe, 'accuracy': accs}

                    df = pd.DataFrame(d)
                    loss_and_acc_table = loss_and_acc_table.append(df)
                    # import ipdb; ipdb.set_trace()

        loss_and_acc_table['seed'] = loss_and_acc_table['seed'].astype('category')
        if not hue_dictionary_None:
            loss_and_acc_table[hue_target_str] = loss_and_acc_table[hue_target_str].astype('category')
        return loss_and_acc_table, hue_target_str

    loss_and_acc_table, hue_target_str = memoized_core(train_params, seeds, hue_dictionary, hue_target, epoch_list,
                                                       epoch_plot)

    fig, ax = utils.make_fig(figsize)
    # sns.lineplot(ax=ax, x='epoch', y='accuracy', data=loss_and_acc_table, hue=hue_target_str)
    g = sns.lineplot(ax=ax, x='num_training_samples', y='accuracy', data=loss_and_acc_table, hue=hue_target_str,
                     estimator=est_acc, ci=ci_acc)
    # if g.legend_ is not None:
    #     g.legend_.remove()
    out_fig(fig, figname, train_params, subfolder=train_params['network'] + '/acc_and_loss_over_training/',
            data=loss_and_acc_table)


def cluster_holdout_test_acc_stat_fun(h, y, clust_identity, classifier_type='logistic_regression', num_repeats=5,
                                      train_ratio=0.8, seed=11):
    np.random.seed(seed)
    num_clusts = np.max(clust_identity) + 1
    num_clusts_train = int(round(num_clusts * train_ratio))
    # hid_epoch_count = h.shape[0]
    # num_time_pnts = h.shape[2]
    num_samples = h.shape[0]
    test_accs = np.zeros(num_repeats)
    train_accs = np.zeros(num_repeats)

    for i0 in range(num_repeats):
        permutation = np.random.permutation(np.arange(len(clust_identity)))
        perm_inv = np.argsort(permutation)
        clust_identity_shuffled = clust_identity[permutation]
        train_idx = clust_identity_shuffled <= num_clusts_train
        test_idx = clust_identity_shuffled > num_clusts_train
        # hid_permute = hid[:, permutation]
        hid_train = h[train_idx[perm_inv]]
        y_train = y[train_idx[perm_inv]]
        y_test = y[test_idx[perm_inv]]
        hid_test = h[test_idx[perm_inv]]
        if classifier_type == 'svm':
            classifier = svm.LinearSVC(random_state=3 * i0 + 1)
        else:
            classifier = linear_model.LogisticRegression(random_state=3 * i0 + 1, solver='lbfgs')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier.fit(hid_train, y_train)
        train_accs[i0] = classifier.score(hid_train, y_train)
        test_accs[i0] = classifier.score(hid_test, y_test)

    return train_accs, test_accs


def clust_holdout_over_layers(seeds, gs, train_params, dim_curve_style='before_after',
                              figname="clust_holdout_over_layers"):
    """
    Figures that require instantiation of a single model to generate.
    Args:
        g ():

    Returns:

    """
    # Todo: look at margins before and after training
    # stp()
    # %% Get an example of plot_dim_vs_epoch_and_time compression down to two classes
    if not hasattr(gs, '__len__'):
        gs = [gs]
    g_str = ''
    for g in gs:
        g_str = g_str + '_' + str(g)
    g_str = g_str[1:]
    layer_label = 'layer'

    # @memory.cache()
    def memoized_core_clust_holdout_over_layers(seeds, gs, train_params, dim_curve_style):
        layer_label = 'layer'
        clust_acc_table = pd.DataFrame(columns=['seed', 'g', 'training', layer_label, 'LR training', 'LR testing'])

        # D_eff_responses = []
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

                if dim_curve_style == 'before_after':
                    for epoch, epoch_label in zip([0, -1], ['before', 'after']):
                        loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
                        # hid = model.get_post_activations(X.type(torch.float32))[:-1]
                        hid = [X0]
                        hid += model.get_post_activations(X)[:-1]
                        if len(Y.shape) > 1:
                            Y = Y[:, -1]
                        cluster_identity = class_datasets['train'].cluster_identity
                        # stats = [cluster_holdout_test_acc_stat_fun(h.numpy(), Y.numpy(), cluster_identity) for h in
                        #          hid]
                        ds = []
                        for lay, h in enumerate(hid):
                            stat = cluster_holdout_test_acc_stat_fun(h.numpy(), Y.numpy(), cluster_identity)
                            ds.extend([{'seed': seed, 'g': g, 'training': epoch_label, layer_label: lay, 'LR training':
                                stat[0][k], 'LR testing': stat[1][k]} for k in range(len(stat[0]))])

                            # for k in range(len(stat[0])):
                            #     d = {'seed': seed, 'g': g, 'training': epoch_label, 't': t, 'LR training':
                            #         stat[0][k], 'LR testing': stat[1][k]}
                            #     ds.append(d)

                            # df = pd.DataFrame(d)
                        clust_acc_table = clust_acc_table.append(pd.DataFrame(ds), ignore_index=True)

        clust_acc_table['seed'] = clust_acc_table['seed'].astype('category')
        clust_acc_table['g'] = clust_acc_table['g'].astype('category')
        clust_acc_table['training'] = clust_acc_table['training'].astype('category')
        return clust_acc_table

    clust_acc_table = memoized_core_clust_holdout_over_layers(seeds, gs, train_params, dim_curve_style)
    layers = set(clust_acc_table[layer_label])

    # sns.lineplot(ax=ax, x='t', y=stat_key, data=dim_table, hue='training', style='g')
    # medians = {'train': np.zeros((len(gs), 2, len(layers))), 'test': np.zeros((len(gs), 2, len(layers)))}
    # intervals = {'train': np.zeros((len(gs), 2, len(layers), 2)), 'test': np.zeros((len(gs), 2, len(layers), 2))}
    train_str = ['before', 'after']
    stage_key = {'train': 'LR training', 'test': 'LR testing'}
    train_mark = ['--', '-']
    for stage in ['train', 'test']:
        if stage == 'train':
            clust_acc_table_stage = clust_acc_table.drop(columns=['LR testing'])
        else:
            clust_acc_table_stage = clust_acc_table.drop(columns=['LR training'])
        fig, ax = make_fig((1.5, 1.2))
        g = sns.lineplot(ax=ax, x=layer_label, y=stage_key[stage], data=clust_acc_table_stage, estimator=est_acc,
                         ci=ci_acc,
                         style='training',
                         hue='g'
                         )
        # if g.legend_ is not None:
        #     g.legend_.remove()
        ax.set_ylim([-.01, 1.01])
        ax.set_xticks(range(len(layers)))
        out_fig(fig, figname + '_' + stage, train_params, subfolder=train_params['network'] +
                                                                    '/clust_holdout_over_layers/',
                show=False, save=True, axis_type=0, name_order=0, data=clust_acc_table)

    plt.close('all')


def dim_over_layers(seeds, gs, train_params, dim_curve_style='before_after', figname="dim_over_layers", T=0):
    """
    Figures that require instantiation of a single model to generate.
    Args:
        g ():

    Returns:

    """
    # Todo: look at margins before and after training
    # stp()
    # %% Get an example of plot_dim_vs_epoch_and_time compression down to two classes
    if not hasattr(gs, '__len__'):
        gs = [gs]
    g_str = ''
    for g in gs:
        g_str = g_str + '_' + str(g)
    g_str = g_str[1:]
    stat_key = 'dim'
    layer_label = 'layer'

    # D_eff_responses = []
    train_params_loc = copy.deepcopy(train_params)
    num_epochs = train_params_loc['num_epochs']
    dim_table = pd.DataFrame(columns=['seed', 'g', 'training', layer_label, stat_key])

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
            if T > 0:
                X = utils.extend_input(X, T)
                X0 = X[:, 0]
            elif train_params_loc['network'] != 'feedforward':
                X0 = X[:, 0]
            else:
                X0 = X

            if dim_curve_style == 'before_after':
                for epoch, epoch_label in zip([0, -1], ['before', 'after']):
                    loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
                    # hid = model.get_post_activations(X.type(torch.float32))[:-1]
                    hid = [X0]
                    hid += model.get_post_activations(X)[:-1]
                    if len(Y.shape) > 1:
                        Y = Y[:, -1]
                    stats = [utils.get_effdim(h, False).item() for h in hid]
                    ds = {'seed': seed, 'g': g, 'training': epoch_label, layer_label: list(range(len(hid))),
                          stat_key: stats}
                    dim_table = dim_table.append(pd.DataFrame(ds), ignore_index=True)

    dim_table['seed'] = dim_table['seed'].astype('category')
    dim_table['g'] = dim_table['g'].astype('category')
    dim_table['training'] = dim_table['training'].astype('category')
    layers = set(dim_table[layer_label])

    fig, ax = make_fig((1.5, 1.2))
    g = sns.lineplot(ax=ax, x=layer_label, y=stat_key, data=dim_table, estimator=est_dim,
                     ci=ci_dim, style='training', hue='g')
    # g = sns.lineplot(ax=ax, x=layer_label, y=stat_key, data=dim_table, style='training', hue='g')
    if g.legend_ is not None:
        g.legend_.remove()
    ax.set_xticks(range(len(layers)))
    out_fig(fig, figname, train_params_loc, subfolder=train_params_loc['network'] + '/dim_over_layer/',
            show=False, save=True, axis_type=0, data=dim_table)

    plt.close('all')


def lyaps(seeds, train_params, epochs_plot, figname="lyaps"):
    """
    Figures that require instantiation of a single model to generate.
    Args:
        g ():

    Returns:

    """
    # Todo: look at margins before and after training
    # stp()
    # %% Get an example of plot_dim_vs_epoch_and_time compression down to two classes
    stat_key = 'dim'
    layer_label = 'layer'

    ICs = 'random'

    # D_eff_responses = []
    train_params_loc = copy.deepcopy(train_params)
    num_epochs = train_params_loc['num_epochs']
    lyap_table = pd.DataFrame(columns=['seed', 'epoch', 'lyap', 'lyap_num', 'sem', 'chaoticity'])

    k_LE = 10

    for i0, seed in enumerate(seeds):
        train_params_loc['model_seed'] = seed

        num_pnts_dim_red = 500

        model, params, run_dir = initialize_and_train.initialize_and_train(**train_params_loc)

        # class_datasets = params['datasets']
        # num_train_samples = len(class_datasets['train'])
        # class_datasets['train'].max_samples = num_pnts_dim_red
        torch.manual_seed(train_params_loc['model_seed'])
        # X, Y = class_datasets['train'][:]

        for epoch in epochs_plot:
            loader.load_model_from_epoch_and_dir(model, run_dir, epoch)
            Wrec = model.Wrec.detach().numpy()
            Win = model.Win.detach().numpy()
            Wout = model.Wout.detach().numpy()
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
            # spectra, errors = utils.lyaps_through_training(w, epochs_chaos, ICs, k_LE, max_IC, num_iter,
            #                                               tol=2e-3)
            chaoticity = np.sum(LEs[:3] / np.arange(1, len(LEs[:3]) + 1))
            d = [{'seed': seed, 'epoch': epoch, 'lyap': LEs[k], 'lyap_num': k, 'sem': sem,
                  'chaoticity': chaoticity} for k in range(len(LEs))]
            # d = {'seed': seed, 'g': g, 'epoch': epoch, 'lyap': LEs, 'sem': sem,
            #      'chaoticity': chaoticity}
            lyap_table = lyap_table.append(d, ignore_index=True)

    lyap_table['seed'] = lyap_table['seed'].astype('category')
    lyap_table['epoch'] = lyap_table['epoch'].astype('category')

    fig, ax = make_fig((2, 1.2))
    # g = sns.lineplot(ax=ax, x=layer_label, y=stat_key, data=dim_table, estimator=est_dim,
    #                  ci=ci_dim, style='training', hue='g')
    # g = sns.lineplot(ax=ax, x='lyap_num', y='lyap', data=lyap_table, style='training', hue='g')
    # g = sns.scatterplot(ax=ax, x='lyap_num', y='lyap', data=lyap_table, style='training', hue='g')
    lyap_table_plot = lyap_table.drop(columns=['sem', 'chaoticity'])
    # lyap_table_plot = lyap_table[lyap_table_plot['epoch'] == 0]
    g = sns.pointplot(ax=ax, x='lyap_num', y='lyap', data=lyap_table_plot, style='training', hue='epoch', ci=95,
                      scale=0.5)
    # g.legend_.remove()
    ax.set_xticks(sorted(list(set(lyap_table['lyap_num']))))
    ax.axhline(y=0, color='black', linestyle='--')
    out_fig(fig, figname, train_params_loc, subfolder=train_params_loc['network'] + '/lyaps/',
            show=False, save=True, axis_type=0, data=lyap_table)
    #
    # plt.close('all')


if __name__ == '__main__':
    train_params = dict(N=200,
                        # num_epochs=40,
                        num_epochs=10,
                        # num_epochs=3,
                        num_train_samples_per_epoch=1250,
                        X_clusters=60,
                        X_dim=200,
                        # X_dim=2,
                        num_classes=2,
                        # n_lag=11,
                        # n_lag=10,
                        n_lag=9,
                        # n_lag=51,
                        # n_lag=5001,
                        # n_lag=4,
                        # g_radius=1,
                        g_radius=5,
                        # g_radius=250,
                        clust_sig=.02,
                        # input_scale=10.0,
                        n_hold=1,
                        n_out=1,
                        # loss='mse',
                        loss='cce',
                        # optimizer='sgd',
                        optimizer='rmsprop',
                        # optimizer='adam',
                        # momentum=0.1,
                        dt=.01,
                        # learning_rate=1e-3,
                        learning_rate=1e-4,
                        # learning_rate=1e-5,
                        batch_size=10,
                        freeze_input=False,
                        network='vanilla_rnn',
                        # network='sompolinsky',
                        # network='feedforward',
                        Win=Win,  # todo: refactor code so this can be defined here.
                        Wrec_rand_proportion=.2,
                        patience_before_stopping=6000,
                        # hid_nonlin='relu',
                        hid_nonlin='tanh',
                        # hid_nonlin='linear',
                        # saves_per_epoch=1,
                        model_seed=0,
                        rerun=False)

    train_params_lyap = copy.copy(train_params)
    train_params_lyap['num_epochs'] = 40
    lyaps([0], train_params_lyap, [0, 40])
    train_params_lyap['g_radius'] = 250
    lyaps([0], train_params_lyap, [0, 40])

    # train_params['samples_per_epoch'] = 800
    # batch_size = int(round(train_params['num_train_samples_per_epoch'] * (1 - train_params['perc_val'])))
    # dim_over_training([0], [1, batch_size], [6, 6 * train_params['samples_per_epoch']], train_params)
    # dim_over_training([0, 1], [1, batch_size], [40, 40], train_params)
    # dim_over_training([0,1], [10, 200], [20, 20*train_params['samples_per_epoch']], train_params)

    # activity_visualization(train_params)
    # fn = "dim_over_layers"
    # dim_over_layers(range(5), [5, 250], train_params, colors=chaos_colors, dim_curve_style='before_after',
    #                 comparison='before_after', figname=fn)
    # clust_holdout_over_layers(list(range(5)), [5, 250], train_params, colors=chaos_colors,
    #                           dim_curve_style='before_after',
    #                           comparison='before_after', figname="clust_holdout")
    # acc_and_loss_over_training(train_params, range(5), hue_dictionary={'g_radius': [5, 250]}, hue_target=('g_radius',
    #                                                                                                       None))

    # tp = train_params.copy()
    # tp.update({'network': 'feedforward', 'num_epochs': 10, 'X_dim': 200, 'g_radius': 1, 'hid_nonlin': 'relu'})
    # activity_visualization(tp)
    # dim_over_layers(range(5), [tp['g_radius']], tp, colors=chaos_colors, dim_curve_style='before_after',
    #                 comparison='before_after')
    # clust_holdout_over_layers(list(range(5)), [tp['g_radius']], tp, colors=chaos_colors,
    #                           dim_curve_style='before_after',
    #                           comparison='before_after', figname="clust_holdout")
    # acc_and_loss_over_training(tp, range(5))

    ## Experiments with transfer learning
    # train_params['pretrain_params'] = dict(model_seed=train_params['model_seed'] + 1, num_epochs=10)
    # train_params['g_radius'] = 1
    # acc_and_loss_over_training(train_params, [0], epoch_list=[range(20)], figname="acc_and_loss_g_1")
    #
    # train_params['g_radius'] = 250
    # acc_and_loss_over_training(train_params, [0], epoch_list=[range(20)], figname="acc_and_loss_g_250")
