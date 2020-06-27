import copy
import inspect
from typing import *
import math
import torch
from torch.utils.data import Dataset
import numpy as np
import classification_task
import model_output_manager as mom
import models
import model_trainer
import torchvision
import torchvision.transforms as transforms
import utils

TABLE_PATH = 'output/output_table.csv'

def get_max_eigval(A):
    ew, __ = torch.eig(A)
    mags = torch.sqrt(ew[:, 0]**2 + ew[:, 1]**2)
    idx_sort = torch.flip(torch.argsort(mags), dims=[0])
    ew = ew[idx_sort]
    return ew

def nested_tuple_to_str(inp_tuple):
    temp = ''
    for x in inp_tuple:
        for y in x:
            temp = temp + str(y) + '&'
        temp = temp[:-1]
        temp = temp + '_'
    inp_tuple = temp[:-1]
    return inp_tuple

class LoopedIterator:
    def __init__(self, generator_maker):
        self.generator_maker = generator_maker
        self.generator = generator_maker()

    def __next__(self):
        try:
            return next(self.generator)
        except StopIteration:
            self.generator = self.generator_maker()
            return next(self.generator)

class SubsetSampler(torch.utils.data.Sampler):
    r"""Samples elements from a subset based on indices. May be randomly sampled or deterministic.

    Arguments:
        indices (list): indices to sample from
    """

    def __init__(self, indices, shuffle=False):
        indices = torch.tensor(indices)
        if shuffle:
            shuffle_idx = torch.randperm(len(indices))
            indices = indices[shuffle_idx]
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# class AccuracyTracker:
#     def __init__(self):
#         self.accuracy_list = []
#     def __call__(self, stat_dict):
#         if
#         self.acc_list.append(stat_dict['accuracy'])
#
#     def export_stats(self):
#         out_stats = dict(epoch_losses=self.epoch_losses, epoch_accuracies=self.epoch_accuracies)
#         return out_stats

## Custom types
Numeric = Union[int, float]

## This is the main method for this module
def initialize_and_train(
        N, X_clusters, n_lag, n_hold, n_out, X_dim,
        num_classes, clust_sig=0.1, model_seed=2, hid_nonlin='tanh', num_epochs=20,
        learning_rate=0.005, patience_before_stopping=10, batch_size=10, loss='cce',
        optimizer='rmsprop', momentum=0.9, scheduler='plateau', learning_patience=5,
        scheduler_factor=0.5, Win='orthog', Wrec_rand_proportion=1,
        network='vanilla_rnn', input_scale=1., g_radius=1., dt=0.01,
        num_train_samples_per_epoch=None,
        num_test_samples_per_epoch=None,
        param_l1_regularization_weight=0,
        param_l2_regularization_weight=0,
        activity_l1_regularization_weight=0,
        activity_l2_regularization_weight=0,
        input_dim_regularization_weight: Numeric = 0,
        output_dim_regularization_weight: Numeric = 0,
        dim_target_input_layers: float = 200,
        start_input_layer_to_dim_regularize: int = 0,
        end_input_layer_to_dim_regularize: int = 0,
        dim_target_output_layers: float = 10,
        start_output_layer_to_dim_regularize: int = 0,
        end_output_layer_to_dim_regularize: int = 0,
        cca_regularization_weight: Numeric = 0,
        cca_target: Numeric = 30,
        num_batches_per_actitity_regularization: int = 1,
        num_samples_activity_regularizer: int = 600,
        freeze_input=False,
        input_style='hypercube',
        saves_per_epoch=1,
        # batch_normalization: bool = True,
        rerun: bool = False,
        load_prev_model: bool = True,
        pretrain_params: Optional[dict] = None
):
    """
    Parameters
    ----------
    N : int
        Number of units in the "hidden" layer, i.e. number of neurons making up the recurrent layer.
    X_clusters : int
        Number of clusters.
    n_lag : int
        Number of timesteps from stimulus onset to end of loss evaluation.
    n_hold : int
        Number of timesteps for which the input is presented.
    n_out : int
        Number of timesteps for which loss is evaluated.
    X_dim : int
        Dimension of the ambient space in which clusters are generated.
    num_classes : int
        Number of class labels.
    clust_sig : float
        Standard deviation of each cluster.
    model_seed : int
        Seed for generating input and model weights.
    hid_nonlin : str
        Activation function for the hidden units, or if using a sompolinsky style recurrent network the nonlinear
        transfer function.
    num_epochs : int
        The number of epochs to train for.
    learning_rate : float
        Learning rate for optimizer.
    patience_before_stopping : int
        Number of consecutive epochs to wait for which there is no improvement to the (cumulative average) validation
        loss before ending training.
    batch_size : int
        Size of each training data minibatch.
    loss : str
        The loss function to use. Options are "mse" for mean squared error and "cce" for categorical cross entropy.
    optimizer : str
        The optimizer to use. Options are "sgd" for standard stochastic gradient descent and "rmsprop" for RMSProp.
    perc_val : float
        The percent of input data to use for validation.
    scheduler : str
        The strategy used to adjust the learning rate through training. Options are None for constant learning rate
        through training, "plateau" for reducing the learning rate by a multiplicative factor after a plateau of a
        certain number of epochs, and "steplr" for reducing the learning rate by a multiplicative factor. In both
        cases, the number of epochs is specified by scheduler_patience and the multiplicative factor by
        scheduler_factor.
    learning_patience : int
        If using plateau scheduler, this is the number of epochs over which to measure that a plateau has been
        reached. If using steplr scheduler, this is the number of epochs after which to reduce the learning rate.
    scheduler_factor : float
        The multiplicative factor by which to reduce the learning rate.
    Win : str
        Type of input weights to use. Can be 'diagonal_first_two' for feeding inputs to only the first two neurons
         in the network or 'orthogonal' for a (truncated) orthogonal/unitary matrix.
    Wrec_rand_proportion : float
        The proportion of Wrec that should initially be random. Wrec will be initialized as a convex combination of a
        random matrix and an orthogonal matrix, weighted by Wrec_rand_proportion.
    network : str
        The type of network architecture to use. Options are "vanilla_rnn" for a vanilla RNN, "sompolinsky" for a
        Sompolinsky style RNN, and "feedforward" for a feedforward network.
    input_scale : float
        Global scaling of the inputs.
    g_radius : float
        Magnitude of the largest eigenvalue of the random part of the recurrent weight matrix. This holds exactly
        (i.e. the random matrix is rescaled so that this is satisfied exactly), not just on average.
    dt : float
        Size of the timestep to use for the discretization of the dynamics if using an RNN.
    input_style: str
        Input style. Currently 'hypercube' is the only valid option.
    param_l1_regularization_weight: Numeric
        The weight to use for l1 regularization of the parameters of the model. If 0 (default), don't use
        regularization.
        Code is optimized to be faster if this is set to 0.
    param_l2_regularization_weight: Numeric
        The weight to use for l2 regularization of the parameters of the model. If 0 (default), don't use
        regularization.
        Code is optimized to be faster if this is set to 0 (same for the regularization tomorrow).
    activity_l1_regularization_weight: Numeric
        The weight to use for l1 regularization of the activity of the model. If 0 (default), don't use regularization.
    activity_l2_regularization_weight: Numeric
        The weight to use for l2 regularization of the activity of the model. If 0 (default), don't use regularization.
    input_dim_regularization_weight: Numeric
        The weight to use for effective dimension regularization of the input layers. If 0 (default), don't use
        regularization.
    output_dim_regularization_weight: Numeric
        The weight to use for effective dimension regularization of the output layers. If 0 (default), don't use
        regularization.
    dim_target_input_layers: float
        The target effective dimension for the input layers.
    start_input_layer_to_dim_regularize: int
        The starting input layer to regularize. The layer activations that are regularized are given by
        range(start_input_layer_to_dim_regularize, end_input_layer_to_dim_regularize).
    end_input_layer_to_dim_regularize: int
        The ending input layer to regularize.
    dim_target_output_layers: float
        The target effective dimension for the output layers.
    start_output_layer_to_dim_regularize: int
        The starting output layer to regularize. The layer activations that are regularized are given by
        range(start_output_layer_to_dim_regularize, end_output_layer_to_dim_regularize).
    end_output_layer_to_dim_regularize: int
        The ending output layer to regularize.
    cca_regularization_weight: Numeric
        The weight to use for cca dimension regularization of the input layers. If 0 (default), don't use
        regularization. Uses num_input_layer_to_dim_regularize.
    cca_target: Optional[Numeric]
        The target cca dimension for the input layers.
    num_batches_per_actitity_regularization: int
        Number of batches to go through before applying activity regularization. For instance, if this is 2, then
        activity regularization is used every other batch.
    num_samples_activity_regularizer: int
        Number of input samples to use for the various regularizers on the activity of the network.
    freeze_input: bool
        Whether or not to present the same input every epoch. If False, new input samples are drawn every epoch
    rerun: bool
        Whether or not to run the simulation again even if a matching run is found on disk. True means run the
        simulation again.

    Returns
    -------
        torch.nn.Module
            The trained network model.
        dict
            A collection of all the (meta) parameters used to specify the run. This is basically a dictionary of the
            input arguments to this function.
        str
            The directory where the output for the run is stored, including model parameters over training.
    """
    ## Check to see if there is a pretrain step
    pretrain = not pretrain_params is None and pretrain_params['num_epochs'] > 0
    if not pretrain:
        pretrain_params = {}

    ## Since the pretrain_params dictionary is used so much, it makes code much more concise to use the shorthand "pp"
    pp = pretrain_params

    ## Redefine parameter options for consistency
    if num_train_samples_per_epoch in (None, 'None', 'NA', 'na'):
        num_train_samples_per_epoch = 'na'
    if num_test_samples_per_epoch in (None, 'None', 'NA', 'na'):
        num_test_samples_per_epoch = 'na'
    if num_classes in (None, 'None', 'NA', 'na'):
        num_classes = 'na'
    if hasattr(saves_per_epoch, '__len__'):
        saves_per_epoch_copy = copy.copy(saves_per_epoch)
        saves_per_epoch = str(saves_per_epoch)
    # dataset = dataset.lower()
    network = network.lower()
    loss = loss.lower()
    scheduler = scheduler.lower()
    optimizer = optimizer.lower()
    learning_patience_copy = copy.copy(learning_patience)
    if hasattr(learning_patience, '__len__'):
        learning_patience = '_'.join([str(x) for x in learning_patience])
    if optimizer == 'adam':
        momentum = 0

    ## Record the input parameters in a dictionary
    loc = locals()
    args = inspect.getfullargspec(initialize_and_train)[0]
    arg_dict = {arg: loc[arg] for arg in args}
    print("initialize_and_train called with parameters ", arg_dict)
    del arg_dict['rerun']
    del arg_dict['pretrain_params']
    del arg_dict['load_prev_model']

    learning_patience = learning_patience_copy

    ## Redefine parameter options for consistency
    for key, value in arg_dict.items():
        if value in (None, 'None', 'NA'):
            arg_dict[key] = 'na'

    ## Process pretrain_params to fill in parameters that aren't specified, with 'na' (or 0)
    pretrain_possible_vars = {key for key in arg_dict}
    pretrain_possible_vars.discard('network')
    pretrain_possible_vars.discard('model_seed')

    ## Set default parameter values for pretrain parameters
    if pretrain:
        for key in pretrain_possible_vars:
            if key not in pp or pp[key] in (None, 'None', 'NA', 'na'):
                pp[key] = 'na'
        defaults_set_to_zero = ['param_l1_regularization_weight', 'param_l2_regularization_weight',
                                'activity_l1_regularization_weight', 'activity_l2_regularization_weight',
                                'input_dim_regularization_weight', 'output_dim_regularization_weight',
                                'start_input_layer_to_dim_regularize', 'end_input_layer_to_dim_regularize',
                                'start_output_layer_to_dim_regularize', 'end_output_layer_to_dim_regularize',
                                'cca_regularization_weight']
        if pp['loss'] == 'na':
            pp['loss'] = loss
        if pp['num_batches_per_actitity_regularization'] == 'na':
            pp['num_batches_per_actitity_regularization'] = num_batches_per_actitity_regularization
        if pp['dim_target_input_layers'] == 'na':
            pp['dim_target_input_layers'] = dim_target_input_layers
        if pp['dim_target_output_layers'] == 'na':
            pp['dim_target_output_layers'] = dim_target_output_layers
        if pp['learning_rate'] == 'na':
            pp['learning_rate'] = learning_rate
        if pp['patience_before_stopping'] == 'na':
            pp['patience_before_stopping'] = 100000
        if pp['batch_size'] == 'na':
            pp['batch_size'] = batch_size
        if pp['optimizer'] == 'na':
            pp['optimizer'] = optimizer
        if pp['momentum'] == 'na':
            pp['momentum'] = momentum
        if pp['scheduler'] == 'na':
            pp['scheduler'] = scheduler
        if pp['learning_patience'] == 'na':
            pp['learning_patience'] = learning_patience
        if pp['scheduler_factor'] == 'na':
            pp['scheduler_factor'] = scheduler_factor
        if pp['cca_target'] == 'na':
            pp['cca_target'] = cca_target
        if pp['num_train_samples_per_epoch'] == 'na':
            pp['num_train_samples_per_epoch'] = num_train_samples_per_epoch
        if pp['num_samples_activity_regularizer'] == 'na':
            pp['num_samples_activity_regularizer'] = num_samples_activity_regularizer
        # if pp['perc_val'] == 'na':
        #     pp['perc_val'] = perc_val
        if pp['X_dim'] == 'na':
            pp['X_dim'] = X_dim
        if pp['num_classes'] == 'na':
            pp['num_classes'] = num_classes
        if pp['X_clusters'] == 'na':
            pp['X_clusters'] = X_clusters
        if pp['n_hold'] == 'na':
            pp['n_hold'] = n_hold
        if pp['n_lag'] == 'na':
            pp['n_lag'] = n_lag
        if pp['clust_sig'] == 'na':
            pp['clust_sig'] = clust_sig
        if pp['freeze_input'] == 'na':
            pp['freeze_input'] = freeze_input

        for key in defaults_set_to_zero:
            if pp[key] == 'na':
                pp[key] = 0
        if hasattr(pp['saves_per_epoch'], '__len__'):
            saves_per_epoch_pretrain_copy = copy.copy(pp['saves_per_epoch'])
            pp['saves_per_epoch'] = str(pp['saves_per_epoch'])

        if pp['input_dim_regularization_weight'] == 0:
            pp['dim_target_input_layers'] = 'na'
            # pp['num_input_layer_to_dim_regularize'] = 0
            pp['start_input_layer_to_dim_regularize'] = 0
            pp['end_input_layer_to_dim_regularize'] = 0
        if pp['output_dim_regularization_weight'] == 0:
            pp['dim_target_output_layers'] = 'na'
            pp['start_output_layer_to_dim_regularize'] = 0
            pp['end_output_layer_to_dim_regularize'] = 0

        ## Check that pretrain_params is valid and update arg_dict to include items in pretrain_params
        if not set(pp.keys()).issubset(arg_dict.keys()):
            print(set(pp.keys()).difference(arg_dict.keys()))
            breakpoint()
            raise ValueError('pretrain_params key not valid.')

        # pp['dataset'] = pp['dataset'].lower()
        pp['loss'] = pp['loss'].lower()
        pp['scheduler'] = pp['scheduler'].lower()
        pp['optimizer'] = pp['optimizer'].lower()

        learning_patience_pretrain_copy = copy.copy(pp['learning_patience'])
        if hasattr(pp['learning_patience'], '__len__'):
            pp['learning_patience'] = '_'.join([str(x) for x in pp['learning_patience']])

        pp_renamed = {}
        for key in list(pp.keys()):
            pp_renamed[key + '_pretrain'] = pp[key]
        arg_dict.update(pp_renamed)

        pp['learning_patience'] = learning_patience_pretrain_copy

    learning_patience = learning_patience_copy
    if isinstance(saves_per_epoch, str):
        saves_per_epoch = saves_per_epoch_copy
    if pretrain:
        if isinstance(pp['saves_per_epoch'], str):
            pp['saves_per_epoch'] = saves_per_epoch_pretrain_copy

    #### Okay, with parameter dictionaries out of the way, the real coding can begin.
    torch.manual_seed(model_seed)

    ### Initialize Data. todo: write code to enable having different pretraining data.
    print('==> Preparing data..')

    ## Pretraining datasets
    if pretrain:
        if network == 'feedforward':
            out = classification_task.delayed_mixed_gaussian(pp['num_train_samples_per_epoch'],
                                                             .2, pp['X_dim'],
                                                             pp['num_classes'], pp['X_clusters'],
                                                             1, 0, clust_sig, 2*pp['model_seed'] + 1,
                                                             3*pp['model_seed'] + 13, cluster_method=5, avg_magn=1,
                                                             freeze_input=freeze_input)
        else:
            out = classification_task.delayed_mixed_gaussian(pp['num_train_samples_per_epoch'],
                                                             .2, pp['X_dim'],
                                                             pp['num_classes'], pp['X_clusters'],
                                                             pp['n_hold'], pp['n_lag'], pp['clust_sig'],
                                                             2*pp['model_seed'] + 1,
                                                             3*pp['model_seed'] + 13, cluster_method=5, avg_magn=1,
                                                             freeze_input=pp['freeze_input'])

        datasets_pretrain, centers_pretrain, cluster_class_label_pretrain = out
        trainset_pretrain = datasets_pretrain['train']
        testset_pretrain = datasets_pretrain['val']

        if pp['num_train_samples_per_epoch'] != 'na':
            subset_indices = range(pp['num_train_samples_per_epoch'])
            # subset_indices = indices_for_classes_train[
            #     torch.randperm(len(indices_for_classes_train))[:num_train_samples_per_epoch]]
            trainloader_pretrain = torch.utils.data.DataLoader(trainset_pretrain, batch_size=pp['batch_size'],
                                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                                   subset_indices),
                                                               num_workers=0)
        else:
            trainloader_pretrain = torch.utils.data.DataLoader(trainset_pretrain, batch_size=pp['batch_size'],
                                                               num_workers=0,
                                                               shuffle=True)

        trainloader_regularizer_pretrain = torch.utils.data.DataLoader(trainset_pretrain,
                                                                       batch_size=pp[
                                                                           'num_samples_activity_regularizer'],
                                                                       shuffle=True,
                                                                       num_workers=0)
        def make_trainloader_iterator_pretrain():
            return iter(trainloader_regularizer_pretrain)
        regularizer_looped_iterator_pretrain = LoopedIterator(make_trainloader_iterator_pretrain)

        # testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
        if pp['num_test_samples_per_epoch'] != 'na':
            subset_indices = range(pp['num_test_samples_per_epoch'])
            # subset_indices = indices_for_classes_train[
            #     torch.randperm(len(indices_for_classes_test))[:num_test_samples_per_epoch]]
            # subset_indices = range(num_test_samples_per_epoch)
            testloader_pretrain = torch.utils.data.DataLoader(testset_pretrain, batch_size=pp['batch_size'],
                                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                                  subset_indices),
                                                              num_workers=0)
        else:
            testloader_pretrain = torch.utils.data.DataLoader(testset_pretrain, batch_size=pp['batch_size'],
                                                              num_workers=0,
                                                              shuffle=True)

        datasets_pretrain = {'train': trainset_pretrain, 'val': testset_pretrain}
        dataloaders_pretrain = {'train': trainloader_pretrain, 'val': testloader_pretrain}

    ### Training datasets
    # if num_test_samples_per_epoch == 'na':
    #     num_test_samples_per_epoch = len(datasets['train'])
    # perc_val = num_test_samples_per_epoch / (num_test_samples_per_epoch+num_test_samples_per_epoch)
    perc_val = 0.2  # todo: resolve
    if network == 'feedforward':
        out = classification_task.delayed_mixed_gaussian(num_train_samples_per_epoch, perc_val, X_dim, num_classes,
                                                         X_clusters,
                                                         1, 0, clust_sig, 2*model_seed + 1,
                                                         3*model_seed + 13, cluster_method=5, avg_magn=1,
                                                         freeze_input=freeze_input)
    else:
        out = classification_task.delayed_mixed_gaussian(num_train_samples_per_epoch, perc_val, X_dim, num_classes,
                                                         X_clusters,
                                                         n_hold, n_lag, clust_sig, 2*model_seed + 1,
                                                         3*model_seed + 13, cluster_method=5, avg_magn=1,
                                                         freeze_input=freeze_input)

    datasets, centers, cluster_class_label = out
    trainset = datasets['train']
    testset = datasets['val']
    # indices_for_classes_train =

    if num_train_samples_per_epoch != 'na':
        subset_indices = range(num_train_samples_per_epoch)
        # subset_indices = indices_for_classes_train[
        #     torch.randperm(len(indices_for_classes_train))[:num_train_samples_per_epoch]]
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
                                                  num_workers=0)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True)

    trainloader_regularizer = torch.utils.data.DataLoader(trainset, batch_size=num_samples_activity_regularizer,
                                                          shuffle=True,
                                                          num_workers=0)
    def make_trainloader_iterator():
        return iter(trainloader_regularizer)
    regularizer_looped_iterator = LoopedIterator(make_trainloader_iterator)

    # testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    if num_test_samples_per_epoch != 'na':
        subset_indices = range(num_test_samples_per_epoch)
        # subset_indices = indices_for_classes_train[
        #     torch.randperm(len(indices_for_classes_test))[:num_test_samples_per_epoch]]
        # subset_indices = range(num_test_samples_per_epoch)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices),
                                                 num_workers=0)
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0, shuffle=True)

    datasets = {'train': trainset, 'val': testset}
    dataloaders = {'train': trainloader, 'val': testloader}

    # print(len(trainloader))

    ## Convenience functions and variable definitions
    def ident(x):
        return x
    def zero_fun():
        return 0
    if hid_nonlin == 'linear'.casefold():
        nonlin = ident
    elif hid_nonlin == 'tanh'.casefold():
        nonlin = torch.tanh
    elif hid_nonlin == 'relu'.casefold():
        nonlin = torch.relu
    else:
        raise ValueError('Unrecognized option for hid_nonlin')

    class NonzeroCounter:
        def __init__(self, jump_size):
            self.jump_size = jump_size
            self.cnt = 0
        def __call__(self):
            self.cnt = self.cnt + 1
            self.cnt = self.cnt%self.jump_size
            return self.cnt == 0

    activity_regularizer_counter = NonzeroCounter(num_batches_per_actitity_regularization)
    def l1_regularizer(param_list):
        reg = 0
        cnt = 0
        for p in param_list:
            # mean is over all elements of p, even if p is a matrix
            reg += torch.mean(torch.abs(p))
            cnt = cnt + 1
        return reg/cnt
    def l2_regularizer(param_list):
        # mean is over all elements of p, even if p is a matrix
        reg = 0
        cnt = 0
        for p in param_list:
            # U, s, Vt = torch.svd(p, some=True, compute_uv=False)
            # reg += param_regularization_weight*s[0]
            reg += torch.norm(p, 'fro')
            cnt = cnt + 1
        return reg/cnt
    def l1_activity_regularizer():
        x, targ = next(regularizer_looped_iterator)
        x = x.to(device[0])
        hs = model.get_activations(x, detach=False)
        reg = 0
        cnt = 0
        for h in hs:
            # mean is over all elements of p, even if p is a matrix
            reg += torch.mean(torch.abs(h))
            cnt = cnt + 1
        return reg/cnt
    def l2_activity_regularizer():
        x, targ = next(regularizer_looped_iterator)
        x = x.to(device[0])
        hs = model.get_activations(x, detach=False)
        reg = 0
        cnt = 0
        for h in hs:
            # U, s, Vt = torch.svd(p, some=True, compute_uv=False)
            # reg += param_regularization_weight*s[0]
            reg += torch.norm(h, 'fro')
            cnt = cnt + 1
        return reg/cnt
    def effective_dim_regularizer(dim_target, layer_idx):
        x, targ = next(regularizer_looped_iterator)
        x = x.to(device[0])
        hs = model.get_activations(x, detach=False)
        hs = [h.to('cpu') for h in hs]

        dim_loss = 0
        dim_avg = 0
        for k in layer_idx:
            print(k)
            h = hs[k]  # get the second layer activations.
            h = h.reshape(h.shape[0], -1)

            dim = utils.get_effdim(h.to(device[0]))
            dim_avg = dim_avg + dim
            dim_loss = dim_loss + (dim - dim_target)**2

        dim_avg = dim_avg/len(layer_idx)
        print(dim_avg)
        return dim_loss/len(layer_idx)

    def cca_regularizer(dim_target, layer_idx):
        x, targ = next(regularizer_looped_iterator)
        x = x.to(device[0])
        hs = model.get_activations(x, detach=False)
        dim_loss = 0
        m = 20
        for k in range(1, m + 1):
            h1 = hs[k]  # get the second layer activations.
            h2 = hs[k]  # get the second layer activations.
            h1_reshaped = h1.reshape(h1.shape[0], -1)
            h2_reshaped = h2.reshape(h2.shape[0], -1)

            dim = utils.get_effcca(h1_reshaped, h2_reshaped)
            dim_loss = dim_loss + (dim - dim_target)**2

        return dim_loss/m

    ## Find requested network model and put model on appropriate device
    if Win in ('identity', 'diagonal_first_two'):
        Win_instance = input_scale*torch.eye(X_dim, N).T.clone()
        # Win_instance = input_scale*torch.eye(N, X_dim)
    elif Win in ('orth', 'orthogonal', 'orthog'):
        temp = torch.empty(X_dim, N)
        temp = torch.nn.init.orthogonal_(temp)
        temp = temp/torch.mean(torch.abs(temp))
        temp = input_scale*temp/math.sqrt(X_dim)
        Win_instance = temp.T.clone()
    else:
        raise AttributeError("Win option not recognized.")

    Wout_instance = torch.randn(num_classes, N)*(.3/math.sqrt(N))

    brec = torch.zeros(N)
    bout = torch.zeros(num_classes)
    J = torch.randn(N, N)/math.sqrt(N)
    top_ew = get_max_eigval(J)[0]
    top_ew_mag = torch.sqrt(top_ew[0]**2 + top_ew[1]**2)
    J_scaled = g_radius*(J/top_ew_mag)
    Q = torch.nn.init.orthogonal_(torch.empty(N, N))
    Q_scaled = g_radius*Q
    if network in ('somp', 'sompolinsky', 'sompolinskyrnn'):
        Wrec = Wrec_rand_proportion*J_scaled + (1 - Wrec_rand_proportion)*Q
        model = models.SompolinskyRNN(Win_instance, Wrec, Wout_instance, brec, bout, nonlin, dt=dt,
                                      output_over_recurrent_time=True)

    elif network == 'vanilla_rnn'.casefold():
        Wrec = (1 - dt)*torch.eye(N, N) + dt*g_radius*(J/top_ew_mag)
        model = models.RNN(Win_instance, Wrec, Wout_instance, brec, bout, nonlin, output_over_recurrent_time=True)

    elif network == 'feedforward'.casefold():
        # layer_weights: List[Tensor], biases: List[Tensor], nonlinearities: List[Callable]
        Wrec = (1 - dt)*torch.eye(N, N) + dt*g_radius*(J/top_ew_mag)
        # Wrec = g_radius * (J / top_ew_mag)
        layer_weights = [Win_instance]
        biases = [torch.zeros(N)]
        nonlinearities = [nonlin]
        for i0 in range(n_lag + n_hold - 1):
            layer_weights.append(Wrec.clone())
            biases.append(torch.zeros(N))
            nonlinearities.append(nonlin)
        layer_weights.append(Wout_instance)
        biases.append(torch.zeros(num_classes))
        nonlinearities.append(ident)

        model = models.FeedForward(layer_weights, biases, nonlinearities)
    else:
        raise AttributeError('Option for net_architecture not recognized.')
    # if torch.cuda.device_count() == 2:
    #     device = [torch.device("cuda:0"), torch.device("cuda:1")]
    # elif torch.cuda.device_count() == 1:
    #     device = [torch.device("cuda:0")]
    # else:
    #     device = [torch.device("cpu")]
    device = [torch.device("cpu"), torch.device("cpu")]

    print("Using {}".format(device[0]))
    model = model.to(device[0])

    ## Get layers that correspond to dimensionality regularization.
    model.layer_names = ['']
    num_layers = len(model.layer_names)
    input_layer_idx = range(start_input_layer_to_dim_regularize, end_input_layer_to_dim_regularize)
    output_layer_idx = range(start_output_layer_to_dim_regularize, end_output_layer_to_dim_regularize)
    if pretrain:
        input_layer_idx_pretrain = range(pp['start_input_layer_to_dim_regularize'],
                                         pp['end_input_layer_to_dim_regularize'])
        output_layer_idx_pretrain = range(pp['start_output_layer_to_dim_regularize'],
                                          pp['end_output_layer_to_dim_regularize'])

    ## Initialize regularizers for the pretraining stage.
    if pretrain:
        if pp['param_l1_regularization_weight'] > 0:
            def param_l1_regularization_f_pretrain():
                return pp['param_l1_regularization_weight']*l1_regularizer(model.params())
        else:
            param_l1_regularization_f_pretrain = zero_fun
        if pp['param_l2_regularization_weight'] > 0:
            def param_l2_regularization_f_pretrain():
                return pp['param_l2_regularization_weight']*l2_regularizer(model.params())
        else:
            param_l2_regularization_f_pretrain = zero_fun
        if pp['activity_l1_regularization_weight'] > 0:
            def activity_l1_regularization_f_pretrain():
                return pp['activity_l1_regularization_weight']*l1_activity_regularizer()
        else:
            activity_l1_regularization_f_pretrain = zero_fun
        if pp['activity_l2_regularization_weight'] > 0:
            def activity_l2_regularization_f_pretrain():
                return pp['activity_l2_regularization_weight']*l2_activity_regularizer()
        else:
            activity_l2_regularization_f_pretrain = zero_fun
        if pp['input_dim_regularization_weight'] > 0:

            def input_dim_regularization_f_pretrain():
                dt = pp['dim_target_input_layers']
                return pp['input_dim_regularization_weight']*effective_dim_regularizer(dt, input_layer_idx_pretrain)
        else:
            input_dim_regularization_f_pretrain = zero_fun
        if pp['output_dim_regularization_weight'] > 0:
            def output_dim_regularization_f_pretrain():
                dt = pp['dim_target_output_layers']
                return pp['output_dim_regularization_weight']*effective_dim_regularizer(dt, output_layer_idx_pretrain)
        else:
            output_dim_regularization_f_pretrain = zero_fun
        if pp['cca_regularization_weight'] > 0:
            def cca_regularization_f_pretrain():
                return pp['cca_regularization_weight']*cca_regularizer(cca_target, output_layer_idx_pretrain)
        else:
            cca_regularization_f_pretrain = zero_fun

    ## Initializing regularizers for the training stage
    if param_l1_regularization_weight > 0:
        def param_l1_regularization_f():
            return param_l1_regularization_weight*l1_regularizer(model.params())
    else:
        param_l1_regularization_f = zero_fun
    if param_l2_regularization_weight > 0:
        def param_l2_regularization_f():
            return param_l2_regularization_weight*l2_regularizer(model.params())
    else:
        param_l2_regularization_f = zero_fun
    if activity_l1_regularization_weight > 0:
        def activity_l1_regularization_f():
            return activity_l1_regularization_weight*l1_activity_regularizer()
    else:
        activity_l1_regularization_f = zero_fun
    if activity_l2_regularization_weight > 0:
        def activity_l2_regularization_f():
            return activity_l2_regularization_weight*l2_activity_regularizer()
    else:
        activity_l2_regularization_f = zero_fun
    if input_dim_regularization_weight > 0:
        def input_dim_regularization_f():
            return input_dim_regularization_weight*effective_dim_regularizer(dim_target_input_layers, input_layer_idx)
    else:
        input_dim_regularization_f = zero_fun
    if output_dim_regularization_weight > 0:
        def output_dim_regularization_f():
            return output_dim_regularization_weight*effective_dim_regularizer(dim_target_output_layers,
                                                                              output_layer_idx)
    else:
        output_dim_regularization_f = zero_fun
    if cca_regularization_weight > 0:
        def cca_regularization_f():
            return cca_regularization_weight*cca_regularizer(cca_target, output_layer_idx)
    else:
        cca_regularization_f = zero_fun

    if network != 'feedforward':
        loss_points = torch.arange(n_lag - n_out, n_lag + n_hold - 1)
    else:
        loss_points = torch.tensor([0], dtype=int)
    # num_train = int(round((1 - perc_val)*num_train_samples_per_epoch))

    ## Initializing loss functions for the pretraining stage
    if pretrain:
        m = len(loss_points)
        if pretrain and 'loss' in pp:
            if pp['loss'] in ('categorical_crossentropy', 'cce'):
                loss_CEL = torch.nn.CrossEntropyLoss()
                if network == 'feedforward':
                    loss_function_pretrain = loss_CEL
                else:
                    def loss_function_pretrain(output, label):
                        return loss_CEL(output[:, loss_points].transpose(1,2), label[:, loss_points])
            elif pp['loss'] in ('mean_square_error', 'mse'):
                criterion_mse = torch.nn.MSELoss()
                def criterion_single_timepoint(output, label):  # The output does not have a time dimension
                    label_onehot = torch.zeros(label.shape[0], num_classes)
                    for i0 in range(num_classes):
                        label_onehot[label == i0, i0] = 1
                    return criterion_mse(output, .7*label_onehot)
                def loss_function_pretrain(output, label):
                    cum_loss = 0
                    for i0 in loss_points:
                        cum_loss += criterion_single_timepoint(output[:, i0], label[:, i0])
                    cum_loss = cum_loss/m
                    return cum_loss
            elif pp['loss'] == 'zero':
                def loss_function_pretrain(output, label):
                    return 0
            else:
                raise AttributeError("pretrain loss option not recognized.")

        def regularized_loss_pretrain(output, label):
            crit = loss_function_pretrain(output, label)
            if activity_regularizer_counter():
                act_reg = (activity_l1_regularization_f_pretrain() + activity_l2_regularization_f_pretrain()
                           + input_dim_regularization_f_pretrain() + output_dim_regularization_f_pretrain() +
                           cca_regularization_f_pretrain())
            else:
                act_reg = 0
            param_reg = param_l1_regularization_f_pretrain() + param_l2_regularization_f_pretrain()
            return crit + act_reg + param_reg

    ## Initializing loss functions for the training stage
    if loss in ('categorical_crossentropy', 'cce'):
        loss_CEL = torch.nn.CrossEntropyLoss()
        if network == 'feedforward':
            loss_function = loss_CEL
        else:
            def loss_function(output, label):
                return loss_CEL(output[:, loss_points].transpose(1, 2), label[:, loss_points])
    elif loss in ('mean_square_error', 'mse'):
        criterion_mse = torch.nn.MSELoss()
        def criterion_single_timepoint(output, label):  # The output does not have a time dimension
            label_onehot = torch.zeros(label.shape[0], num_classes)
            for i0 in range(num_classes):
                label_onehot[label == i0, i0] = 1
            return criterion_mse(output, .7*label_onehot)
        def loss_function(output, label):
            cum_loss = 0
            for i0 in loss_points:
                cum_loss += criterion_single_timepoint(output[:, i0], label[:, i0])
            cum_loss = cum_loss/m
            return cum_loss
    elif loss == 'zero':
        def loss_function(output, label):
            return 0
    else:
        raise AttributeError("loss option not recognized.")

    def regularized_loss(output, label):
        crit = loss_function(output, label)
        if activity_regularizer_counter():
            act_reg = (activity_l1_regularization_f() + activity_l2_regularization_f() + input_dim_regularization_f() +
                       output_dim_regularization_f() + cca_regularization_f())
        else:
            act_reg = 0
        param_reg = param_l1_regularization_f() + param_l2_regularization_f()
        return crit + act_reg + param_reg

    ## Initialize optimizer and learning scheduler for pretraining stage
    if pretrain:
        if pp['optimizer'] == 'sgd':
            # optimizer_instance = torch.optim.SGD(net_par.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            optimizer_instance_pretrain = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                                          lr=pp['learning_rate'], momentum=pp['momentum'])
        elif pp['optimizer'] == 'rmsprop':
            # noinspection PyUnresolvedReferences
            optimizer_instance_pretrain = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                                              lr=pp['learning_rate'], momentum=pp['momentum'])
        elif pp['optimizer'] == 'adam':
            # noinspection PyUnresolvedReferences
            optimizer_instance_pretrain = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                                           lr=pp['learning_rate'])
        else:
            raise AttributeError('pretrain optimizer option not recognized.')
        if pretrain:
            if pp['scheduler'] == 'plateau':
                learning_scheduler_torch_pretrain = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_instance_pretrain,
                    factor=scheduler_factor,
                    patience=learning_patience,
                    threshold=1e-7,
                    threshold_mode='abs',
                    min_lr=0,
                    verbose=True)
            elif pp['scheduler'] == 'steplr':
                learning_scheduler_torch_pretrain = torch.optim.lr_scheduler.StepLR(optimizer_instance_pretrain,
                                                                                    step_size=pp['learning_patience'],
                                                                                    gamma=pp['scheduler_factor'])
            elif pp['scheduler'] == 'multisteplr':
                learning_scheduler_torch_pretrain = torch.optim.lr_scheduler.MultiStepLR(optimizer_instance_pretrain,
                                                                                         pp['learning_patience'],
                                                                                         pp['scheduler_factor'])
            else:
                raise AttributeError('pretrain scheduler option not recognized.')
            learning_scheduler_instance_pretrain = model_trainer.DefaultLearningScheduler(
                learning_scheduler_torch_pretrain)

    ## Initialize optimizer and learning scheduler for training stage
    if optimizer == 'sgd':
        # optimizer_instance = torch.optim.SGD(net_par.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer_instance = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=learning_rate, momentum=momentum)
    elif optimizer == 'rmsprop':
        # noinspection PyUnresolvedReferences
        optimizer_instance = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                                 lr=learning_rate, momentum=momentum)
    elif optimizer == 'adam':
        # noinspection PyUnresolvedReferences
        optimizer_instance = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                              lr=learning_rate)
    else:
        raise AttributeError('optimizer option not recognized.')
    if scheduler == 'plateau':
        learning_scheduler_torch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_instance,
                                                                              factor=scheduler_factor,
                                                                              patience=learning_patience,
                                                                              threshold=1e-7,
                                                                              threshold_mode='abs',
                                                                              min_lr=0,
                                                                              verbose=True)
    elif scheduler == 'steplr':
        learning_scheduler_torch = torch.optim.lr_scheduler.StepLR(optimizer_instance, step_size=learning_patience,
                                                                   gamma=scheduler_factor)
    elif scheduler == 'multisteplr':
        learning_scheduler_torch = torch.optim.lr_scheduler.MultiStepLR(optimizer_instance,
                                                                        learning_patience,
                                                                        scheduler_factor)
    else:
        raise AttributeError('scheduler option not recognized.')
    learning_scheduler_instance = model_trainer.DefaultLearningScheduler(learning_scheduler_torch)

    # stats_trackers = {x: model_trainer.DefaultStatsTracker(batches_per_epoch[x], x, accuracy=False) for x in
    #                   ('train', 'val')}

    ## Determine if the training needs to be run again or if it can be loaded from disk
    dirs, ids, output_exists = mom.get_dirs_and_ids_for_run(arg_dict, TABLE_PATH, ['num_epochs'], maximize='num_epochs')
    if len(dirs) == 0:
        run_id, run_dir = mom.make_dir_for_run(arg_dict, TABLE_PATH)
    else:
        run_id = ids[0]
        run_dir = dirs[0]
    ## Now train the model (if necessary)
    saves_per_epoch_is_number = not hasattr(saves_per_epoch, '__len__')
    batches_per_epoch = len(trainloader)
    if saves_per_epoch_is_number and saves_per_epoch > 1:
        mod_factor = int(math.ceil((batches_per_epoch - 1)/saves_per_epoch))
        print(mod_factor)
        def save_model_criterion(stat_dict):
            # print('batch: ', stat_dict['batch'], '  out of: ', batches_per_epoch, '  mod_factor: ', mod_factor)
            return stat_dict['batch']%mod_factor == 0
    elif saves_per_epoch_is_number:
        epochs_per_save = round(1/saves_per_epoch)
        def save_model_criterion(stat_dict):
            # print('save_model_factor:', stat_dict['batch'] % mod_factor)
            save_epoch = (stat_dict['epoch'] - 1)%epochs_per_save == 0 or stat_dict['epoch'] == 0
            return stat_dict['epoch_end'] and save_epoch
    else:
        def save_model_criterion(stat_dict):
            # print('save_model_factor:', stat_dict['batch'] % mod_factor)
            saves_this_epoch = saves_per_epoch[stat_dict['epoch']]
            if saves_this_epoch == 1 and stat_dict['epoch_end']:
                return True
            elif saves_this_epoch > 1:
                mod_factor = int(math.ceil((batches_per_epoch - 1)/saves_this_epoch))
                return stat_dict['batch']%mod_factor == 0
            else:
                return False

    if pretrain:  # First pretraining stages.
        start_epoch = pp['num_epochs'] + 1
        total_num_epochs = num_epochs + pp['num_epochs']
        # This modifies model by reference
        # model.batch_normalization = pp['batch_normalization']
        print('\n==> Pretraining network')

        model_trainer.train_model(model, dataloaders_pretrain, device[0], regularized_loss_pretrain,
                                  optimizer_instance_pretrain, pp['num_epochs'], run_dir,
                                  not rerun,
                                  learning_scheduler=learning_scheduler_instance_pretrain,
                                  save_model_criterion=save_model_criterion)
        # stats_history_pretrain = history_and_machinery['stats_history']
    else:
        start_epoch = 0
        total_num_epochs = num_epochs
    # model.batch_normalization = batch_normalization
    # This modifies model by reference
    print('\n==> Training network')
    # import ipdb; ipdb.set_trace()
    load_prev = pretrain or not rerun
    model_trainer.train_model(model, dataloaders, device[0], regularized_loss,
                              optimizer_instance, total_num_epochs, run_dir,
                              load_prev,
                              learning_scheduler=learning_scheduler_instance,
                              save_model_criterion=save_model_criterion)
    # stats_history = history_and_machinery['stats_history']
    # stats_trackers = history_and_machinery['stats_trackers']
    # learning_scheduler_instance = history_and_machinery['learning_scheduler']
    # optimizer_instance = history_and_machinery['optimizer']
    if pretrain:
        params = dict(dataloaders=dataloaders, dataloaders_pretrain=dataloaders_pretrain,
                      datasets=datasets, datasets_pretrain=datasets_pretrain)
    else:
        params = dict(dataloaders=dataloaders, datasets=datasets)
    params.update(arg_dict)
    # params = arg_dict
    # outputs = dict(stats_history=stats_history, stats_trackers=stats_trackers)
    # outputs = dict(stats_history=stats_history)
    # if pretrain:
    #     outputs['stats_history_pretrain'] = stats_history_pretrain

    # mom.write_output(outputs, {}, arg_dict, run_dir, overwrite=True)
    return model, params, run_dir

if __name__ == '__main__':
    # Instantiate and train a model based on passed hyperparameters.
    network = 'sompolinsky'
    # network = 'vanilla_rnn'
    # network = 'feedforward'
    model, params, run_dir = initialize_and_train(200, 60, 11, 1, 1, 200, 2,
                                                  network=network,
                                                  num_epochs=10,
                                                  num_train_samples_per_epoch=1250,
                                                  num_test_samples_per_epoch=20,
                                                  # optimizer='sgd',
                                                  optimizer='rmsprop',
                                                  g_radius=1,
                                                  input_scale=10.0,
                                                  clust_sig=.02,
                                                  learning_rate=1e-4,
                                                  batch_size=10,
                                                  freeze_input=False,
                                                  Wrec_rand_proportion=.2,
                                                  hid_nonlin='tanh',
                                                  dt=.01,
                                                  )


# # def train_extracted(model, N, X_clusters, n_lag, n_hold, n_out, X_dim,
# #         num_classes, clust_sig=0.1, model_seed=2, hid_nonlin='tanh', num_epochs=20,
# #         learning_rate=0.005, patience_before_stopping=10, batch_size=10, loss='cce',
# #         optimizer='rmsprop', momentum=0.9, scheduler='plateau', learning_patience=5,
# #         scheduler_factor=0.5, Win='orthog', Wrec_rand_proportion=1,
# #         network='vanilla_rnn', input_scale=1., g_radius=1., dt=0.01,
# #         num_train_samples_per_epoch=None,
# #         num_test_samples_per_epoch=None,
# #         param_l1_regularization_weight=0,
# #         param_l2_regularization_weight=0,
# #         activity_l1_regularization_weight=0,
# #         activity_l2_regularization_weight=0,
# #         input_dim_regularization_weight=0,
# #         output_dim_regularization_weight=0,
# #         dim_target_input_layers=200,
# #         start_input_layer_to_dim_regularize: int = 0,
# #         end_input_layer_to_dim_regularize: int = 0,
# #         dim_target_output_layers=10,
# #         start_output_layer_to_dim_regularize: int = 0,
# #         end_output_layer_to_dim_regularize: int = 0,
# #         cca_regularization_weight=0,
# #         cca_target=30,
# #         num_batches_per_actitity_regularization: int = 1,
# #         num_samples_activity_regularizer: int = 600,
# #         freeze_input: bool = False,
# #         input_style: str = 'hypercube',
# #         saves_per_epoch: int = 1,
# #         # batch_normalization: bool = True,
# #         rerun=False,
# #         load_prev_model: bool = True,
# #                     pretrain: bool=False):
# #     """activity_l1_regularization_weight, activity_l2_regularization_weight, arg_dict,
# #                     cca_regularization_weight, dataloaders, datasets, device, input_dim_regularization_weight,
# #                     learning_patience, learning_rate, loss, model, momentum, n_hold, n_lag, n_out, network,
# optimizer,
# #                     output_dim_regularization_weight, param_l1_regularization_weight,
# param_l2_regularization_weight,
# #                     rerun, run_dir, save_model_criterion, scheduler, scheduler_factor, total_num_epochs, zero_fun"""
# #
# #     # zero_fun, total_num_epochs
# #
# #     ## Initializing regularizers for the training stage
# #     if param_l1_regularization_weight > 0:
# #         def param_l1_regularization_f():
# #             return param_l1_regularization_weight*l1_regularizer(model.params())
# #     else:
# #         param_l1_regularization_f = zero_fun
# #     if param_l2_regularization_weight > 0:
# #         def param_l2_regularization_f():
# #             return param_l2_regularization_weight*l2_regularizer(model.params())
# #     else:
# #         param_l2_regularization_f = zero_fun
# #     if activity_l1_regularization_weight > 0:
# #         def activity_l1_regularization_f():
# #             return activity_l1_regularization_weight*l1_activity_regularizer()
# #     else:
# #         activity_l1_regularization_f = zero_fun
# #     if activity_l2_regularization_weight > 0:
# #         def activity_l2_regularization_f():
# #             return activity_l2_regularization_weight*l2_activity_regularizer()
# #     else:
# #         activity_l2_regularization_f = zero_fun
# #     if input_dim_regularization_weight > 0:
# #         def input_dim_regularization_f():
# #             return input_dim_regularization_weight*effective_dim_regularizer(dim_target_input_layers,
# input_layer_idx)
# #     else:
# #         input_dim_regularization_f = zero_fun
# #     if output_dim_regularization_weight > 0:
# #         def output_dim_regularization_f():
# #             return output_dim_regularization_weight*effective_dim_regularizer(dim_target_output_layers,
# #                                                                               output_layer_idx)
# #     else:
# #         output_dim_regularization_f = zero_fun
# #     if cca_regularization_weight > 0:
# #         def cca_regularization_f():
# #             return cca_regularization_weight*cca_regularizer(cca_target, output_layer_idx)
# #     else:
# #         cca_regularization_f = zero_fun
# #     if network != 'feedforward':
# #         loss_points = torch.arange(n_lag - n_out, n_lag + n_hold - 1)
# #     else:
# #         loss_points = torch.tensor([0], dtype=int)
# #     # num_train = int(round((1 - perc_val)*num_train_samples_per_epoch))
# #     ## Initializing loss functions for the training stage
# #     if loss in ('categorical_crossentropy', 'cce'):
# #         loss_CEL = torch.nn.CrossEntropyLoss()
# #         if network == 'feedforward':
# #             loss_function = loss_CEL
# #         else:
# #             def loss_function(output, label):
# #                 return loss_CEL(output[:, loss_points].transpose(1, 2), label[:, loss_points])
# #     elif loss in ('mean_square_error', 'mse'):
# #         criterion_mse = torch.nn.MSELoss()
# #         def criterion_single_timepoint(output, label):  # The output does not have a time dimension
# #             label_onehot = torch.zeros(label.shape[0], num_classes)
# #             for i0 in range(num_classes):
# #                 label_onehot[label == i0, i0] = 1
# #             return criterion_mse(output, .7*label_onehot)
# #         def loss_function(output, label):
# #             cum_loss = 0
# #             for i0 in loss_points:
# #                 cum_loss += criterion_single_timepoint(output[:, i0], label[:, i0])
# #             cum_loss = cum_loss/m
# #             return cum_loss
# #     elif loss == 'zero':
# #         def loss_function(output, label):
# #             return 0
# #     else:
# #         raise AttributeError("loss option not recognized.")
# #     def regularized_loss(output, label):
# #         crit = loss_function(output, label)
# #         if activity_regularizer_counter():
# #             act_reg = (activity_l1_regularization_f() + activity_l2_regularization_f() +
# input_dim_regularization_f() +
# #                        output_dim_regularization_f() + cca_regularization_f())
# #         else:
# #             act_reg = 0
# #         param_reg = param_l1_regularization_f() + param_l2_regularization_f()
# #         return crit + act_reg + param_reg
# #     ## Initialize optimizer and learning scheduler for training stage
# #     if optimizer == 'sgd':
# #         # optimizer_instance = torch.optim.SGD(net_par.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# #         optimizer_instance = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
# #                                              lr=learning_rate, momentum=momentum)
# #     elif optimizer == 'rmsprop':
# #         # noinspection PyUnresolvedReferences
# #         optimizer_instance = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
# #                                                  lr=learning_rate, momentum=momentum)
# #     else:
# #         raise AttributeError('optimizer option not recognized.')
# #     if scheduler == 'plateau':
# #         learning_scheduler_torch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_instance,
# #                                                                               factor=scheduler_factor,
# #                                                                               patience=learning_patience,
# #                                                                               threshold=1e-7,
# #                                                                               threshold_mode='abs',
# #                                                                               min_lr=0,
# #                                                                               verbose=True)
# #     elif scheduler == 'steplr':
# #         learning_scheduler_torch = torch.optim.lr_scheduler.StepLR(optimizer_instance, step_size=learning_patience,
# #                                                                    gamma=scheduler_factor)
# #     elif scheduler == 'multisteplr':
# #         learning_scheduler_torch = torch.optim.lr_scheduler.MultiStepLR(optimizer_instance,
# #                                                                         learning_patience,
# #                                                                         scheduler_factor)
# #     else:
# #         raise AttributeError('scheduler option not recognized.')
# #     learning_scheduler_instance = model_trainer.DefaultLearningScheduler(learning_scheduler_torch)
# #     # stats_trackers = {x: model_trainer.DefaultStatsTracker(batches_per_epoch[x], x, accuracy=False) for x in
# #     #                   ('train', 'val')}
# #     print('\n==> Training network')
# #     # import ipdb; ipdb.set_trace()
# #     model_trainer.train_model(model, dataloaders, device[0], regularized_loss,
# #                               optimizer_instance, total_num_epochs, run_dir,
# #                               not rerun,
# #                               learning_scheduler=learning_scheduler_instance,
# #                               save_model_criterion=save_model_criterion)
# #     # stats_history = history_and_machinery['stats_history']
# #     # stats_trackers = history_and_machinery['stats_trackers']
# #     # learning_scheduler_instance = history_and_machinery['learning_scheduler']
# #     # optimizer_instance = history_and_machinery['optimizer']
# #     # params = dict(dataloaders=dataloaders, datasets=datasets,
# #     #               # stats_trackers=stats_trackers,
# #     #               learning_scheduler_instance=learning_scheduler_instance,
# #     #               optimizer_instance=optimizer_instance)
# #     # params.update(arg_dict)
# #     # return params