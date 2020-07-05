from typing import Union, List, Callable, Optional
import torch
from torch import Tensor
from torch import nn

class FeedForward(nn.Module):
    """
    Class for feedforward neural network model. Takes a list of pytorch tensors holding the weight initializations and
    ties these together into a trainable neural network.
    """

    def __init__(self, layer_weights: List[Tensor], biases: List[Tensor], nonlinearities: List[Callable]):
        """

        Parameters
        ----------
        layer_weights : List[Tensor]
            List of the layer initializations.
        biases : List[Tensor]
            List of the bias initializations.
        nonlinearities : List[Callable]
            List of the nonlinearities used in the layers.
        """
        super().__init__()
        self.layer_weights = nn.ParameterList([nn.Parameter(layer, requires_grad=True) for layer in layer_weights])
        self.biases = nn.ParameterList([nn.Parameter(bias, requires_grad=True) for bias in biases])
        self.nonlinearities = nonlinearities

    def forward(self, inputs: Tensor):
        hid = inputs
        for layer, nonlinearity, bias in zip(self.layer_weights, self.nonlinearities, self.biases):
            hid = nonlinearity(hid@layer + bias)
        return hid

    def get_pre_activations(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = inputs
            pre_activations = []
            for layer, nonlinearity, bias in zip(self.layer_weights, self.nonlinearities, self.biases):
                pre_activation = hid@layer + bias
                hid = nonlinearity(pre_activation)
                pre_activations.append(detacher(pre_activation))
            return pre_activations

    def get_post_activations(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = inputs
            post_activations = []
            for layer, nonlinearity, bias in zip(self.layer_weights, self.nonlinearities, self.biases):
                hid = nonlinearity(hid@layer + bias)
                post_activations.append(detacher(hid))
            return post_activations

    def get_activations(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = inputs
            activations = []
            for layer, nonlinearity, bias in zip(self.layer_weights, self.nonlinearities, self.biases):
                pre_activation = hid@layer + bias
                hid = nonlinearity(pre_activation)
                activations.append(detacher(pre_activation))
                activations.append(detacher(hid))
            return activations

# noinspection PyArgumentList
class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN). This is a "vanilla" implementation with the typical machine-learning style
    equations:

        h_{t+1} = nonlinearity(h_{t} @ recurrent_weights + recurrent_bias)    --  hidden unit update
    """

    def __init__(self, input_weights: Tensor, recurrent_weights: Tensor, output_weights: Tensor,
                 recurrent_bias: Tensor, output_bias: Tensor, nonlinearity: Optional[Union[str, Callable]],
                 hidden_unit_init: Optional[Union[str, Callable]] = None, train_input: bool = False,
                 train_recurrent: bool = True, train_output: bool = True, train_recurrent_bias: bool = True,
                 train_output_bias: bool = True, output_over_recurrent_time: bool = False):
        """
        Parameters
        ----------
        input_weights : Tensor
            Input weight initialization.
        recurrent_weights : Tensor
            Recurrent weight initialization.
        output_weights : Tensor
            Output weight initialization.
        recurrent_bias : Tensor
            Recurrent bias vector initialization.
        output_bias : Tensor
            Output bias vector initialization.
        nonlinearity : Optional[Union[str, Callable]]
            The nonlinearity to use for the hidden unit activation function.
        hidden_unit_init : Optional[Union[str, Callable]]
            Initial value for the hidden units. The network is set to this value at the beginning of every input
            batch. Todo: make it so the hidden state can carry over input batches.
        train_input : bool
            True: train the input weights, i.e. set requires_grad = True for the input weights. False: keep the input
            weights fixed to their initial value over training.
        train_recurrent : bool
            True: train the recurrent weights. False: keep the recurrent weights fixed to their initial value over training.
        train_output : bool
            True: train the output weights. False: keep the output weights fixed to their initial value over
            training.
        train_recurrent_bias : bool
            True: train the recurrent bias. False: keep the recurrent bias fixed to its initial value over training.
        train_output_bias : bool
            True: train the output bias. False: keep the output bias fixed to its initial value over training.
        output_over_recurrent_time : bool
            True: Return network output over the recurrent timesteps. False: Only return the network output at the
            last timestep.
        """

        super().__init__()
        if isinstance(nonlinearity, Callable):
            self.nonlinearity = nonlinearity
        elif isinstance(nonlinearity, str):
            if nonlinearity == 'tanh' or nonlinearity == 'Tanh':
                self.nonlinearity = torch.tanh
            elif nonlinearity == 'relu' or nonlinearity == 'ReLU':
                def relu(x):
                    return torch.clamp(x, min=0)

                self.nonlinearity = relu
            else:
                raise AttributeError("nonlinearity not recognized.")
        else:
            raise AttributeError("nonlinearity not recognized.")

        if hidden_unit_init is None:
            self.hidden_unit_init = torch.zeros(recurrent_weights.shape[0])
        elif isinstance(hidden_unit_init, Tensor):
            self.hidden_unit_init = hidden_unit_init.clone()
        else:
            raise AttributeError("hidden_unit_init option not recognized.")

        if train_input:
            self.Win = nn.Parameter(input_weights.clone(), requires_grad=True)
        else:
            self.Win = nn.Parameter(input_weights.clone(), requires_grad=False)

        if train_recurrent:
            self.Wrec = nn.Parameter(recurrent_weights.clone(), requires_grad=True)
        else:
            self.Wrec = nn.Parameter(recurrent_weights.clone(), requires_grad=False)

        if train_output:
            self.Wout = nn.Parameter(output_weights.clone(), requires_grad=True)
        else:
            self.Wout = nn.Parameter(output_weights.clone(), requires_grad=False)

        if train_recurrent_bias:
            self.brec = nn.Parameter(recurrent_bias.clone(), requires_grad=True)
        else:
            self.brec = nn.Parameter(recurrent_bias.clone(), requires_grad=False)

        if train_output_bias:
            self.bout = nn.Parameter(output_bias.clone(), requires_grad=True)
        else:
            self.bout = nn.Parameter(output_bias.clone(), requires_grad=False)

        self.Wrec_T = self.Wrec.T
        self.Win_T = self.Win.T
        self.Wout_T = self.Wout.T

        self.output_over_recurrent_time = output_over_recurrent_time

    def forward(self, inputs: Tensor):
        hid = self.hidden_unit_init
        if self.output_over_recurrent_time:
            out = torch.zeros(inputs.shape[0], inputs.shape[1], self.Wout_T.shape[-1])
            for i0 in range(inputs.shape[1]):
                preactivation = hid@self.Wrec_T + inputs[:, i0]@self.Win_T + self.brec
                hid = self.nonlinearity(preactivation)
                out[:, i0] = hid@self.Wout_T + self.bout
            return out
        else:
            for i0 in range(inputs.shape[1]):
                preactivation = hid@self.Wrec_T + inputs[:, i0]@self.Win_T + self.brec
                hid = self.nonlinearity(preactivation)
            out = hid@self.Wout_T + self.bout
            return out

    def get_pre_activations(self, inputs: Tensor):
        hid = self.hidden_unit_init
        preactivations = []
        for i0 in range(inputs.shape[1]):
            preactivation = hid@self.Wrec + inputs[:, i0]@self.Win_T + self.brec
            hid = self.nonlinearity(preactivation)
            preactivations.append(preactivation.detach())
        out = hid@self.Wout_T + self.bout
        preactivations.append(out.detach())
        return preactivations

    def get_post_activations(self, inputs: Tensor):
        hid = self.hidden_unit_init
        postactivations = []
        for i0 in range(inputs.shape[1]):
            preactivation = hid@self.Wrec_T + inputs[:, i0]@self.Win_T + self.brec
            hid = self.nonlinearity(preactivation)
            postactivations.append(hid.detach())
        out = hid@self.Wout_T + self.bout
        postactivations.append(out.detach())
        return postactivations

    def get_activations(self, inputs: Tensor):
        hid = self.hidden_unit_init
        activations = []
        for i0 in range(inputs.shape[1]):
            preactivation = hid@self.Wrec_T + inputs[:, i0]@self.Win_T + self.brec
            hid = self.nonlinearity(preactivation)
            activations.append(preactivation.detach())
            activations.append(hid.detach())
        out = hid@self.Wout_T + self.bout
        activations.append(out.detach())
        return activations

class SompolinskyRNN(RNN):
    """
    Recurrent Neural Network (RNN) with style dynamics as used in Sompolinsky et al. 1988:

    h' = -h + nonlinearity(h)@Wrec + input@Win + recurrent_bias.

    These are discretized via forward Euler method to get the update

    h_{t+1} = h_{t} + dt(-h_{t} + nonlinearity(h_{t}) @ Wrec + input_{t}@Win + recurrent_bias)

    Here h is like a current input (membrane potential) and nonlinearity(h_{t}) is like a "firing rate".
    """

    def __init__(self, input_weights: Tensor, recurrent_weights: Tensor, output_weights: Tensor,
                 recurrent_bias: Tensor, output_bias: Tensor, nonlinearity: Optional[Union[str, Callable]],
                 hidden_unit_init: Optional[Union[str, Tensor]] = None, train_input: bool = False,
                 train_recurrent: bool = True, train_output: bool = True, train_recurrent_bias: bool = True,
                 train_output_bias: bool = True, dt: float = .01, output_over_recurrent_time: bool = False):
        """
        Parameters
        ----------
        input_weights : Tensor
            Input weight initialization.
        recurrent_weights : Tensor
            Recurrent weight initialization.
        output_weights : Tensor
            Output weight initialization.
        recurrent_bias : Tensor
            Recurrent bias vector initialization.
        output_bias : Tensor
            Output bias vector initialization.
        nonlinearity : Optional[Union[str, Callable]]
            The nonlinearity to use for the hidden unit activation function.
        hidden_unit_init : Optional[Union[str, Callable]]
            Initial value for the hidden units. The network is set to this value at the beginning of every input
            batch. Todo: make it so the hidden state can carry over input batches.
        train_input : bool
            True: train the input weights, i.e. set requires_grad = True for the input weights. False: keep the input
            weights fixed to their initial value over training.
        train_recurrent : bool
            True: train the recurrent weights. False: keep the recurrent weights fixed to their initial value over training.
        train_output : bool
            True: train the output weights. False: keep the output weights fixed to their initial value over
            training.
        train_recurrent_bias : bool
            True: train the recurrent bias. False: keep the recurrent bias fixed to its initial value over training.
        train_output_bias : bool
            True: train the output bias. False: keep the output bias fixed to its initial value over training.
        output_over_recurrent_time : bool
            True: Return network output over the recurrent timesteps. False: Only return the network output at the
            last timestep.
        """
        super().__init__(input_weights, recurrent_weights, output_weights, recurrent_bias, output_bias, nonlinearity,
                         hidden_unit_init, train_input, train_recurrent, train_output, train_recurrent_bias,
                         train_output_bias, output_over_recurrent_time)
        self.dt = dt

    def forward(self, inputs: Tensor):
        hid = self.hidden_unit_init
        if self.output_over_recurrent_time:
            # out = [hid]
            out = torch.zeros(inputs.shape[0], inputs.shape[1], self.Wout.shape[-1])
            for i0 in range(inputs.shape[1]):
                hid = hid + self.dt*(self.nonlinearity(hid)@self.Wrec_T + inputs[:, i0]@self.Win_T + self.brec)
                out[:, i0] = hid@self.Wout_T + self.bout
            return out
        else:
            for i0 in range(inputs.shape[1]):
                hid = hid + self.dt*(self.nonlinearity(hid)@self.Wrec_T + inputs[:, i0]@self.Win_T + self.brec)
            out = hid@self.Wout_T + self.bout
            return out

    def get_currents(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = self.hidden_unit_init
            currents = []
            for i0 in range(inputs.shape[1]):
                hid = hid + self.dt*(self.nonlinearity(hid)@self.Wrec_T + inputs[:, i0]@self.Win_T + self.brec)
                currents.append(hid)
            return currents

    def get_firing_rates(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = self.hidden_unit_init
            firing_rates = []
            for i0 in range(inputs.shape[1]):
                hid = hid + self.dt*(self.nonlinearity(hid)@self.Wrec_T + inputs[:, i0]@self.Win_T + self.brec)
                firing_rates.append(self.nonlinearity(hid))
            return firing_rates

    def get_activations(self, inputs: Tensor, detach: bool = True):
        raise AttributeError("get_activations is not implemented for this model. Try get_currents or get_firing_rates.")

    def get_pre_activations(self, inputs: Tensor, detach: bool = True): # Alias for compatibility with other models
        return self.get_currents(inputs, detach)

    def get_post_activations(self, inputs: Tensor, detach: bool = True): # Alias for compatibility with other models
        return self.get_firing_rates(inputs, detach)
