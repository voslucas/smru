# CPU based implementation of the SMRU 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils  
import warnings
import itertools
import numbers
import math

# This is a modified copy of the non cuRNN based rnn tools found in Pytorch 0.4.1 
# Used to implement SMRU as layer.


# Default SMRU cell (smru)
def SMRUCell1(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)

    gates = gi+gh
    k,f,a  = gates.chunk(3, 1)
    
    #Apply softmax to the gates
    tmp = torch.stack([k,f,a],dim=2)
    st = torch.softmax(tmp, dim=2) 
    ksm, fsm ,asm = torch.unbind(st, 2)

   
    hy = torch.mul(ksm, hidden) + asm
    return hy

# Remove Variant (smru-r)
def SMRUCell2(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)

    gates = gi+gh
    k,f,a  = gates.chunk(3, 1)
    
    #Apply softmax to the gates
    tmp = torch.stack([k,f,a],dim=2)
    st = torch.softmax(tmp, dim=2) 
    ksm, fsm ,asm = torch.unbind(st, 2)
    #Remove variant -fsm 
    hy = torch.mul(ksm, hidden) + asm - fsm
    return hy

# Softmax First + Remove Variant (smru-rs)
def SMRUCell3(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)

    ki,fi,ai = gi.chunk(3,1)
    kh,fh,ah = gh.chunk(3,1)

    #First apply softmax to the gates
    tmp = torch.stack([ki,fi,ai,kh,fh,ah],dim=2)
    st = torch.softmax(tmp, dim=2) 
    ksm, fsm ,asm , ksmh, fsmh ,asmh = torch.unbind(st, 2)

    #Then apply addition
    k = ksm+ksmh
    f = fsm+fsmh
    a = asm+asmh
    #Remove variant -f 
    hy = torch.mul(k, hidden) + a - f
    return hy

# Softmax First Variant (smru-s)
def SMRUCell4(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)

    ki,fi,ai = gi.chunk(3,1)
    kh,fh,ah = gh.chunk(3,1)

    #First apply softmax to the gates
    tmp = torch.stack([ki,fi,ai,kh,fh,ah],dim=2)
    st = torch.softmax(tmp, dim=2) 
    ksm, fsm ,asm , ksmh, fsmh ,asmh = torch.unbind(st, 2)
    
    #Then apply addition
    k = ksm+ksmh
    f = fsm+fsmh
    a = asm+asmh
    hy = torch.mul(k, hidden) + a 
    return hy



def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight, batch_sizes):
        assert(len(weight) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l], batch_sizes)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight, batch_sizes):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def variable_recurrent_factory(inner, reverse=False):
    if reverse:
        return VariableRecurrentReverse(inner)
    else:
        return VariableRecurrent(inner)


def VariableRecurrent(inner):
    def forward(input, hidden, weight, batch_sizes):

        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def VariableRecurrentReverse(inner):
    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for i in reversed(range(len(batch_sizes))):
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward


def AutogradRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False,
                dropout=0, train=True, bidirectional=False, variable_length=False,
                dropout_state=None, flat_weight=None):

    if mode == 'SMRU1':
        cell = SMRUCell1
    elif mode == 'SMRU2':
        cell = SMRUCell2
    elif mode == 'SMRU3':
        cell = SMRUCell3
    elif mode == 'SMRU4':
        cell = SMRUCell4
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    rec_factory = variable_recurrent_factory if variable_length else Recurrent

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM'),
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden, batch_sizes):
        if batch_first and not variable_length:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight, batch_sizes)

        if batch_first and not variable_length:
            output = output.transpose(0, 1)

        return output, nexth

    return forward

def BackendRNN(*args, **kwargs):

    def forward(input, *fargs, **fkwargs):
        # if cudnn.is_acceptable(input.data):
        #     func = CudnnRNN(*args, **kwargs)
        # else:
        func = AutogradRNN(*args, **kwargs)

        # Hack for the tracer that allows us to represent RNNs as single
        # nodes and export them to ONNX in this form
        # Check the first argument explicitly to reduce the overhead of creating
        # the lambda. We need special handling here because the forward()
        # function gets reconstructed each and every time when RNN() is invoked
        # and we don't want to pay the cost of decorator invocation
        # import torch
        # if torch._C._jit_is_tracing(input):
        #     import torch.onnx.symbolic
        #     sym = torch.onnx.symbolic.RNN_symbolic_builder(*args, **kwargs)
        #     cell_type = args[0]

        #     bound_symbolic = partial(torch.onnx.symbolic.rnn_trace_override_symbolic,
        #                              cell_type, func, sym)

        #     decorator = torch.onnx.symbolic_override_first_arg_based(bound_symbolic)
        #     func = decorator(func)

        return func(input, *fargs, **fkwargs)

    return forward



class SMRU(nn.Module):

    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, mode="SMRU-VERSION-MUST-BE-SPECIFIED", bmode="bz", wmode="nn" ):
        super(SMRU, self).__init__()


        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.bmode = bmode
        self.wmode = wmode
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        gate_size = 3 * hidden_size
    
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = nn.Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = nn.Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = nn.Parameter(torch.Tensor(gate_size))
                b_hh = nn.Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return
        
        # No SMRU implementation ... 
        return
        
    def _apply(self, fn):
        ret = super(SMRU, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        # stdv =  1 / math.sqrt(self.hidden_size)
        # for weight in self.parameters():
        #      weight.data.uniform_(-stdv, stdv)
        for name, weight in self.named_parameters():
            #Weights
            if name.startswith('weight'):
                if self.wmode == "xn":
                    nn.init.xavier_normal_(weight)
                elif self.wmode == "xu":
                    nn.init.xavier_uniform_(weight)
                elif self.wmode == "id":
                    step_ih = weight.size(1)
                    for i in range(0, weight.size(0), step_ih):
                        nn.init.eye_(weight.data[i:i+step_ih])
                else:
                    stdv =  1 / math.sqrt(self.hidden_size)
                    weight.data.uniform_(-stdv, stdv)
            #Bias
            if name.startswith('bias'):
                #bz  
                nn.init.zeros_(weight)
                if self.bmode == "bf":
                    nn.init.ones_(weight[self.hidden_size:self.hidden_size*2])
                elif self.bmode == "bk":
                    nn.init.ones_(weight[0:self.hidden_size])
                elif self.bmode == "ba":
                    nn.init.ones_(weight[self.hidden_size*2:self.hidden_size*3])
            

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        check_hidden_size(hidden, expected_hidden_size)


    def forward(self, input, hx=None):
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions,
                                 max_batch_size, self.hidden_size,
                                 requires_grad=False)

        self.check_forward_args(input, hx, batch_sizes)

        func = BackendRNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            dropout_state=None,
            variable_length=is_packed,
            flat_weight=None
        )
        output, hidden = func(input, self.all_weights, hx, batch_sizes)

        if is_packed:
            output = torch.nn.utils.rnn.PackedSequence(output, batch_sizes)
        return output, hidden

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(SMRU, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def _flat_weights(self):
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]