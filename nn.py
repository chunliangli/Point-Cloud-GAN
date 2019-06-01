import math
import torch
import tensorflow as tf
import torch.nn as tnn
from collections import OrderedDict
import pdb


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Module(object):
    def __init__(self):
        self.training = True
        self._parameters = []
        self._modules = OrderedDict()

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def parameters(self):
        return self._parameters

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
            self._parameters += value._parameters
            super(Module, self).__setattr__(name, value)
        else:
            super(Module, self).__setattr__(name, value)

    def __repr__(self):
        my_line = self.__class__.__name__ + '('

        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)

        if len(child_lines) > 0:
            my_line += '\n  ' + '\n  '.join(child_lines) + '\n'

        my_line += ')'
        return my_line

    def __str__(self):
        return self.__repr__()

    def transfer_to_pytorch_model(self, sess, pth_model):
        for key, val in self._modules.items():
            #print(key, val)
            pth_submod = getattr(pth_model, key)
            if isinstance(pth_submod, tnn.Module): 
                val.transfer_to_pytorch_model(sess, pth_submod)
            else:
                raise ValueError('Models do not match.')

    def transfer_from_pytorch_model(self, sess, pth_model):
        for key, val in self._modules.items():
            #print(key, val)
            pth_submod = getattr(pth_model, key)
            if isinstance(pth_submod, tnn.Module): 
                val.transfer_from_pytorch_model(sess, pth_submod)
            else:
                raise ValueError('Models do not match.')

class Linear(Module):
    counter = 0

    def __init__(self, in_features, out_features, bias=True, initializer=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if initializer is None:
            stdv = 1. / math.sqrt(self.in_features)
            initializer=tf.random_uniform_initializer(minval=-stdv, maxval=stdv)

        self.scope = "Linear_{0}".format(Linear.counter)
        with tf.variable_scope(self.scope, initializer=initializer):
            self.weight = tf.get_variable('w', (in_features, out_features))
            self._parameters.append(self.weight)
            if bias:
                self.bias = tf.get_variable('b', (out_features,))
                self._parameters.append(self.bias)
            else:
                self.bias = None

        Linear.counter += 1

    def forward(self, x):
        with tf.variable_scope(self.scope):
            outputs = tf.tensordot(x, self.weight, [[-1],[0]])
            if self.bias is not None:
                outputs = tf.nn.bias_add(outputs, self.bias)
        return outputs

    def __repr__(self):
        return 'Linear(in_features={0}, out_features={1}, bias={2})'.format(self.in_features, self.out_features, self.bias is not None)

    def transfer_to_pytorch_model(self, sess, pth_model):
        if isinstance(pth_model, tnn.Linear):
            if pth_model.in_features == self.in_features and pth_model.out_features == self.out_features:
                tmp = sess.run(self.weight)
                pth_model.weight.data.copy_(torch.from_numpy(tmp.T))
                if self.bias is not None:
                    tmp = sess.run(self.bias)
                    pth_model.bias.data.copy_(torch.from_numpy(tmp))
            else:
                raise ValueError('Dimensions of Linear layer mismatch')

    def transfer_from_pytorch_model(self, sess, pth_model):
        if isinstance(pth_model, tnn.Linear):
            if pth_model.in_features == self.in_features and pth_model.out_features == self.out_features:
                tmp = pth.model.weight.data.numpy()
                self.weight.load(tmp, sess)
                if self.bias is not None:
                    tmp = pth.model.bias.data.numpy()
                    self.bias.load(tmp, sess)
            else:
                raise ValueError('Dimensions of Linear layer mismatch')


class ReLU(Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()

    def forward(self, x):
        return tf.nn.relu(x)

    def __repr__(self):
        return 'ReLU()'

    def transfer_to_pytorch_model(self, sess, pth_model):
        pass

    def transfer_from_pytorch_model(self, sess, pth_model):
        pass


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super(LeakyReLU, self).__init__()
        self.neg_slope = negative_slope

    def forward(self, x):
        return tf.nn.leaky_relu(x, self.neg_slope)

    def __repr__(self):
        return 'LeakyReLU({0})'.format(self.neg_slope)

    def transfer_to_pytorch_model(self, sess, pth_model):
        pass

    def transfer_from_pytorch_model(self, sess, pth_model):
        pass


class ELU(Module):
    def __init__(self, inplace=False):
        super(ELU, self).__init__()

    def forward(self, x):
        return tf.nn.elu(x)

    def __repr__(self):
        return 'ELU()'

    def transfer_to_pytorch_model(self, sess, pth_model):
        pass

    def transfer_from_pytorch_model(self, sess, pth_model):
        pass


class Softplus(Module):
    def __init__(self, inplace=False):
        super(Softplus, self).__init__()

    def forward(self, x):
        return tf.nn.softplus(x+1)-1

    def __repr__(self):
        return 'Softplus()'

    def transfer_to_pytorch_model(self, sess, pth_model):
        pass

    def transfer_from_pytorch_model(self, sess, pth_model):
        pass


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return tf.nn.tanh(x)

    def __repr__(self):
        return 'Tanh()'

    def transfer_to_pytorch_model(self, sess, pth_model):
        pass

    def transfer_from_pytorch_model(self, sess, pth_model):
        pass


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
            self._parameters += module._parameters

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
