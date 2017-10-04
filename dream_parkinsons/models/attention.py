"""
attention.py - A neural soft attention layer.
                Following:
                Yang, Z., Yang, D., Dyer, C., He, X., Smola, A. J., & Hovy, E. H. (2016).
                Hierarchical Attention Networks for Document Classification. In HLT-NAACL (pp. 1480-1489).

Copyright (C) 2017  Patrick Schwab, ETH Zurich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import keras.backend as K
from keras.layers import Layer
from keras import initializers, activations, constraints, regularizers


class SoftAttention(Layer):
    """
    Soft attention layer for recurrent neural networks.

    Following the method from:
    Yang, Z., Yang, D., Dyer, C., He, X., Smola, A. J., & Hovy, E. H. (2016).
    Hierarchical Attention Networks for Document Classification. In HLT-NAACL (pp. 1480-1489).
    """

    def __init__(self,
                 activation='tanh',
                 use_bias=True,
                 w_initializer='glorot_uniform',
                 u_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 w_regularizer=None,
                 u_regularizer=None,
                 bias_regularizer=None,
                 w_constraint=None,
                 u_constraint=None,
                 bias_constraint=None,
                 attention_dropout=0.,
                 seed=909,
                 **kwargs):
        self.supports_masking = True

        self.w_initializer = initializers.get(w_initializer)
        self.u_initializer = initializers.get(u_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.w_constraint = constraints.get(w_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.w_regularizer = regularizers.get(w_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.attention_dropout = max(min(1., float(attention_dropout)), 0)
        self.seed = seed
        self.w = None
        self.context = None
        self.bias = None

        super(SoftAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'w_initializer': initializers.serialize(self.w_initializer),
                  'u_initializer': initializers.serialize(self.u_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'w_regularizer': regularizers.serialize(self.w_regularizer),
                  'u_regularizer': regularizers.serialize(self.u_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'w_constraint': constraints.serialize(self.w_constraint),
                  'u_constraint': constraints.serialize(self.u_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'attention_dropout': self.attention_dropout}
        return config

    def build(self, input_shape):
        self.w = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.w_initializer,
                                 name='{}_w'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        if self.use_bias:
            self.bias = self.add_weight((input_shape[-1],),
                                        initializer=self.bias_initializer,
                                        name='{}_bias'.format(self.name),
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.context = self.add_weight((input_shape[-1],),
                                       initializer=self.u_initializer,
                                       name='{}_context'.format(self.name),
                                       regularizer=self.u_regularizer,
                                       constraint=self.u_constraint)

        super(SoftAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None  # Any masking is removed at this layer.

    def call(self, input, input_mask=None, training=None):
        u_it = K.dot(input, self.w)
        if self.use_bias:
            u_it = K.bias_add(u_it, self.bias)
        u_it = self.activation(u_it)

        context_embedding = wrapped_dot(u_it, self.context)

        attention_weights = K.exp(context_embedding)
        if input_mask is not None:
            attention_weights *= K.cast(input_mask, K.floatx())

        if 0. < self.attention_dropout < 1.:
            def dropped_inputs():
                return K.dropout(attention_weights, self.attention_dropout, seed=self.seed)

            attention_weights = K.in_train_phase(dropped_inputs, attention_weights, training=training)

        # Add a small value to avoid division by zero if sum of weights is very small.
        attention_weights /= K.cast(K.sum(attention_weights, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        attention_weights = K.expand_dims(attention_weights)

        # Weight initial input by attention.
        weighted_input = input * attention_weights
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        # The temporal dimension (-2) is collapsed when using attention.
        return tuple(shape for i, shape in enumerate(input_shape) if i != len(input_shape) - 2)

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


def wrapped_dot(x, y):
    if K.backend() == 'tensorflow':
        tmp_y = K.expand_dims(y)
        tmp_dot = K.dot(x, tmp_y)
        return K.squeeze(tmp_dot, axis=-1)
    else:
        return K.dot(x, y)