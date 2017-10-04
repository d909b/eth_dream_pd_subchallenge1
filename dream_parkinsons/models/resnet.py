"""
resnet.py - Construct generative and discriminative ResNet models.

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
from __future__ import division

from keras.models import Model
from keras import backend as K
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, UpSampling2D
from keras.layers import Input, Activation, Dense, Lambda, Dropout, LSTM, Bidirectional, Masking, concatenate
from dream_parkinsons.models.attention import SoftAttention


def build_bn_relu_block(input, with_bn):
    if with_bn:
        input = BatchNormalization(axis=1)(input)
    return Activation("relu")(input)


def build_initial_residual_block(input, dim):
    shortcut = AveragePooling2D((2, 1), data_format="channels_first")(input)
    shortcut = Conv2D(dim, kernel_size=1, padding="same", data_format="channels_first")(shortcut)

    output = input
    output = Conv2D(dim, kernel_size=3, padding="same",
                    data_format="channels_first", kernel_initializer="he_normal")(output)
    output = Activation("relu")(output)
    output = Conv2D(dim, kernel_size=3, padding="same",
                    data_format="channels_first", kernel_initializer="he_normal")(output)
    output = AveragePooling2D((2, 1), data_format="channels_first")(output)

    return add([shortcut, output])


def build_residual_block(input, dim, kernel_size, with_bn, resample=None):
    output = input
    output = build_bn_relu_block(output, with_bn)

    if resample == "up":
        output = UpSampling2D(2)(output)
        output = Conv2D(dim, kernel_size=kernel_size, data_format="channels_first", padding="same",
                        kernel_initializer="he_normal")(output)
    else:
        output = Conv2D(dim, kernel_size=kernel_size, data_format="channels_first", padding="same",
                        kernel_initializer="he_normal")(output)

    output = build_bn_relu_block(output, with_bn=with_bn)

    if resample == "down":
        output = Conv2D(dim, kernel_size=kernel_size, data_format="channels_first", padding="same",
                        kernel_initializer="he_normal")(output)
        output = AveragePooling2D((2, 1), data_format="channels_first")(output)
    else:
        output = Conv2D(dim, kernel_size=kernel_size, data_format="channels_first", padding="same",
                        kernel_initializer="he_normal")(output)

    input_shape = K.int_shape(input)
    output_shape = K.int_shape(output)

    if input_shape[-3] == output_shape[-3] and resample is None:
        shortcut = input
    else:
        if resample == "down":
            shortcut = Conv2D(output_shape[-3], data_format="channels_first", kernel_size=1, padding="same")(input)
            shortcut = AveragePooling2D((2, 1), data_format="channels_first")(shortcut)
        elif resample == "up":
            shortcut = UpSampling2D(2)(input)
            shortcut = Conv2D(output_shape[-1], data_format="channels_first", kernel_size=1, padding="same")(shortcut)
        else:
            shortcut = Conv2D(output_shape[-1], data_format="channels_first", kernel_size=1, padding="same")(input)

    return add([shortcut, output])


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, with_bn=True, dim=128, attention_dropout=0.2, p_dropout=0.4):
        input = Input(shape=input_shape)

        output = Masking()(input)
        output = Lambda(lambda x: K.expand_dims(x, axis=1), output_shape=(1,) + input_shape)(output)
        output = build_initial_residual_block(output, dim)  # Implicit "down"

        num_blocks = 5
        for _ in range(num_blocks):
            output = build_residual_block(output, dim, 3, with_bn, resample="down")

        output = build_bn_relu_block(output, with_bn=with_bn)
        output = Conv2D(1, kernel_size=3, data_format="channels_first",
                        padding="same", kernel_initializer="he_normal")(output)
        output = Lambda(lambda x: K.squeeze(x, 1), output_shape=input_shape)(output)
        output = Bidirectional(LSTM(dim, return_sequences=True))(output)
        output = SoftAttention(attention_dropout=attention_dropout)(output)
        output = Dense(dim)(output)
        output = build_bn_relu_block(output, with_bn=with_bn)
        output = Dropout(p_dropout)(output)
        output = Dense(dim)(output)
        output = build_bn_relu_block(output, with_bn=with_bn)
        output = penultimate_layer = Dropout(p_dropout)(output)

        # We input age and gender at the penultimate layer to obtain a co-adapted representation.
        age_input = Input(shape=(1,))
        gender_input = Input(shape=(1,))

        output = concatenate([age_input, gender_input, output], axis=-1)
        output = Dense(num_outputs, activation="sigmoid", name="discriminator_output")(output)

        model = Model(inputs=[input, age_input, gender_input], outputs=output)

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        return model, Model(inputs=input, outputs=penultimate_layer)
