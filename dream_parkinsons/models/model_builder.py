"""
model_builder.py - Build classification models for the DREAM Parkinson's challenge.

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

from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Dropout, concatenate
from dream_parkinsons.models.resnet import ResnetBuilder


class ModelBuilder(object):
    @staticmethod
    def build_model(input_shape, num_units=128, attention_dropout=0.2, p_dropout=0.4):
        model, penultimate_layer_model = ResnetBuilder.build(input_shape,
                                                             num_outputs=1,
                                                             with_bn=True,
                                                             dim=num_units,
                                                             attention_dropout=attention_dropout,
                                                             p_dropout=p_dropout)
        model.summary()
        return model, penultimate_layer_model

    @staticmethod
    def build_per_record_model(input_shapes, num_units=128, num_layers=3, p_dropout=0.25):
        input_layers = [Input(batch_shape=shape) for shape in input_shapes]

        last_layer = concatenate(input_layers, axis=-1)

        if num_layers >= 3:
            last_layer = Dense(80)(last_layer)
            last_layer = BatchNormalization()(last_layer)
            last_layer = LeakyReLU()(last_layer)
            last_layer = Dropout(p_dropout)(last_layer)

        if num_layers >= 2:
            last_layer = Dense(64)(last_layer)
            last_layer = BatchNormalization()(last_layer)
            last_layer = LeakyReLU()(last_layer)
            last_layer = Dropout(p_dropout)(last_layer)

        last_layer = Dense(num_units)(last_layer)
        last_layer = BatchNormalization()(last_layer)
        last_layer = last_penultimate_layer = LeakyReLU()(last_layer)
        last_layer = Dropout(p_dropout)(last_layer)

        # We input age and gender at the penultimate layer to obtain a co-adapted representation.
        age_input = Input(shape=(1,))
        gender_input = Input(shape=(1,))

        output_layer = concatenate([age_input, gender_input, last_layer], axis=-1)
        output_layer = Dense(1, activation="sigmoid")(output_layer)

        model = Model(inputs=input_layers + [age_input, gender_input], outputs=output_layer)
        penultimate_layer_model = Model(inputs=input_layers, outputs=last_penultimate_layer)
        model.summary()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        return model, penultimate_layer_model
