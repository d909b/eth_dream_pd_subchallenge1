"""
train_dnn.py - Training deep neural network models for classifying Parkinson's from mobile phone data.

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

import sys
import numpy as np
from os.path import join
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from dream_parkinsons.models.model_builder import ModelBuilder
from dream_parkinsons.apps.evaluate import EvaluationApplication


def make_keras_generator(args, wrapped_generator, num_steps, batch_size=32):
    num_steps = num_steps // batch_size

    def generator():
        while True:
            batch_x, batch_y = [], []
            age_batch, gender_batch = [], []
            for _ in range(batch_size):
                x, y = None, None
                age, gender = None, None
                while x is None or y is None:
                    samples = next(wrapped_generator)
                    converted_samples = TrainDNNApplication.convert_inputs_train(samples[:-1], args["signal"])

                    if converted_samples is None:
                        continue

                    x, age, gender = converted_samples
                    y = samples[-1]

                batch_x.append(x)
                age_batch.append(age)
                gender_batch.append(gender)
                batch_y.append([y])

            # Pad samples to same length to be able to use them in a batch.
            # The padding will be masked by the model later and thus does not influence what is learned.
            batch_x = pad_sequences(batch_x,
                                    maxlen=np.max(map(lambda x: x.shape[-2], batch_x)),
                                    padding="post",
                                    dtype="float32")

            yield [batch_x, np.asarray(age_batch), np.asarray(gender_batch)], np.asarray(batch_y)
    return generator(), num_steps


class TrainDNNApplication(EvaluationApplication):
    def __init__(self):
        super(TrainDNNApplication, self).__init__()

    @staticmethod
    def convert_inputs_evaluate(inputs, signal):
        if signal == "outbound":
            indices = (0, 4)
        elif signal == "rest":
            indices = (5, 9)
        else:
            indices = (10, 14)

        input_list = inputs[indices[0]:indices[1]]
        if any(elem is None for elem in input_list):
            return None

        return_value = np.concatenate(input_list, axis=-1)
        return return_value

    @staticmethod
    def convert_inputs_train(inputs, signal):
        return_value = TrainDNNApplication.convert_inputs_evaluate(inputs, signal)

        if return_value is None:
            return None

        age = inputs[-2]
        gender = inputs[-1]

        age = age if age is not None else 0
        gender = 1 if gender == "Female" else 0

        age = age / 100.

        return return_value, age, gender

    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        print("INFO: Started training feature extraction.", file=sys.stderr)

        num_units = int(self.args["num_units"])
        num_epochs = int(self.args["num_epochs"])
        batch_size = int(self.args["batch_size"])
        dropout = float(self.args["dropout"])
        attention_dropout = float(self.args["attention_dropout"])

        model, penultimate_layer_model = ModelBuilder.build_model(input_shape=(None, 13),
                                                                  num_units=num_units,
                                                                  p_dropout=dropout,
                                                                  attention_dropout=attention_dropout)
        model.summary()

        best_model_path = join(self.args["output_directory"], "model.h5")
        penultimate_model_path = join(self.args["output_directory"], "penultimate.h5")

        callbacks = [
            EarlyStopping(patience=7),
            ModelCheckpoint(filepath=best_model_path,
                            save_best_only=True)
        ]

        train_generator, train_steps = make_keras_generator(self.args,
                                                            train_generator,
                                                            train_steps,
                                                            batch_size)

        val_generator, val_steps = make_keras_generator(self.args,
                                                        val_generator,
                                                        val_steps,
                                                        batch_size)

        assert train_steps > 0, "You specified a batch_size that is bigger than the employed size of the train set."
        assert val_steps > 0, "You specified a batch_size that is bigger than the employed size of the validation set."

        if self.args["load_existing"]:
            print("INFO: Loading weights from", self.args["load_existing"], file=sys.stderr)
            model.load_weights(self.args["load_existing"])

        if self.args["do_train"]:
            model.fit_generator(train_generator,
                                train_steps,
                                epochs=num_epochs,
                                validation_data=val_generator,
                                validation_steps=val_steps,
                                callbacks=callbacks)

            # Reset to the best model observed in training.
            model.load_weights(best_model_path)

            # Also save the penultimate layer model.
            penultimate_layer_model.save(penultimate_model_path)

        def extract_features(x):
            converted_input = TrainDNNApplication.convert_inputs_evaluate(x, self.args["signal"])

            if converted_input is None:
                return None

            converted_input = np.expand_dims(converted_input, axis=0)

            return penultimate_layer_model.predict(converted_input)

        return extract_features

if __name__ == "__main__":
    app = TrainDNNApplication()
    app.run()
