"""
train_merged.py - Training deep neural network models for classifying Parkinson's from mobile phone data.

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
from threading import Lock
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from dream_parkinsons.models.model_builder import ModelBuilder
from dream_parkinsons.apps.evaluate import EvaluationApplication
from dream_parkinsons.apps.train_dnn import TrainDNNApplication

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class ThreadsafeIterator:
    """
    Threadsafe iterator class to wrap our Keras generators which themselves call models.
    """
    def __init__(self, it):
        self.it = it
        self.lock = Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    def g(*a, **kw):
        g, steps = f(*a, **kw)
        return ThreadsafeIterator(g), steps
    return g


TF_GRAPH = None

@threadsafe_generator
def make_keras_generator(args, wrapped_generator, num_steps, outbound_model, rest_model, return_model, batch_size=32):
    num_steps = num_steps // batch_size

    def generator():
        while True:
            batch_x, batch_y = [[] for _ in range(6)], []
            age_batch, gender_batch = [], []

            global TF_GRAPH
            with TF_GRAPH.as_default():
                for _ in range(batch_size):
                    samples = next(wrapped_generator)
                    converted_samples = TrainMergedApplication.convert_inputs_train(samples[:-1],
                                                                                    outbound_model,
                                                                                    rest_model,
                                                                                    return_model)
                    for i in range(6):
                        batch_x[i].append(converted_samples[i])

                    age, gender = converted_samples[-2], converted_samples[-1]
                    y = samples[-1] if samples[-1] is not None else 0

                    age_batch.append(age)
                    gender_batch.append(gender)
                    batch_y.append([y])

            for i in range(len(batch_x)):
                batch_x[i] = np.asarray(batch_x[i])

            yield batch_x + [np.asarray(age_batch), np.asarray(gender_batch)], np.asarray(batch_y)
    return generator(), num_steps


class TrainMergedApplication(EvaluationApplication):
    CACHE_OUTBOUND, CACHE_RETURN, CACHE_REST = {}, {}, {}

    def __init__(self):
        super(TrainMergedApplication, self).__init__()

    def store_cache(self):
        try:
            cache_data_path = join(self.args["output_directory"], "cache_data.pickle")
            print("INFO: Saving cache data to", cache_data_path, file=sys.stderr)
            pickle.dump((TrainMergedApplication.CACHE_OUTBOUND,
                         TrainMergedApplication.CACHE_RETURN,
                         TrainMergedApplication.CACHE_REST),
                        open(cache_data_path, "wb"))
        except:
            print("ERROR: Caught exception in TrainMergedApplication.store_cache.", file=sys.stderr)

    @staticmethod
    def convert_inputs_evaluate(inputs, outbound_model, rest_model, return_model):
        outbound_value = TrainDNNApplication.convert_inputs_evaluate(inputs, "outbound")
        rest_value = TrainDNNApplication.convert_inputs_evaluate(inputs, "rest")
        return_value = TrainDNNApplication.convert_inputs_evaluate(inputs, "return")

        record_id = inputs[-1]

        # NOTE: We consider time series that are too short for our CNN architecture as missing.
        if outbound_value is not None and outbound_value.shape[-2] < 2**8:
            outbound_value = None

        if rest_value is not None and rest_value.shape[-2] < 2**8:
            rest_value = None

        if return_value is not None and return_value.shape[-2] < 2**8:
            return_value = None

        if outbound_value is None:
            outbound_indicator = 1
            outbound_value = np.zeros(outbound_model.output_shape[1:])
        else:
            outbound_indicator = 0
            if record_id in TrainMergedApplication.CACHE_OUTBOUND:
                outbound_value = TrainMergedApplication.CACHE_OUTBOUND[record_id]
            else:
                outbound_value = outbound_model.predict(np.expand_dims(outbound_value, axis=0))
                TrainMergedApplication.CACHE_OUTBOUND[record_id] = outbound_value

        if rest_value is None:
            rest_indicator = 1
            rest_value = np.zeros(rest_model.output_shape[1:])
        else:
            rest_indicator = 0
            if record_id in TrainMergedApplication.CACHE_REST:
                rest_value = TrainMergedApplication.CACHE_REST[record_id]
            else:
                rest_value = rest_model.predict(np.expand_dims(rest_value, axis=0))
                TrainMergedApplication.CACHE_REST[record_id] = rest_value

        if return_value is None:
            return_indicator = 1
            return_value = np.zeros(return_model.output_shape[1:])
        else:
            return_indicator = 0
            if record_id in TrainMergedApplication.CACHE_RETURN:
                return_value = TrainMergedApplication.CACHE_RETURN[record_id]
            else:
                return_value = return_model.predict(np.expand_dims(return_value, axis=0))
                TrainMergedApplication.CACHE_RETURN[record_id] = return_value

        return np.squeeze(outbound_value), outbound_indicator, \
               np.squeeze(rest_value), rest_indicator, \
               np.squeeze(return_value), return_indicator

    @staticmethod
    def convert_inputs_train(inputs, outbound_model, rest_model, return_model):
        age = inputs[-2]
        gender = inputs[-1]

        age = age if age is not None else 0
        gender = 1 if gender == "Female" else 0

        age = age / 100.

        return TrainMergedApplication.convert_inputs_evaluate(inputs[:-2], outbound_model, rest_model, return_model) + \
               (age, gender)

    def load_model(self, model_file, file_name="outbound.h5", with_architecture=False):
        if with_architecture:
            return None, load_model(model_file)

        model, penultimate_layer_model = ModelBuilder.build_model(input_shape=(None, 13),
                                                                  num_units=32)

        print("INFO: Loading weights of", model_file, file=sys.stderr)
        model.load_weights(model_file)

        model_path = join(self.args["output_directory"], file_name)
        model.save(model_path, overwrite=True)
        print("INFO: Saved used model to", model_path, file=sys.stderr)

        return model, penultimate_layer_model

    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        print("INFO: Started training feature extraction.", file=sys.stderr)

        num_units = int(self.args["num_units"])
        num_epochs = int(self.args["num_epochs"])
        num_workers = int(self.args["n_jobs"])
        batch_size = int(self.args["batch_size"])
        model_files = self.args["per_walk_models"].split(",")

        _, outbound_model = self.load_model(model_files[0], file_name="outbound.h5")
        _, rest_model = self.load_model(model_files[1], file_name="rest.h5")
        _, return_model = self.load_model(model_files[2], file_name="return.h5")

        # Re-create the same model another time for the validation set generator.
        # This is necessary to avoid threading issues when using one model in two generators with Keras.
        _, val_outbound_model = self.load_model(model_files[0], file_name="outbound.h5")
        _, val_rest_model = self.load_model(model_files[1], file_name="rest.h5")
        _, val_return_model = self.load_model(model_files[2], file_name="return.h5")

        best_model_path = join(self.args["output_directory"], "model.h5")
        penultimate_model_path = join(self.args["output_directory"], "penultimate.h5")

        missing_indicator_shape = (None, 1,)
        model, penultimate_layer_model = ModelBuilder.build_per_record_model(input_shapes=[outbound_model.output_shape,
                                                                                           missing_indicator_shape,
                                                                                           rest_model.output_shape,
                                                                                           missing_indicator_shape,
                                                                                           return_model.output_shape,
                                                                                           missing_indicator_shape],
                                                                             num_units=num_units)

        # This is necessary to avoid issues with multithreading, tensorflow and Keras.
        # Source: https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
        global TF_GRAPH
        TF_GRAPH = tf.get_default_graph()

        callbacks = [
            EarlyStopping(patience=14),
            ModelCheckpoint(filepath=best_model_path,
                            save_best_only=True)
        ]

        train_generator, train_steps = make_keras_generator(self.args,
                                                            train_generator,
                                                            train_steps,
                                                            outbound_model,
                                                            rest_model,
                                                            return_model,
                                                            batch_size)

        val_generator, val_steps = make_keras_generator(self.args,
                                                        val_generator,
                                                        val_steps,
                                                        val_outbound_model,
                                                        val_rest_model,
                                                        val_return_model,
                                                        batch_size)

        assert train_steps > 0, "You specified a batch_size that is bigger than the size of the train set."
        assert val_steps > 0, "You specified a batch_size that is bigger than the size of the validation set."

        if self.args["load_existing"]:
            model.load_weights(self.args["load_existing"])

        if self.args["do_train"]:
            model.fit_generator(train_generator,
                                train_steps,
                                epochs=num_epochs,
                                validation_data=val_generator,
                                validation_steps=val_steps,
                                callbacks=callbacks,
                                workers=num_workers)

            # Reset to the best model observed in training.
            model.load_weights(best_model_path)

            # Also save the penultimate layer model.
            penultimate_layer_model.save(penultimate_model_path)

        def extract_features(x):
            converted_input = TrainMergedApplication.convert_inputs_evaluate(x,
                                                                             outbound_model,
                                                                             rest_model,
                                                                             return_model)

            batch_x = [[] for _ in range(6)]
            for i in range(6):
                batch_x[i].append(converted_input[i])

            for i in range(len(batch_x)):
                batch_x[i] = np.asarray(batch_x[i])

            return penultimate_layer_model.predict(batch_x)

        return extract_features

if __name__ == "__main__":
    app = TrainMergedApplication()
    app.run()
