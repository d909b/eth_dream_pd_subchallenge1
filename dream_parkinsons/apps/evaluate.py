"""
evaluate.py - Methods for evaluating features for the DREAM Parkinson's challenge.

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
import pandas as pd
from os.path import join
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from dream_parkinsons.apps.parameters import parse_parameters, clip_percentage
from dream_parkinsons.data_access.data_access import DataAccess
from dream_parkinsons.data_access.generator import make_generator

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle

# Number of iterations to train the ensemble models that require multiple iterations.
N_ITERS_MLP = 60
N_ITERS_SGD = 60


def collect_data(generator, steps, model, has_demographics=True):
    x, y = [], []
    for i in range(steps):
        if i % 500 == 0 and i != 0:
            print("INFO: Completed", i, "steps.", file=sys.stderr)

        sample = next(generator)
        features = model(sample)

        if features is None:
            continue

        features = np.squeeze(np.asarray(features))

        if not has_demographics:
            x.append(features)
            y.append(sample[-1])
            continue

        age = sample[-3]
        gender = sample[-2]
        y_val = sample[-1]

        if sample[-1] is None:
            y_val = 0

        additional_data = np.asarray([age if age is not None else 0,
                                      1 if gender == "Female" else 0])

        if features.ndim == 0:
            features = np.asarray([features])

        x_val = np.concatenate([features, additional_data], axis=-1)
        x.append(x_val)
        y.append(y_val)
    x, y = np.asarray(x), np.asarray(y)
    return x, y


class EvaluationApplication(object):
    def __init__(self):
        self.args = parse_parameters()
        print("INFO: Args are:", self.args, file=sys.stderr)

        self.data_access = DataAccess(data_dir=self.args["dataset"])
        self.setup()

    def setup(self):
        np.random.seed(self.args["seed"])

    def store_cache(self):
        print("INFO: Nothing to store.", file=sys.stderr)

    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        print("INFO: Started training feature extraction.", file=sys.stderr)

        for _ in range(train_steps):
            outbound_attitude, \
            outbound_user_acceleration, \
            outbound_rotation_rate, \
            outbound_gravity, \
            outbound_magnetic_field, \
            rest_attitude, \
            rest_user_acceleration, \
            rest_rotation_rate, \
            rest_gravity, \
            rest_magnetic_field, \
            return_attitude, \
            return_user_acceleration, \
            return_rotation_rate, \
            return_gravity, \
            return_magnetic_field, \
            age, \
            gender, \
            professional_diagnosis = next(train_generator)

            # Train your model here. Above is the data you will get per record.
            # Note: Some values and/or walks may not be present for all records.

        num_features = 5

        # Your feature extractor is a function that takes as input all the above features in a tuple and returns
        # a numpy array of arbitrary size as a feature vector.
        def my_feature_extractor(sample):
            return np.random.random_sample(size=(num_features,))

        # Return your feature extractor and the number of features (length of feature vector) it extracts per record.
        return my_feature_extractor

    def evaluate_model(self, model, train_generator, train_steps, val_generator, val_steps):
        print("INFO: Started evaluation.", file=sys.stderr)

        validation_models = [MLPClassifier(max_iter=N_ITERS_MLP),
                             SGDClassifier(max_iter=N_ITERS_SGD, penalty="elasticnet"),
                             KNeighborsClassifier(),
                             svm.SVC(kernel="linear"),
                             RandomForestClassifier(random_state=999, n_estimators=48)]

        print("INFO: Collecting training data for", train_steps, "samples.", file=sys.stderr)
        x_train, y_train = collect_data(train_generator, train_steps, model)

        train_data_path = join(self.args["output_directory"], "train_data.pickle")
        print("INFO: Saving training data to", train_data_path, file=sys.stderr)
        pickle.dump((x_train, y_train), open(train_data_path, "wb"))

        print("INFO: Started training evaluation ensemble.", file=sys.stderr)
        for i, validation_model in enumerate(validation_models):
            num_iters = 1 if i > 1 else N_ITERS_MLP
            for _ in range(num_iters):
                validation_model.fit(np.squeeze(x_train), y_train)

        print("INFO: Collecting validation data for", val_steps, "samples.", file=sys.stderr)
        x_val, y_val = collect_data(val_generator, val_steps, model)

        val_data_path = join(self.args["output_directory"], "val_data.pickle")
        print("INFO: Saving validation data to", val_data_path, file=sys.stderr)
        pickle.dump((x_val, y_val), open(val_data_path, "wb"))

        self.store_cache()

        auc_scores = []
        for validation_model in validation_models:
            y_pred = validation_model.predict(np.squeeze(x_val))
            auc_score = roc_auc_score(y_val, y_pred)
            print("INFO: Evaluation score for model", validation_model,
                  "was:", roc_auc_score(y_val, y_pred),
                  file=sys.stderr)
            auc_scores.append(auc_score)

        # NOTE: This is not at all an accurate estimate for the challenge evaluation setting
        #       as predictions in the challenge are scored per health code.
        print("INFO: Average AUC was:", np.mean(np.asarray(auc_scores)),
              "(on", val_steps, "samples).", file=sys.stderr)

    def run(self):
        fraction_of_data_set = clip_percentage(self.args["fraction_of_data_set"])
        validation_fraction = clip_percentage(self.args["validation_set_fraction"])

        train_generator, train_steps = make_generator(self.data_access,
                                                      is_validation=False,
                                                      validation_fraction=validation_fraction)

        val_generator, val_steps = make_generator(self.data_access,
                                                  is_validation=True,
                                                  validation_fraction=validation_fraction)

        adjusted_train_steps = int(np.floor(fraction_of_data_set * train_steps))
        adjusted_val_steps = val_steps

        print("INFO: Built generators with", train_steps, "training samples and", val_steps, "validation samples.",
              "We are using", adjusted_train_steps, "/", train_steps, "for training and",
              adjusted_val_steps, "/", val_steps, "for validation.",
              file=sys.stderr)

        model = self.train_model(train_generator,
                                 adjusted_train_steps,
                                 val_generator,
                                 adjusted_val_steps)

        # Reset validation generator in case it was (partially) used in the training phase.
        val_generator, val_steps = make_generator(self.data_access,
                                                  is_validation=True,
                                                  validation_fraction=validation_fraction)

        if self.args["do_evaluate"]:
            self.evaluate_model(model,
                                train_generator, adjusted_train_steps,
                                val_generator, adjusted_val_steps)

        if self.args["create_submission_file"]:
            self.create_submission_file(model)

    def get_data_frame(self, model, data_folder):
        print("INFO: Started creating submission file for", data_folder, file=sys.stderr)

        data = DataAccess(data_dir=data_folder)
        generator, steps = make_generator(data, validation_fraction=0., with_demographics=False)
        x, record_ids = collect_data(generator, steps, model, has_demographics=False)

        num_features = x.shape[-1]

        df = pd.DataFrame(data=x,
                          index=record_ids,
                          columns=["Feature" + str(i) for i in range(num_features)])

        save_file_path = join(self.args["output_directory"], data_folder.replace("/", "_") + ".csv")
        df.to_csv(save_file_path)

        return df

    @staticmethod
    def list_duplicates(sequence):
        """
        Finds duplicated records in a list.
        """
        seen_samples = set()
        seen_add_cached = seen_samples.add
        duplicates = set(x for x in sequence if x in seen_samples or seen_add_cached(x))
        return list(duplicates)

    def create_submission_file(self, model):
        print("INFO: Started creating submission file.", file=sys.stderr)

        # The missing_dataset contains those recordIds which were not downloaded in the first call
        # to load_db.py but do appear in the provided submission template.
        miss_df, supp_df, test_df, train_df = self.get_data_frame(model, self.args["missing_dataset"]), \
                                              self.get_data_frame(model, self.args["supplemental_dataset"]), \
                                              self.get_data_frame(model, self.args["test_dataset"]), \
                                              self.get_data_frame(model, self.args["dataset"])

        df = train_df.append(test_df).append(supp_df).append(miss_df)

        # The submission file's first header column must be named "recordId".
        df.index.name = "recordId"

        submission_file_path = join(self.args["output_directory"], "submission.csv")
        df.to_csv(submission_file_path)

        print("INFO: Duplicated records were:", EvaluationApplication.list_duplicates(df.index.values),
              file=sys.stderr)

        submission_template = pd.read_csv(self.args["submission_template"], header=0, index_col=0)

        # Make sure all requested recordIds are in the submission file.
        missing = set(df.index.values) - set(submission_template.index.values)
        print("INFO: We are missing records for a full submission:", missing, file=sys.stderr)

        self.store_cache()

if __name__ == "__main__":
    app = EvaluationApplication()
    app.run()
