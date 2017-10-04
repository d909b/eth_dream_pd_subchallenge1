"""
apply_pca.py - Methods for calculating a PCA transformation on an extracted set of feature vectors.

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
from sklearn.decomposition import PCA
from dream_parkinsons.apps.parameters import parse_parameters, clip_percentage
from dream_parkinsons.data_access.data_access import DataAccess
from dream_parkinsons.data_access.generator import make_generator

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


def collect_data(generator, steps):
    record_ids = []
    for i in range(steps):
        if i % 500 == 0 and i != 0:
            print("INFO: Completed", i, "steps.", file=sys.stderr)

        sample = next(generator)
        record_id = sample[-4]
        record_ids.append(record_id)
    return record_ids


class ApplyPCAApplication(object):
    def __init__(self):
        self.args = parse_parameters()
        self.data_access = DataAccess(data_dir=self.args["dataset"])
        self.setup()

    def setup(self):
        np.random.seed(self.args["seed"])

    def run(self):
        validation_fraction = clip_percentage(self.args["validation_set_fraction"])

        val_generator, val_steps = make_generator(self.data_access,
                                                  is_validation=True,
                                                  validation_fraction=validation_fraction)

        record_ids = collect_data(val_generator, val_steps)

        pickle_path = join(self.args["output_directory"], "record_ids.pickle")
        pickle.dump(record_ids, open(pickle_path, "wb"), pickle.HIGHEST_PROTOCOL)

        submission_data = pd.read_csv(self.args["load_existing"], header=0, index_col=0)
        val_data = submission_data[submission_data.index.isin(set(record_ids))]
        val_data = val_data.values

        print("INFO: Fitting PCA with validation data.", file=sys.stderr)
        pca = PCA(n_components=0.98, whiten=True)
        pca.fit(val_data)

        print("INFO: Explained variance was", pca.explained_variance_ratio_, file=sys.stderr)

        print("INFO: Transforming submission data.", file=sys.stderr)
        all_data = submission_data.values
        all_data = pca.transform(all_data)

        new_submission_data = pd.DataFrame(data=all_data,
                                           index=submission_data.index.values,
                                           columns=["Feature" + str(i) for i in range(all_data.shape[-1])])
        new_submission_data.index.name = "recordId"

        save_file_path = join(self.args["output_directory"], "transformed_submission.csv")
        print("INFO: Saving transformed submission CSV to", save_file_path, file=sys.stderr)
        new_submission_data.to_csv(save_file_path)


if __name__ == "__main__":
    app = ApplyPCAApplication()
    app.run()
