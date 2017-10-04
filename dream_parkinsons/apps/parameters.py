"""
parameters.py

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

import os
from argparse import ArgumentParser, Action, ArgumentTypeError


class ReadableDir(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise ArgumentTypeError("readable_dir:{} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise ArgumentTypeError("readable_dir:{} is not a readable dir".format(prospective_dir))


def parse_parameters():
    parser = ArgumentParser(description="Parkinson's diagnosis from mobile phone data.")
    parser.add_argument("--dataset", action=ReadableDir, required=True,
                        help="Folder containing the data set to be loaded.")
    parser.add_argument("--test_dataset", default="/cluster/scratch/schwabpa/dream-test",
                        help="Folder containing the test data set to be loaded (for submission).")
    parser.add_argument("--supplemental_dataset", default="/cluster/scratch/schwabpa/dream-test",
                        help="Folder containing the supplemental data set to be loaded (for submission).")
    parser.add_argument("--missing_dataset", default="/cluster/scratch/schwabpa/dream-missing",
                        help="Folder containing the missing data set to be loaded (for submission).")
    parser.add_argument("--submission_template",
                        default="/Volumes/ssd2/dream_models/submission1/PDChallenge_SC1_SubmissionTemplate.csv",
                        help="Template CSV file for submission containing all record ids to be submitted.")
    parser.add_argument("--output_directory", default="./models",
                        help="Base directory of all output files.")
    parser.add_argument("--load_existing", default="",
                        help="Existing model to load.")
    parser.add_argument("--per_walk_models", default="./models/outbound.h5,./models/rest.h5,./models/return.h5",
                        help="Submodels to load per walk. In order outbound, rest and return. Separated by a comma.")
    parser.add_argument("--seed", type=int, default=909,
                        help="Seed for the random number generator.")
    parser.add_argument("--validation_set_fraction", default=0.3, type=float,
                        help="Fraction of dataset to use for validation.")
    parser.add_argument("--fraction_of_data_set", default=0.0025, type=float,
                        help="Fraction of dataset to use for training and validation.")
    parser.add_argument("--num_units", type=int, default=4,
                        help="Number of neurons to use in DNN layers.")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size to use for training.")
    parser.add_argument("--dropout", default=0.4, type=float,
                        help="Value of the dropout parameter used in training.")
    parser.add_argument("--attention_dropout", default=0.2, type=float,
                        help="Value of the attention dropout parameter used in training.")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Processes or threads to use for tasks that allow multithreading.")
    parser.add_argument("--signal", default="outbound",
                        help="The type of signal to train on. One of (outbound, rest, return).")
    parser.set_defaults(create_submission_file=False)
    parser.add_argument("--create_submission_file", dest='create_submission_file', action='store_true',
                        help="Whether or not to create a submission file.")
    parser.set_defaults(do_train=False)
    parser.add_argument("--do_train", dest='do_train', action='store_true',
                        help="Whether or not to train a model.")
    parser.set_defaults(do_evaluate=False)
    parser.add_argument("--do_evaluate", dest='do_evaluate', action='store_true',
                        help="Whether or not to evaluate a model.")

    return vars(parser.parse_args())


def clip_percentage(value):
    return max(0., min(1., float(value)))
