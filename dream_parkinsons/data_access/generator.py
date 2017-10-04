"""
generator.py - Record-by-record generators for the mPower data.

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

import numpy as np
from dream_parkinsons.data_access.data_access import DataAccess


def random_cycle_generator(start_sample_idx, end_sample_idx):
    while 1:
        # Plus 1 because rowid indices start at 1, not 0.
        samples = np.random.permutation(np.arange(start_sample_idx, end_sample_idx) + 1)
        for sample in samples:
            yield sample


def make_generator(data_access, is_validation=False, validation_fraction=0.3, with_demographics=True):
    num_rows = data_access.get_num_rows(DataAccess.TABLE_RECORDS)
    cross_point = int(np.floor(num_rows * validation_fraction))

    # TODO: This way of splitting assumes random insertion order in the database, which does not hold.
    if is_validation:
        start_idx = 0
        end_idx = cross_point
    else:
        start_idx = cross_point
        end_idx = num_rows

    num_steps = end_idx - start_idx

    def generator():
        random = random_cycle_generator(start_idx, end_idx)
        while True:
            result = None
            while result is None:
                next_id = next(random)
                result = data_access.get_record(next_id, with_demographics)
            yield result

    return generator(), num_steps
