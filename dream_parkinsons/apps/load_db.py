"""
load_db.py - Methods for loading the mPower study data into a SQLite database format.

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
import json
import numpy as np
import synapseclient
from synapseclient.exceptions import SynapseTimeoutError
from collections import OrderedDict
from dream_parkinsons.data_access.data_access import DataAccess

IS_TEST = False
DEMOGRAPHICS_TABLE_SYNAPSE_ID = "syn10146552"
WALKING_TABLE_SYNAPSE_ID = "syn10146553"
TEST_WALKING_TABLE_SYNAPSE_ID = "syn10733842"
SUPPLEMENTAL_TRAINING_SYNAPSE_ID = "syn10733835"
# TEST_WALKING_TABLE_SYNAPSE_ID = SUPPLEMENTAL_TRAINING_SYNAPSE_ID


# This script loads all of the DREAM dataset (demographics, walking and rest time series) into an SQLite database.
def run(argv):
    if len(argv) != 4:
        print("HELP: load_db.py (SYNAPSE-USER) (PASSWORD) (DATABASE-DIRECTORY)", file=sys.stderr)
        sys.exit(1)

    synapse_user = argv[1]
    synapse_password = argv[2]
    database_directory = argv[3]

    data_access = DataAccess(data_dir=database_directory)
    synapse_client = synapseclient.login(email=synapse_user, password=synapse_password, rememberMe=True)

    if not IS_TEST:
        print("INFO: Querying demographics table.", file=sys.stderr)
        demographics_table = synapse_client.tableQuery(
            "SELECT * "
            "FROM {table_name}"
            .format(table_name=DEMOGRAPHICS_TABLE_SYNAPSE_ID)
        )

        demographics = demographics_table.asDataFrame()
        health_code_list = ", ".join(repr(i) for i in demographics["healthCode"])

        print("INFO: Inserting", len(demographics), "demographics.", file=sys.stderr)
        with data_access.db:
            data_access.insert_many_demographics(map(lambda x: tuple(x[1:]), demographics.values))

        print("INFO: Querying walking table with", len(demographics), "healthcodes.", file=sys.stderr)
    else:
        print("INFO: Querying test walking table.", file=sys.stderr)

    synapse_id = WALKING_TABLE_SYNAPSE_ID if not IS_TEST else TEST_WALKING_TABLE_SYNAPSE_ID

    # The load limit defines the maximum number of files to load in one iteration.
    load_limit = 200

    source_table1, source_table2, source_table3 = "deviceMotion_walking_return.json.items", \
                                                  "deviceMotion_walking_outbound.json.items", \
                                                  "deviceMotion_walking_rest.json.items"

    if not IS_TEST:
        count = synapse_client.tableQuery("SELECT COUNT(*) "
                                          "FROM {table_name} "
                                          "WHERE healthCode IN ({health_code_list})"
                                          .format(table_name=synapse_id,
                                                  health_code_list=health_code_list)).asInteger()
    else:
        count = synapse_client.tableQuery("SELECT COUNT(*) "
                                          "FROM {table_name}"
                                          .format(table_name=synapse_id)).asInteger()

    print("INFO: Downloading", count, "records.", file=sys.stderr)

    return_files, outbound_files, rest_files = OrderedDict(), OrderedDict(), OrderedDict()
    health_codes, record_ids, return_ids, outbound_ids, rest_ids = [], [], [], [], []
    for i in range(9999):
        offset = i*load_limit

        if not IS_TEST:
            walking_table = synapse_client.tableQuery(
                ('SELECT "recordId", '
                 '"healthCode", '
                 '"{source_table1}", "{source_table2}", "{source_table3}" '
                 'FROM {table_name} '
                 'WHERE '
                 'healthCode IN ({health_code_list}) '
                 'LIMIT {limit}'
                 'OFFSET {offset}'
                 ).format(table_name=synapse_id,
                          source_table1=source_table1,
                          source_table2=source_table2,
                          source_table3=source_table3,
                          health_code_list=health_code_list,
                          limit=load_limit,
                          offset=offset))
        else:
            walking_table = synapse_client.tableQuery(
                ('SELECT "recordId", '
                 '"healthCode", '
                 '"{source_table1}", "{source_table2}", "{source_table3}" '
                 'FROM {table_name} '
                 'LIMIT {limit}'
                 'OFFSET {offset}'
                 ).format(table_name=synapse_id,
                          source_table1=source_table1,
                          source_table2=source_table2,
                          source_table3=source_table3,
                          limit=load_limit,
                          offset=offset))

        print("INFO: Loaded offset", offset, file=sys.stderr)

        walking = walking_table.asDataFrame()
        walking['idx'] = walking.index

        health_codes = health_codes + list(walking["healthCode"].values)
        record_ids = record_ids + list(walking["recordId"].values)
        return_ids = return_ids + list(walking[source_table1].values)
        outbound_ids = outbound_ids + list(walking[source_table2].values)
        rest_ids = rest_ids + list(walking[source_table3].values)

        # Bulk download walk JSON files containing sensor data.
        print("INFO: Downloading walking json files.", file=sys.stderr)

        # Retry indefinitely in case of timeout.
        while 1:
            try:
                return_json_files = synapse_client.downloadTableColumns(walking_table,
                                                                        source_table1)
                break
            except SynapseTimeoutError:
                continue

        while 1:
            try:
                outbound_json_files = synapse_client.downloadTableColumns(walking_table,
                                                                          source_table2)
                break
            except SynapseTimeoutError:
                continue

        while 1:
            try:
                rest_json_files = synapse_client.downloadTableColumns(walking_table,
                                                                      source_table3)
                break
            except SynapseTimeoutError:
                continue

        return_files.update(return_json_files)
        outbound_files.update(outbound_json_files)
        rest_files.update(rest_json_files)

        num_rows = walking.shape[0]

        if num_rows != load_limit:
            break

    print("INFO: Finished downloading walking json files.", file=sys.stderr)
    print("INFO: Loading json files into the database.", file=sys.stderr)

    assert len(health_codes) == len(record_ids)
    assert len(return_ids) == len(record_ids)
    assert len(outbound_ids) == len(record_ids)
    assert len(rest_ids) == len(record_ids)

    seen_record_ids = {}

    num_returns, num_outbounds, num_rests = 0, 0, 0
    return_entries, outbound_entries, rest_entries, record_entries = [], [], [], []
    for i, health_code, record_id, return_id, outbound_id, rest_id in zip(range(len(record_ids)),
                                                                          health_codes,
                                                                          record_ids,
                                                                          return_ids,
                                                                          outbound_ids,
                                                                          rest_ids):
        if record_id in seen_record_ids:
            print("WARN: Skipping a duplicate record_id (", record_id, ") at index", i, ".",
                  "Previous index was:", seen_record_ids[record_id], file=sys.stderr)
            continue
        else:
            seen_record_ids[record_id] = i

        def clean_id(id):
            try:
                return str(int(id))
            except ValueError:
                return None

        return_id = clean_id(return_id)
        outbound_id = clean_id(outbound_id)
        rest_id = clean_id(rest_id)

        def get_time_series(id, files):
            if id in files:
                this_file = files[id]
                time_series = load_time_series(this_file, record_id, health_code)
            else:
                time_series = None
            return time_series

        return_time_series = get_time_series(return_id, return_files)
        outbound_time_series = get_time_series(outbound_id, outbound_files)
        rest_time_series = get_time_series(rest_id, rest_files)

        if return_time_series is None:
            return_entry = None
        else:
            return_entries.append(return_time_series)
            num_returns += 1
            return_entry = num_returns

        if outbound_time_series is None:
            outbound_entry = None
        else:
            outbound_entries.append(outbound_time_series)
            num_outbounds += 1
            outbound_entry = num_outbounds

        if rest_time_series is None:
            rest_entry = None
        else:
            rest_entries.append(rest_time_series)
            num_rests += 1
            rest_entry = num_rests

        record_entries.append((record_id, outbound_entry, rest_entry, return_entry, health_code))

        if (i % load_limit == 0 and i != 0) or i == len(health_codes) - 1:
            print("INFO: Loaded entries up to", i, file=sys.stderr)
            with data_access.db:
                data_access.insert_many_time_series(DataAccess.TABLE_REST, rest_entries)
                data_access.insert_many_time_series(DataAccess.TABLE_OUTBOUND, outbound_entries)
                data_access.insert_many_time_series(DataAccess.TABLE_RETURN, return_entries)
                data_access.insert_many_time_series(DataAccess.TABLE_RECORDS, record_entries)
            return_entries, outbound_entries, rest_entries, record_entries = [], [], [], []


def load_time_series(file_name, record_id, health_code):
    get_vars = lambda item, vars: [item.get(var) for var in vars]

    with open(file_name) as json_data:
        data = json.load(json_data)

        if data is None:
            return None

        timestamps, attitude, rotation_rate, user_acceleration, gravity, magnetic_field = \
            [], [], [], [], [], []
        for item in data:
            timestamps.append(item.get("timestamp"))
            attitude.append(get_vars(item.get("attitude"), ["x", "y", "z", "w"]))
            rotation_rate.append(get_vars(item.get("rotationRate"), ["x", "y", "z"]))
            user_acceleration.append(get_vars(item.get("userAcceleration"), ["x", "y", "z"]))
            gravity.append(get_vars(item.get("gravity"), ["x", "y", "z"]))
            magnetic_field.append(get_vars(item.get("magneticField"), ["x", "y", "z", "accuracy"]))

        if len(timestamps) == 0:
            return None

    return record_id, np.asarray(timestamps), np.asarray(attitude), np.asarray(rotation_rate), \
           np.asarray(user_acceleration), np.asarray(gravity), np.asarray(magnetic_field), \
           health_code


if __name__ == "__main__":
    run(sys.argv)
