"""
data_access.py - Methods for accessing a database of records from the mPower study.

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

import io
import sys
import sqlite3
import numpy as np
from os.path import join


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)


class DataAccess(object):
    DB_FILE_NAME = "parkinsons.db"

    TABLE_DEMOGRAPHICS = "demographics"
    TABLE_OUTBOUND = "outbound"
    TABLE_REST = "rest"
    TABLE_RETURN = "return"
    TABLE_RECORDS = "records"

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.db = None
        self.connect()
        self.setup_schema()

    def connect(self):
        self.db = sqlite3.connect(join(self.data_dir, DataAccess.DB_FILE_NAME),
                                  check_same_thread=False,
                                  detect_types=sqlite3.PARSE_DECLTYPES)

    def setup_schema(self):
        self.setup_demographics()
        self.setup_time_series(DataAccess.TABLE_OUTBOUND)
        self.setup_time_series(DataAccess.TABLE_REST)
        self.setup_time_series(DataAccess.TABLE_RETURN)
        self.setup_records()

        self.db.commit()

    def setup_records(self):
        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name}"
                        "("
                        "id TEXT NOT NULL PRIMARY KEY,"
                        "outbound_id INT,"
                        "rest_id INT,"
                        "return_id INT,"
                        "demographics_id TEXT NOT NULL,"
                        "FOREIGN KEY(demographics_id) REFERENCES {demographics_table_name}(id)"
                        ");"
                        .format(table_name=DataAccess.TABLE_RECORDS,
                                demographics_table_name=DataAccess.TABLE_DEMOGRAPHICS))

    def setup_time_series(self, name):
        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name} "
                        "("
                        "id TEXT NOT NULL,"
                        "timestamps ARRAY,"
                        "attitude ARRAY,"
                        "user_acceleration ARRAY,"
                        "rotation_rate ARRAY,"
                        "gravity ARRAY,"
                        "magnetic_field ARRAY,"
                        "demographics_id TEXT NOT NULL,"
                        "FOREIGN KEY(demographics_id) REFERENCES {demographics_table_name}(id)"
                        ");"
                        .format(table_name=name,
                                demographics_table_name=DataAccess.TABLE_DEMOGRAPHICS))

    def setup_demographics(self):
        self.db.execute("CREATE TABLE IF NOT EXISTS {table_name} "
                        "("
                        "id TEXT NOT NULL PRIMARY KEY,"
                        "created_on DATE,"
                        "app_version TEXT,"
                        "phone_info TEXT,"
                        "age INT,"
                        "is_caretaker INT,"
                        "deep_brain_stimulation INT,"
                        "diagnosis_year INT,"
                        "education TEXT,"
                        "employment TEXT,"
                        "gender TEXT,"
                        "health_history TEXT,"
                        "healthcare_provider TEXT,"
                        "home_usage INT,"
                        "last_smoked INT,"
                        "marital_status TEXT,"
                        "medical_usage INT,"
                        "marital_usage_yesterday TEXT,"
                        "medication_start_year INT,"
                        "onset_year INT,"
                        "packs_per_day INT,"
                        "past_participation INT,"
                        "phone_usage TEXT,"
                        "professional_diagnosis INT,"
                        "race TEXT,"
                        "smartphone TEXT,"
                        "smoked INT,"
                        "surgery INT,"
                        "video_usage INT,"
                        "years_smoking INT"
                        ");".format(table_name=DataAccess.TABLE_DEMOGRAPHICS,
                                    time_series_table_name=DataAccess.TABLE_OUTBOUND))

    def insert_many_demographics(self, values):
        self.db.executemany("INSERT INTO {table_name} VALUES ({question_marks});"
                            .format(table_name=DataAccess.TABLE_DEMOGRAPHICS,
                                    question_marks=",".join(["?"]*len(values[0]))),
                            values)

    def insert_many_time_series(self, table, values):
        if len(values) == 0:
            print("WARN: Tried to insert empty values.", file=sys.stderr)
            return

        self.db.executemany("INSERT INTO {table_name} VALUES ({question_marks});"
                            .format(table_name=table,
                                    question_marks=",".join(["?"]*len(values[0]))),
                            values)

    def insert_time_series(self, table, values):
        cursor = self.db.cursor()
        cursor.execute("INSERT INTO {table_name} VALUES ({question_marks});"
                       .format(table_name=table,
                               question_marks=",".join(["?"]*len(values))),
                       values)
        return cursor.lastrowid

    def get_num_rows(self, table_name):
        # NOTE: This query assumes that there has never been any deletions in the time series table.
        return self.db.execute("SELECT MAX(_ROWID_) FROM {} LIMIT 1;".format(table_name))\
                      .fetchone()[0]

    def get_record(self, rowid, with_demographics=True):
        if with_demographics:
            additional_columns = ",{demographics_table}.age," \
                                 " {demographics_table}.gender," \
                                 " {demographics_table}.professional_diagnosis "
            additional_tables = " INNER JOIN {demographics_table}" \
                                "  ON ({record_table}.outbound_id IS NOT NULL AND {demographics_table}.id = {outbound_table}.demographics_id) OR " \
                                "     ({record_table}.rest_id IS NOT NULL AND {demographics_table}.id = {rest_table}.demographics_id) OR " \
                                "     ({record_table}.return_id IS NOT NULL AND {demographics_table}.id = {return_table}.demographics_id) "
        else:
            additional_columns = ""
            additional_tables = ""

        return self.db.execute("SELECT "
                               " {outbound_table}.attitude,"
                               " {outbound_table}.user_acceleration,"
                               " {outbound_table}.rotation_rate,"
                               " {outbound_table}.gravity,"
                               " {outbound_table}.magnetic_field,"
                               " {rest_table}.attitude,"
                               " {rest_table}.user_acceleration,"
                               " {rest_table}.rotation_rate,"
                               " {rest_table}.gravity,"
                               " {rest_table}.magnetic_field,"
                               " {return_table}.attitude,"
                               " {return_table}.user_acceleration,"
                               " {return_table}.rotation_rate,"
                               " {return_table}.gravity,"
                               " {return_table}.magnetic_field,"
                               " {record_table}.id "
                               " {additional_columns} "
                               "FROM {record_table}"
                               " LEFT OUTER JOIN {outbound_table}"
                               "  ON {outbound_table}.rowid = {record_table}.outbound_id"
                               " LEFT OUTER JOIN {rest_table}"
                               "  ON {rest_table}.rowid = {record_table}.rest_id"
                               " LEFT OUTER JOIN {return_table}"
                               "  ON {return_table}.rowid = {record_table}.return_id "
                               " {additional_tables} "
                               "WHERE {record_table}.rowid = ?;"
                               .format(record_table=DataAccess.TABLE_RECORDS,
                                       outbound_table=DataAccess.TABLE_OUTBOUND,
                                       rest_table=DataAccess.TABLE_REST,
                                       return_table=DataAccess.TABLE_RETURN,
                                       additional_columns=additional_columns,
                                       additional_tables=additional_tables)
                               .format(demographics_table=DataAccess.TABLE_DEMOGRAPHICS,
                                       record_table=DataAccess.TABLE_RECORDS,
                                       outbound_table=DataAccess.TABLE_OUTBOUND,
                                       rest_table=DataAccess.TABLE_REST,
                                       return_table=DataAccess.TABLE_RETURN),
                               (rowid,)).fetchone()

    def get_row(self, table_name, id):
        query = "SELECT " \
                "timestamps, " \
                "attitude, " \
                "user_acceleration, " \
                "rotation_rate, " \
                "gravity, " \
                "magnetic_field," \
                "professional_diagnosis " \
                "FROM {table_name} " \
                "JOIN {demographics_table_name} " \
                "ON {table_name}.demographics_id = {demographics_table_name}.id " \
                "WHERE {table_name}.id = ?;".format(table_name=table_name,
                                                    demographics_table_name=DataAccess.TABLE_DEMOGRAPHICS)
        return self.db.execute(query, (id,)).fetchone()

    def insert_record(self, record):
        self.db.execute("INSERT INTO {records_table} VALUES (?, ?, ?, ?);"
                        .format(records_table=DataAccess.TABLE_RECORDS), record)

