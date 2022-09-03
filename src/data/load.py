from argparse import ArgumentParser
import os

import pandas as pd

from .utils import print_info

NUM_SEP = 10


def add_load_specific_args(parent_parser, data_type):
    if data_type == "csv":
        parser = CSVLoader.add_load_specific_args(parent_parser)
        return parser
    else:
        raise ValueError


class Loader:
    def load(self):
        data_df = self.load_data()
        self.log(data_df)

        return data_df

    @staticmethod
    def log(data_df):
        print("-" * NUM_SEP)
        print("Load Logging...")
        print_info(data_df)


class CSVLoader(Loader):
    def __init__(self, data_file, data_dir, **kwargs):
        self.data_file = data_file
        self.data_dir = data_dir

    def load_data(self):
        data_df = pd.read_csv(self.data_file)

        if self.data_dir:
            data_df["image"] = [
                os.path.join(self.data_dir, im) for im in data_df["image"]
            ]

        return data_df

    @staticmethod
    def add_load_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, allow_abbrev=False
        )
        parser.add_argument("--data-file", required=True)
        parser.add_argument("--data-dir", required=False)

        return parser
