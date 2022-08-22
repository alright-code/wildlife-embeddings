from argparse import ArgumentParser

import numpy as np

from .utils import plot_sighting_hist, print_info, print_shared_info

NUM_SEP = 10


def add_split_specific_args(parent_parser, split_type):
    if split_type == "pie":
        parser = PIESplitter.add_split_specific_args(parent_parser)
    else:
        raise ValueError

    return parser


class Splitter:
    def __init__(self, random_seed, weights_save_path, **kwargs):
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed

        self.image_save_path = weights_save_path

    def split(self, data_df, include_all_train_encounter):
        train_df, val_df, test_df = self.split_data(data_df)

        if len(data_df["encounter"].unique()) > 1 and not include_all_train_encounter:
            train_df = train_df.groupby("encounter").first()

        train_df = train_df.reset_index()

        self.log(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def log(self, train_df, val_df, test_df):
        print("-" * NUM_SEP)
        print("Split Logging...")
        for data_df, name in zip([train_df, val_df, test_df], ["train", "val", "test"]):
            print("-" * (NUM_SEP // 2))
            print(f"{name}set info:")
            print_info(data_df)
            if name is not "train":
                print_shared_info(train_df, data_df)
        print("-" * NUM_SEP)

        plot_sighting_hist(train_df, test_df, self.image_save_path)

    @staticmethod
    def add_split_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, allow_abbrev=False
        )
        parser.add_argument("--random-seed", default=0, type=int)

        return parser


class PIESplitter(Splitter):
    def __init__(self, eval_cutoff, **kwargs):
        super().__init__(**kwargs)

        self.eval_cutoff = eval_cutoff

    def split_data(self, data_df):
        other_df = data_df.groupby("encounter").first()
        name_counts = other_df["name"].value_counts()

        train_names = name_counts[name_counts > self.eval_cutoff].index.tolist()
        eval_names = name_counts[name_counts <= self.eval_cutoff].index.tolist()

        train_df = data_df[data_df["name"].isin(train_names)]

        self.rng.shuffle(eval_names)
        val_names = eval_names[: (len(eval_names) // 2)]
        test_names = eval_names[(len(eval_names) // 2) :]

        val_df = data_df[data_df["name"].isin(val_names)]
        test_df = data_df[data_df["name"].isin(test_names)]

        # Only use one annotation per encounter for val/test.
        val_df = val_df.groupby("encounter").first()
        test_df = test_df.groupby("encounter").first()

        train_df = train_df.reset_index()
        val_df = val_df.reset_index()
        test_df = test_df.reset_index()

        return train_df, val_df, test_df

    @staticmethod
    def add_split_specific_args(parent_parser):
        parser = Splitter.add_split_specific_args(parent_parser)
        parser.add_argument("--eval-cutoff", default=5, type=int)

        return parser
