import math
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .df_dataset import DFDataset
from .load import CSVLoader
from .sampler import RandomCopiesIdentitySampler
from .split import PIESplitter
from .transforms import build_transform, inverse_normalize


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        split_type,
        data_type,
        train_transforms,
        eval_transforms,
        num_instances,
        num_copies,
        image_size,
        batch_size,
        num_workers,
        names_instead_of_labels=False,
        include_all_train_encounter=True,
        **kwargs,
    ):
        super().__init__()

        if data_type == "csv":
            self.loader = CSVLoader(**kwargs)
        else:
            raise ValueError

        if split_type == "pie":
            self.splitter = PIESplitter(**kwargs)
        else:
            raise ValueError

        self.train_transform = build_transform(train_transforms, image_size)
        self.eval_transform = build_transform(eval_transforms, image_size)

        self.num_instances = num_instances
        self.num_copies = num_copies

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.names_instead_of_labels = names_instead_of_labels
        self.include_all_train_encounter = include_all_train_encounter

    def get_sampler(self, dataset):
        labels = dataset.labels
        sampler = RandomCopiesIdentitySampler(
            labels, self.batch_size, self.num_instances, self.num_copies
        )
        return sampler

    def setup(self, stage=None):
        # Called on every process in DDP

        self.data_df = self.loader.load()
        self.train_df, self.val_df, self.test_df = self.splitter.split(
            self.data_df, self.include_all_train_encounter
        )

        self.train_dataset = DFDataset(
            self.train_df,
            self.train_transform,
        )
        self.val_dataset = DFDataset(
            self.val_df,
            self.eval_transform,
        )
        self.test_dataset = DFDataset(
            self.test_df,
            self.eval_transform,
        )

    def train_dataloader(self):
        sampler = self.get_sampler(self.train_dataset)

        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )
        eval_dataloader = DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return [train_dataloader, eval_dataloader]

    def get_annot_image(self, annot):
        if annot in self.train_df["annot"].unique():
            dataset = self.train_dataset
        elif annot in self.val_df["annot"].unique():
            dataset = self.val_dataset
        elif annot in self.test_df["annot"].unique():
            dataset = self.test_dataset
        else:
            raise RuntimeError(annot)

        idx = dataset.data.index[dataset.data["annot"] == annot].tolist()[0]
        annot_image, _, _, _ = dataset[idx]

        annot_image = inverse_normalize(annot_image)
        annot_image = (
            annot_image.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )

        return annot_image

    def get_aids(self, dataset):
        if dataset == "train":
            selected_dataset = self.train_dataset
        elif dataset == "val":
            selected_dataset = self.val_dataset
        elif dataset == "test":
            selected_dataset = self.test_dataset
        else:
            raise RuntimeError(dataset)

        annots = selected_dataset.data["annot"].to_list()

        return annots

    def get_names(self, dataset):
        if dataset == "train":
            selected_dataset = self.train_dataset
        elif dataset == "val":
            selected_dataset = self.val_dataset
        elif dataset == "test":
            selected_dataset = self.test_dataset
        else:
            raise RuntimeError(dataset)

        names = selected_dataset.data["name"].to_list()

        return names

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, allow_abbrev=False
        )
        parser.add_argument("--image-size", default=256, type=int)
        parser.add_argument(
            "--train-transforms",
            nargs="*",
            default=[
                "resize",
                "affine",
                "color_jitter",
                "grayscale",
                "blur",
                "center_crop",
                "normalize",
            ],
            type=str,
        )
        parser.add_argument(
            "--eval-transforms",
            nargs="*",
            default=["resize", "center_crop", "normalize"],
            type=str,
        )
        parser.add_argument("--num-copies", type=int, default=4)
        parser.add_argument("--num-instances", type=int, default=4)
        parser.add_argument("--batch-size", default=64, type=int)  # batch size per gpu
        parser.add_argument("--num-workers", default=8, type=int)

        return parser
