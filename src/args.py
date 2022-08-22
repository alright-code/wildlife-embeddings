from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl

from src.data.datamodule import DataModule
from src.data.load import add_load_specific_args
from src.data.split import add_split_specific_args
from src.models.get import add_model_specific_args


def handle_arguments():
    parser = ArgumentParser(description="Embedding Network", allow_abbrev=False)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--model-type", default="piev2", type=str)
    parser.add_argument("--data-type", default="csv", type=str)
    parser.add_argument("--split-type", default="pie", type=str)

    args = parser.parse_known_args()[0]

    parser = ArgumentParser(description="Embedding Network", allow_abbrev=False)
    parser = add_load_specific_args(parser, args.data_type)
    parser = add_split_specific_args(parser, args.split_type)
    parser = add_model_specific_args(parser, args.model_type)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # data args
    parser = DataModule.add_data_specific_args(parser)

    secondary_args = parser.parse_known_args()[0]

    args = Namespace(**vars(args), **vars(secondary_args))

    return args
