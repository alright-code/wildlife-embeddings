import argparse

import pytorch_lightning as pl
import torch

from src.data.datamodule import DataModule
from src.models.get import get_model


def main():
    parser = argparse.ArgumentParser(description="Embedding Network Validation")
    parser.add_argument("--chkpt", required=True)
    parser.add_argument("--data-dir", required=False)

    args = parser.parse_args()

    hyper_parameters = torch.load(args.chkpt)["hyper_parameters"]
    if args.data_dir:
        hyper_parameters["data_dir"] = args.data_dir
    dm = DataModule(**hyper_parameters)

    model = get_model(hyper_parameters["model_type"]).load_from_checkpoint(args.chkpt)

    args.gpus = 1
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
