import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.args import handle_arguments
from src.data.datamodule import DataModule
from src.models.get import get_model


def main():
    args = handle_arguments()
    
    # Setup logging and weight saves.
    logger = TensorBoardLogger('tb-logs', name=args.name, version=args.version)
    args.logger = logger
    args.weights_save_path = os.path.join('tb-logs', args.name, f'version_{args.version}')
    
    dm = DataModule(**args.__dict__)
    dm.setup()
    num_classes = dm.train_dataset.num_labels

    model = get_model(args.model_type)(num_classes=num_classes, **args.__dict__)
    
    trainer = pl.Trainer.from_argparse_args(args)
    
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
    