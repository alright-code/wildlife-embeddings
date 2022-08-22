# Wildlife Embeddings
A pytorch lightning implementation of the WBIA Piev2 Plugin: https://github.com/WildMeOrg/wbia-plugin-pie-v2.

## Data Format
A csv file should be created for your dataset with the following columns:

- `annot`: unique integer identifier for each datapoint
- `image`: image name (not full path)
- `name`: individual name (incomparable sides of the same individual should have different names)
- `encounter`: annotations with the same encounter will not be compared during validation or testing. If unsure set equal to the `annot` column.
- `x`: left bounding box coordinate
- `y`: top bounding box coordinate
- `w`: bounding box width
- `h`: bounding box height
- `- theta`: rotation of the bounding box

## Training
During training we evaluate the model on the validation data every two epochs and retain the checkpoint with the highest 1-vs-all top1 accuracy.

### Arguments

- `--name`: name for logging
- `--version`: version number for logging

- `--data-file`: path to the data csv file
- `--data-dir`: path to the image directory
- `--eval-cutoff`: training is done with individuals with > eval-cutoff encounters, the rest are used for validation/testing

- `--image-size`: input image size
- `--train-transforms`: data augmentation for training, see `src/data/transforms.py`
- `--eval-transforms`: data augmentation for validation/testing
- `--num-copies`: see `src/data/sampler.py`
- `--num-instances`: see `src/data/sampler.py`
- `--batch-size`: batch size per gpu
- `--num-workers`: number of dataloader workers

- `--embedding-dim`: output embedding size
- `--lr`: learning rate
- `--wd`: weight decay
- `--fixbase-epoch`: freeze the weights of the model excluding fully-connected layers for this many epochs during training

- `--margin`: triplet loss margin
- `--weight-t`: triplet loss weight
- `--weight-x`: cross-entropy loss weight

- `--gpus`: number of gpus
- `--max_epochs`: maximum number of epochs

Also included are all the pytorch-lightning trainer flags: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer

