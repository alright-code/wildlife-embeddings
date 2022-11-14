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
- `theta`: rotation of the bounding box

## Training
During training we evaluate the model on the validation data every two epochs and retain the checkpoint with the highest 1-vs-all top1 accuracy.

### Arguments

- `--name`: name for logging
- `--version`: version number for logging

- `--data-file`: path to the data csv file
- `--data-dir`: path to the image directory
- `--eval-cutoff`: training is done with individuals with > eval-cutoff encounters, the rest are used for validation/testing

- `--image-size`: input image size (256)
- `--train-transforms`: data augmentation for training, see `src/data/transforms.py` (resize, affine, color_jitter, grayscale, blur, center_crop, normalize)
- `--eval-transforms`: data augmentation for validation/testing (resize, center_crop, normalize)
- `--num-copies`: see `src/data/sampler.py` (4)
- `--num-instances`: see `src/data/sampler.py` (4)
- `--batch-size`: batch size per gpu (64)
- `--num-workers`: number of dataloader workers (8)

- `--embedding-dim`: output embedding size (512)
- `--lr`: learning rate (1e-5)
- `--wd`: weight decay (5e-4)
- `--fixbase-epoch`: freeze the weights of the model excluding fully-connected layers for this many epochs during training (1)

- `--margin`: triplet loss margin (0.3)
- `--weight-t`: triplet loss weight (1.0)
- `--weight-x`: cross-entropy loss weight (1.0)

- `--gpus`: number of gpus 
- `--max_epochs`: maximum number of epochs

Also included are all the pytorch-lightning trainer flags: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer

