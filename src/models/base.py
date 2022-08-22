from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from src.metrics import (
    avg_pair_score_distances,
    get_topk_acc,
    validation_stats,
)
from torch.optim.lr_scheduler import StepLR


class ResNet50EmbeddingModule(pl.LightningModule):
    def __init__(self, embedding_dim, lr, wd, fixbase_epoch, name, **kwargs):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.embedding_dim)

        self.lr = lr
        self.wd = wd

        self.fixbase_epoch = fixbase_epoch

        self.name = name

    def forward(self, batch):
        x, _, annot, names = batch

        x = self.model(x)

        x = F.normalize(x, p=2, dim=1)

        return {"embeddings": x, "names": names, "annots": annot}

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch

        x = self.model(x)

        simmat = self._get_simmat(x)
        pos, neg = avg_pair_score_distances(simmat, y)

        loss = self.criterion(x, y)

        self.log("train_pos_score", pos)
        self.log("train_neg_score", neg)
        self.log("train_dist", pos - neg, prog_bar=True)
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _, y = batch

        x = self.model(x)

        loss = self.criterion(x, y)

        return {"embeddings": x, "labels": y, "loss": loss}

    def validation_epoch_end(self, val_outs):
        batch_embeddings = [out["embeddings"] for out in val_outs]
        batch_labels = [out["labels"] for out in val_outs]
        batch_losses = [out["loss"].view(1, 1) for out in val_outs]

        embeddings = torch.cat(batch_embeddings)
        labels = torch.cat(batch_labels)
        losses = torch.cat(batch_losses)

        loss = losses.mean()

        simmat = self._get_simmat(embeddings)

        pos, neg = avg_pair_score_distances(simmat, labels)
        self.log("val_pos_score", pos)
        self.log("val_neg_score", neg)
        self.log("val_dist", pos - neg)
        self.log("val_loss", loss, prog_bar=True)

        (top1, top5, top10), m_a_p = get_topk_acc(simmat, labels)
        roc_auc, pr_auc = validation_stats(simmat, labels)

        self.log("top1", top1, prog_bar=True)
        self.log("top5", top5)
        self.log("top10", top10)
        self.log("mAP@R", m_a_p)
        self.log("roc_auc", roc_auc)
        self.log("pr_auc", pr_auc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, _, _, y = batch

        x = self.model(x)

        return {"embeddings": x, "labels": y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": StepLR(optimizer, 120),
        }

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                monitor="top1",
                mode="max",
                filename="{epoch}-{top1:.2f}",
                every_n_val_epochs=2,
            ),
            FixBase(self.fixbase_epoch),
        ]

    def _get_simmat(self, embeddings):
        """
        Must return in the range [0, 1] where 1 is the maximum similarity.
        """
        ...

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, allow_abbrev=False
        )
        parser.add_argument("--embedding-dim", default=512, type=int)
        parser.add_argument("--lr", default=1e-5, type=float)
        parser.add_argument("--wd", default=5e-4, type=float)
        parser.add_argument("--fixbase-epoch", default=1, type=int)

        return parser


class FixBase(Callback):
    def __init__(self, fixbase_epoch):
        self.fixbase_epoch = fixbase_epoch

    def on_train_start(self, trainer, pl_module):
        if self.fixbase_epoch != 0 and pl_module.current_epoch == 0:
            print("\nFreezing backbone weights excluding fc...")
            for name, param in pl_module.model.named_parameters():
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = False

    def on_train_epoch_end(self, trainer, pl_module):
        if (self.fixbase_epoch - 1) == pl_module.current_epoch:
            print("\nUnfreezing backbone weights excluding fc...")
            for name, param in pl_module.model.named_parameters():
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = True
