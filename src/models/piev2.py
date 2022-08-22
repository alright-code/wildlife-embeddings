import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import PIEv2Loss
from src.models.base import ResNet50EmbeddingModule


class PieV2Module(ResNet50EmbeddingModule):
    def __init__(self, margin, num_classes, weight_t, weight_x, **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.criterion = PIEv2Loss(
            margin, self.embedding_dim, num_classes, False, weight_t, weight_x
        )

    def forward(self, batch):
        x, _, annot, names = batch

        x = self.model(x)

        return {"embeddings": x, "names": names, "annots": annot}

    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch

        x = self.model(x)

        # We cannot compute loss on the validation set since it uses different individuals
        loss = torch.tensor(-1.0)

        return {"embeddings": x, "labels": y, "loss": loss}

    def _get_simmat(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)

        distmat = self.get_distmat(embeddings, embeddings)

        # [0, rad2]
        simmat = 1 - distmat / math.sqrt(2)

        return simmat

    @staticmethod
    def get_distmat(emb1, emb2):
        m, n = emb1.size(0), emb2.size(0)
        mat1 = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
        mat2 = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat = mat1 + mat2
        distmat.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
        distmat = distmat.clamp(min=1e-12).sqrt()

        return distmat

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ResNet50EmbeddingModule.add_model_specific_args(parent_parser)
        parser.add_argument("--margin", default=0.3, type=float)
        parser.add_argument("--weight-t", default=1.0, type=float)
        parser.add_argument("--weight-x", default=1.0, type=float)

        return parser
