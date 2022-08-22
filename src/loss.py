import torch
import torch.nn as nn

# Losses from https://github.com/WildMeOrg/wbia-plugin-pie-v2/tree/main/wbia_pie_v2/losses
class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.
    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    With label smoothing, the label :math:`y` for a class is computed by
    .. math::
        \begin{equation}
        (1 - \eps) \times y + \frac{\eps}{K},
        \end{equation}
    where :math:`K` denotes the number of classes and :math:`\eps` is a weight. When
    :math:`\eps = 0`, the loss function reduces to the normal cross entropy.
    Args:
        num_classes (int): number of classes.
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(
        self, embedding_dim, num_classes, eps=0.1, use_gpu=True, label_smooth=True
    ):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.classifier = nn.Linear(embedding_dim, num_classes)
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        logits = self.classifier(inputs)

        log_probs = self.logsoftmax(logits)
        zeros = torch.zeros(log_probs.size())
        targets = zeros.scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.eps) * targets + self.eps / self.num_classes
        return (-targets * log_probs).mean(0).sum()


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, similarity=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.similarity = similarity

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)
        m = inputs.size(0)
        mat1 = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(m, m)
        mat2 = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dist = mat1 + mat2
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class PIEv2Loss(nn.Module):
    def __init__(
        self, margin, embedding_dim, num_classes, similarity, weight_t=1, weight_x=1
    ):
        super().__init__()

        self.trip = TripletLoss(margin, similarity)
        self.ce = CrossEntropyLoss(embedding_dim, num_classes)

        self.weight_t = weight_t
        self.weight_x = weight_x

    def forward(self, x, labels):
        triplet_loss = self.trip(x, labels)
        cross_entropy_loss = self.ce(x, labels)

        return self.weight_t * triplet_loss + self.weight_x * cross_entropy_loss
