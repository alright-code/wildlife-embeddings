import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics


def avg_pair_score_distances(scores, labels):
    """Return the average distance between positive/negative pairs."""
    all_pairs = torch.combinations(torch.arange(len(scores)))
    pair_labels = labels[all_pairs]

    pos_mask = (pair_labels[:, 0] == pair_labels[:, 1]).bool()
    pair_distances = scores[all_pairs[:, 0], all_pairs[:, 1]]

    pos_distances = pair_distances[pos_mask]
    neg_distances = pair_distances[~pos_mask]

    return pos_distances.mean(), neg_distances.mean()


def validation_stats(scores, labels):
    all_pairs = torch.combinations(torch.arange(len(scores)))
    pair_labels = labels[all_pairs]

    pos_mask = (pair_labels[:, 0] == pair_labels[:, 1]).bool().cpu()
    pair_scores = scores[all_pairs[:, 0], all_pairs[:, 1]].cpu()

    if False:
        where_pos = torch.where(pos_mask)
        where_neg = torch.where(~pos_mask)
        pos_scores = pair_scores[where_pos]
        neg_scores = pair_scores[where_neg][: len(pos_scores)]
        pair_scores = torch.cat([pos_scores, neg_scores])
        pos_mask = torch.cat(
            [
                torch.ones(len(pos_scores), dtype=bool),
                torch.zeros(len(neg_scores), dtype=bool),
            ]
        )

    fpr, tpr, roc_thresh = metrics.roc_curve(pos_mask, pair_scores)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, pr_thresh = metrics.precision_recall_curve(pos_mask, pair_scores)
    pr_auc = metrics.auc(recall, precision)

    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]

    # preds = (pair_scores > optimal_threshold)
    # accuracy = metrics.accuracy_score(pos_mask, preds)
    # precision = metrics.precision_score(pos_mask, preds)
    # recall = metrics.recall_score(pos_mask, preds)

    if False:
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        plt.figure()
        plt.plot(
            precision,
            recall,
            color="darkorange",
            label="Precision/Recall curve (area = %0.2f)" % pr_auc,
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc, pr_auc


def get_topk_acc(scores, labels, ks=[1, 5, 10]):
    distmat = 1 - scores

    cmc, m_a_p = eval_onevsall(distmat.cpu().numpy(), labels.cpu().numpy(), max_rank=10)

    res = []
    for k in ks:
        res.append(cmc[k - 1])

    return res, m_a_p


def eval_onevsall(distmat, q_pids, max_rank=50):
    """Evaluation with one vs all on query set."""
    num_q = distmat.shape[0]

    if num_q < max_rank:
        max_rank = num_q
        print("Note: number of gallery samples is quite small, got {}".format(num_q))

    indices = np.argsort(distmat, axis=1)
    #    print('indices\n', indices)

    matches = (q_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    #    print('matches\n', matches)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query

    for q_idx in range(num_q):
        # remove the query itself
        order = indices[q_idx]
        keep = order != q_idx

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep
        ]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity has only one example
            # => cannot evaluate retrieval
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute mean average precision @ R
        # reference: https://arxiv.org/pdf/2003.08505.pdf
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        tmp_cmc = tmp_cmc[:num_rel]
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    # print('Computed metrics on {} examples'.format(len(all_cmc)))

    assert num_valid_q > 0, "Error: all query identities have one example"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
