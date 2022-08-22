import copy
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


# https://github.com/WildMeOrg/wbia-plugin-pie-v2/blob/main/wbia_pie_v2/datasets/sampler.py
class RandomCopiesIdentitySampler(Sampler):
    """Randomly samples C copies of N identities each with K instances.
    Args:
        labels (list): contains annotation labels
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
        num_copies (int): number of copies of each example
    """

    def __init__(self, labels, batch_size, num_instances, num_copies=1):
        if batch_size < num_instances:
            raise ValueError(
                "batch_size={} must be no less "
                "than num_instances={}".format(batch_size, num_instances)
            )

        self.labels = labels
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_copies = num_copies
        self.num_labels_per_batch = self.batch_size // self.num_instances  # 32 // 4 = 8
        self.index_dic = defaultdict(list)
        for index, label in enumerate(labels):
            self.index_dic[label].append(index)
        self.unique_labels = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for label in self.unique_labels:
            idxs = self.index_dic[label]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.unique_labels:
            idxs = copy.deepcopy(
                self.index_dic[label]
            )  # indices of this label in the data
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[label].append(
                        batch_idxs
                    )  # split the label's indices into chunks of size num_instances
                    batch_idxs = []

        avai_labels = copy.deepcopy(self.unique_labels)
        final_idxs = []

        while len(avai_labels) >= self.num_labels_per_batch:
            selected_labels = random.sample(
                avai_labels, self.num_labels_per_batch
            )  # sample some labels
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)  # pop a num_instance chunk
                final_idxs.extend(
                    duplicate_list(batch_idxs, self.num_copies)
                )  # duplicate each chunk element, and add to iterator
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)

        return iter(final_idxs)


def duplicate_list(a, k):
    """Duplicate each element in list a for k times"""
    return [val for val in a for _ in range(k)]
