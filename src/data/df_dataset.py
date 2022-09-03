import math

import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def get_labels(data_df):
    names = pd.unique(data_df["name"])

    id_to_labels_dict = {name: label for label, name in enumerate(names)}
    labels = [id_to_labels_dict[name] for name in data_df["name"]]

    return labels


class DFDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

        self.labels = get_labels(self.data)
        self.num_labels = len(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        annot = row["annot"]
        img_path = row["image"]
        name = row["name"]
        x = row["x"]
        y = row["y"]
        w = row["w"]
        h = row["h"]
        theta = row["theta"]
        label = self.labels[idx]

        with open(img_path, "rb") as f:
            img = ImageOps.exif_transpose(Image.open(f))
            img.load()

        img = img.crop((x, y, min(x + w, img.width), min(y + h, img.height)))
        img = img.rotate(math.degrees(theta))

        if self.transform is not None:
            img = self.transform(img)

        return img, label, annot, name
