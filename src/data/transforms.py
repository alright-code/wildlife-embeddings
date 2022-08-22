import random

import torchvision.transforms as transforms
from PIL import ImageFilter

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# https://github.com/facebookresearch/moco/blob/main/moco/loader.py
class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def build_transform(transforms_str, image_size):
    transform_list = []

    for tfm in transforms_str:
        if tfm == "affine":
            transform_list += [
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    shear=(5, 5),
                    resample=0,
                    fillcolor=0,
                )
            ]

        elif tfm == "blur":
            transform_list += [GaussianBlur()]

        elif tfm == "center_crop":
            transform_list += [transforms.CenterCrop(image_size)]

        elif tfm == "color_jitter":
            transform_list += [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                )
            ]

        elif tfm == "flip":
            transform_list += [transforms.RandomHorizontalFlip()]

        elif tfm == "grayscale":
            transform_list += [transforms.RandomGrayscale(p=0.2)]

        elif tfm == "perspective":
            transform_list += [transforms.RandomPerspective()]

        elif tfm == "random_resized_crop":
            transform_list += [transforms.RandomResizedCrop(image_size, scale=(0.2, 1))]

        elif tfm == "resize":
            transform_list += [transforms.Resize(image_size)]

        elif tfm == "normalize":
            continue

        else:
            raise ValueError

    transform_list += [transforms.ToTensor()]

    if "normalize" in transforms_str:
        transform_list += [transforms.Normalize(mean=MEAN, std=STD)]

    return transforms.Compose(transform_list)


def inverse_normalize(image):
    tfm = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / x for x in STD]),
            transforms.Normalize(mean=[-x for x in MEAN], std=[1.0, 1.0, 1.0]),
        ]
    )

    return tfm(image)
