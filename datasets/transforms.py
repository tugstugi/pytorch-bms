__author__ = 'Erdene-Ochir Tuguldur'

import cv2
import random
import numpy as np
import albumentations as album
from albumentations.pytorch import ToTensorV2


def get_test_transform(args):
    return Compose([
        # TestFix(),
        ApplyAlbumentations(
            album.Compose([
                album.Resize(args.image_size, args.image_size),
                album.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2()
            ])
        )
    ])


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class ApplyAlbumentations(object):
    """Apply transforms from Albumentations."""

    def __init__(self, a_transform):
        self.a_transform = a_transform

    def __call__(self, data):
        data['input'] = self.a_transform(image=data['input'])['image']
        return data


class TestFix(object):
    def __init__(self):
        self.a_transform = album.Compose([album.Transpose(p=1), album.VerticalFlip(p=1)])

    def __call__(self, data):
        h, w, _ = data['input'].shape
        if h > w:
            data['input'] = self.a_transform(image=data['input'])['image']
        return data


class CropAugment(object):
    """Crop pixels from borders. """
    def __init__(self, probability=0.5, crops=(5, 10, 15, 20)):
        self.probability = probability
        self.crops = crops

    def __call__(self, data):
        if random.random() < self.probability:
            img = data['input']
            crop = random.choice(self.crops)
            img = img[crop:-crop, crop:-crop, :]
            data['input'] = img
        return data


class RandomNoiseAugment(object):
    """Add random noise. """
    def __init__(self, probability=0.5, frac=0.005):
        self.probability = probability
        self.frac = frac

    def __call__(self, data):
        if random.random() < self.probability:
            img = data['input']
            max_val = int(round(1 / self.frac))
            indices = np.random.randint(0, max_val, size=img.shape)
            img[indices == 0] = 0
            data['input'] = img
        return data
