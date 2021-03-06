# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
from PIL import Image

from util.data import create_transform
from util.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT

from paddle.io import Dataset
import paddle.vision.transforms as transforms
import paddle.vision.datasets as datasets


class ImageNetDataset(Dataset):
    def __init__(
            self,
            image_root,
            cls_label_path,
            transform=None):
        self._img_root = image_root
        self._cls_path = cls_label_path
        self.transform = transform
        self._load_anno()

    def _load_anno(self):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        images = []
        labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split(" ")
                images.append(os.path.join(self._img_root, l[0]))
                labels.append(int(l[1]))
                assert os.path.exists(images[-1]), images[-1]

        self.samples = list(zip(images, labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if is_train and hasattr(args, 'cls_label_path_train') and args.cls_label_path_train:
        dataset = ImageNetDataset(args.data_path, args.cls_label_path_train, transform=transform)
    elif not is_train and hasattr(args, 'cls_label_path_val') and args.cls_label_path_val:
        dataset = ImageNetDataset(args.data_path, args.cls_label_path_val, transform=transform)
    else:
        root = os.path.join(args.data_path,
            'train' if is_train and not (hasattr(args, 'debug') and args.debug) else 'val')
        dataset = datasets.DatasetFolder(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    crop_pct = args.crop_pct if hasattr(args, 'crop_pct') else DEFAULT_CROP_PCT
    size = int(args.input_size / crop_pct)
    train_interpolation = args.train_interpolation \
        if hasattr(args, 'train_interpolation') else 'bilinear'
    if train_interpolation == 'random':
        train_interpolation = 'bicubic'
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=train_interpolation),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform
