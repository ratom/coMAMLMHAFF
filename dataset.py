"""
dataset.py
----------
Few-shot episode dataset for cattle muzzle identification.

FSLCDataset samples N-way K-shot episodes on the fly.  Each call to
__getitem__ returns one episode consisting of a support set and a query
set, each pre-processed for both the ViT and ResNet-50 backbones.

Expected folder layout:
    root/
        <class_A>/
            img1.jpg
            img2.jpg
            ...
        <class_B>/
            ...

A class is included only if it has at least (num_shots + num_queries)
images available.
"""

import os
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FSLCDataset(Dataset):
    """N-way K-shot episode dataset.

    Args:
        folder      (str): Root directory containing per-class sub-folders.
        num_ways    (int): N — number of classes per episode.
        num_shots   (int): K — support images per class.
        num_queries (int): Query images per class.
        episodes    (int): Virtual dataset length (episodes sampled randomly).
    """

    # ViT preprocessing: normalise to [-1, 1]
    _TRANS_NORM = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # ResNet preprocessing: ImageNet mean/std
    _RES_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __init__(self, folder: str, num_ways: int = 5, num_shots: int = 5,
                 num_queries: int = 5, episodes: int = 1000) -> None:
        self.num_ways    = num_ways
        self.num_shots   = num_shots
        self.num_queries = num_queries
        self.episodes    = episodes

        self.trans_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(**self._TRANS_NORM),
        ])
        self.res_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(**self._RES_NORM),
        ])

        self.data = self._load_class_images(folder)
        self.classes = list(self.data.keys())

        if len(self.classes) < self.num_ways:
            raise ValueError(
                f"Not enough classes in '{folder}': "
                f"need {self.num_ways}, found {len(self.classes)}."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_class_images(self, folder: str) -> dict:
        """Scan folder and return {class_name: [image_paths]} dict."""
        _IMG_EXTS = ('.jpg', '.jpeg', '.png')
        data = {}
        for cls in sorted(os.listdir(folder)):
            cls_dir = os.path.join(folder, cls)
            if not os.path.isdir(cls_dir):
                continue
            imgs = [
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if f.lower().endswith(_IMG_EXTS)
            ]
            if len(imgs) >= self.num_shots + self.num_queries:
                data[cls] = imgs
        return data

    def _load_dual(self, paths) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a list of image paths and return (trans_stack, res_stack)."""
        trans_list, res_list = [], []
        for p in paths:
            img = Image.open(p).convert('RGB')
            trans_list.append(self.trans_transform(img))
            res_list.append(self.res_transform(img))
        return torch.stack(trans_list), torch.stack(res_list)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.episodes

    def __getitem__(self, idx: int):
        """Sample one N-way K-shot episode.

        Returns:
            support_trans  (Tensor): (N*K,  3, 224, 224) — ViT input
            support_res    (Tensor): (N*K,  3, 224, 224) — ResNet input
            support_labels (Tensor): (N*K,)
            query_trans    (Tensor): (N*Q,  3, 224, 224) — ViT input
            query_res      (Tensor): (N*Q,  3, 224, 224) — ResNet input
            query_labels   (Tensor): (N*Q,)
        """
        sampled_classes = random.sample(self.classes, self.num_ways)

        support_paths, support_labels = [], []
        query_paths,   query_labels   = [], []

        for label_idx, cls in enumerate(sampled_classes):
            paths = random.sample(
                self.data[cls], self.num_shots + self.num_queries
            )
            support_paths.extend(paths[:self.num_shots])
            query_paths.extend(paths[self.num_shots:])
            support_labels.extend([label_idx] * self.num_shots)
            query_labels.extend([label_idx]   * self.num_queries)

        support_trans, support_res = self._load_dual(support_paths)
        query_trans,   query_res   = self._load_dual(query_paths)

        return (
            support_trans, support_res, torch.tensor(support_labels),
            query_trans,   query_res,   torch.tensor(query_labels),
        )
