#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License.
# A copy is included in LICENSE.
#
# Incorporates code adapted from MoMask by EricGuo5513
# Original work Copyright (c) 2023 by EricGuo5513
# Source: https://github.com/EricGuo5513/momask-codes
# Originally licensed under MIT License
#
from os.path import join as pjoin

import numpy as np
from data.kitchen_dataset import Text2MotionDatasetEval, collate_fn  # TODO
from torch.utils.data import DataLoader
from utils.get_opt import get_opt
from utils.word_vectorizer import WordVectorizer


def get_dataset_motion_loader(opt_path, batch_size, fname, device):
    opt = get_opt(opt_path, device)

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == "t2m" or opt.dataset_name == "kit":
        print("Loading dataset %s ..." % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
        std = np.load(pjoin(opt.meta_dir, "std.npy"))

        w_vectorizer = WordVectorizer("./glove", "our_vab")
        split_file = pjoin(opt.data_root, "%s.txt" % fname)
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn,
            shuffle=True,
        )
    else:
        raise KeyError("Dataset not Recognized !!")

    print("Ground Truth Dataset Loading Completed!!!")
    return dataloader, dataset


def get_kitchen_motion_loader(opt_path, batch_size, motion_type, device):
    opt = get_opt(opt_path, device)

    print("Loading dataset %s ..." % opt.dataset_name)

    mean = np.load(pjoin(opt.data_root, "Mean.npy"))
    std = np.load(pjoin(opt.data_root, "Std.npy"))

    w_vectorizer = WordVectorizer("./glove", "our_vab")
    dataset = Text2MotionDatasetEval(opt, mean, std, "test", w_vectorizer, motion_type)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn,
        shuffle=True,
    )

    print("Ground Truth Dataset Loading Completed!!!")
    return dataloader, dataset
