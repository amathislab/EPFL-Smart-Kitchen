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
import argparse
import os

import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--name",
            type=str,
            default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns",
            help="Name of this trial",
        )

        self.parser.add_argument(
            "--vq_name",
            type=str,
            default="rvq_nq1_dc512_nc512",
            help="Name of the rvq model.",
        )

        self.parser.add_argument("--gpu_id", type=int, default=-1, help="GPU id")
        self.parser.add_argument(
            "--dataset_name",
            type=str,
            default="t2m",
            help="Dataset Name, {t2m} for humanml3d, {kit} for kit-ml",
        )
        self.parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here.",
        )

        self.parser.add_argument(
            "--latent_dim",
            type=int,
            default=384,
            help="Dimension of transformer latent.",
        )
        self.parser.add_argument(
            "--n_heads", type=int, default=6, help="Number of heads."
        )
        self.parser.add_argument(
            "--n_layers", type=int, default=8, help="Number of attention layers."
        )
        self.parser.add_argument("--ff_size", type=int, default=1024, help="FF_Size")
        self.parser.add_argument(
            "--dropout", type=float, default=0.2, help="Dropout ratio in transformer"
        )

        self.parser.add_argument(
            "--max_motion_length", type=int, default=196, help="Max length of motion"
        )
        self.parser.add_argument(
            "--unit_length", type=int, default=4, help="Downscale ratio of VQ"
        )

        self.parser.add_argument(
            "--force_mask", action="store_true", help="True: mask out conditions"
        )

        self.parser.add_argument(
            "--motion_type", type=str, default="action", help="dataset directory"
        )
        self.parser.add_argument(
            "--model_type",
            type=str,
            default="MARDM-SiT-XL",
            help="Model type, MARDM-SiT-XL or MARDM-DDPM-XL",
        )

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train

        if self.opt.gpu_id != -1:
            # self.opt.gpu_id = int(self.opt.gpu_id)
            torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")
        if self.is_train:
            # save to the disk
            expr_dir = os.path.join(
                self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name
            )
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, "opt.txt")
            with open(file_name, "wt") as opt_file:
                opt_file.write("------------ Options -------------\n")
                for k, v in sorted(args.items()):
                    opt_file.write("%s: %s\n" % (str(k), str(v)))
                opt_file.write("-------------- End ----------------\n")
        return self.opt
