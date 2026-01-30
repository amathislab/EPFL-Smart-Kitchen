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
import os
import warnings
from os.path import join as pjoin

import torch
import utils.eval_t2m as eval_t2m
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from models.vq.model import RVQVAE
from motion_loaders.dataset_motion_loader import (
    get_kitchen_motion_loader,
)
from options.vq_option import arg_parse
from utils.get_opt import get_opt

warnings.filterwarnings("ignore")
import numpy as np


def load_vq_model(vq_opt, which_epoch):

    vq_model = RVQVAE(
        vq_opt,
        dim_pose,
        vq_opt.nb_code,
        vq_opt.code_dim,
        vq_opt.code_dim,
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        vq_opt.vq_norm,
    )
    ckpt = torch.load(
        pjoin(
            vq_opt.checkpoints_dir,
            vq_opt.dataset_name,
            vq_opt.name,
            "model",
            which_epoch,
        ),
        map_location="cpu",
    )
    model_key = "vq_model" if "vq_model" in ckpt else "net"
    vq_model.load_state_dict(ckpt[model_key])
    vq_epoch = ckpt["ep"] if "ep" in ckpt else -1
    print(f"Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}")
    return vq_model, vq_epoch


if __name__ == "__main__":
    ##### ---- Exp dirs ---- #####
    args = arg_parse(False)
    args.device = torch.device(
        "cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id)
    )

    args.out_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, "eval")
    os.makedirs(args.out_dir, exist_ok=True)

    f = open(pjoin(args.out_dir, "%s.log" % args.ext), "w")

    dataset_opt_path = "checkpoints/kitchen_new/Comp_v6_KLD005/opt.txt"

    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    wrapper_opt.motion_type = args.motion_type
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    args.nb_joints = 52
    dim_pose = 327

    eval_val_loader, _ = get_kitchen_motion_loader(
        dataset_opt_path, int(args.batch_size / 8), args.motion_type, device=args.device
    )

    print(len(eval_val_loader))

    ##### ---- Network ---- #####
    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.name, "opt.txt")
    vq_opt = get_opt(vq_opt_path, device=args.device)

    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, "model")
    for file in os.listdir(model_dir):
        if args.which_epoch != "all" and args.which_epoch not in file:
            continue
        print(file)
        net, ep = load_vq_model(vq_opt, file)

        net.eval()
        net.cuda()

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mae = []
        repeat_time = 20
        for i in range(repeat_time):
            best_fid, best_div, Rprecision, best_matching, l1_dist = (
                eval_t2m.evaluation_vqvae_plus_mpjpe(
                    eval_val_loader,
                    net,
                    i,
                    eval_wrapper=eval_wrapper,
                    num_joint=args.nb_joints,
                )
            )
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            mae.append(l1_dist)

        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mae = np.array(mae)

        print(f"{file} final result, epoch {ep}")
        print(f"{file} final result, epoch {ep}", file=f, flush=True)

        msg_final = (
            f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}\n"
            f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}\n"
            f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}\n"
            f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}\n"
            f"\tMAE:{np.mean(mae):.3f}, conf.{np.std(mae)*1.96/np.sqrt(repeat_time):.3f}\n\n"
        )
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()
