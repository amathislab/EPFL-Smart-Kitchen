#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License.
# A copy is included in LICENSE.
#
import os
from os.path import join as pjoin

import numpy as np
import torch
import utils.eval_t2m as eval_t2m
from models.mardm.AE import AE_models
from models.mardm.MARDM import MARDM_models
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from motion_loaders.dataset_motion_loader import get_kitchen_motion_loader
from options.eval_option import EvalT2MOptions
from utils.fixseed import fixseed
from utils.get_opt import get_opt


def load_vq_model(vq_name):
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, vq_name, "opt.txt")
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = AE_models["AE_Model"](input_width=vq_opt.dim_pose)
    ckpt = torch.load(
        pjoin(
            vq_opt.checkpoints_dir,
            vq_opt.dataset_name,
            vq_opt.name,
            "model",
            "net_best_fid.tar",
        ),
        map_location="cpu",
    )
    model_key = "vq_model" if "vq_model" in ckpt else "net"
    vq_model.load_state_dict(ckpt[model_key])
    print(f"Loading VQ Model {opt.vq_name}")
    return vq_model, vq_opt


def load_trans_model(model_opt, which_model):
    t2m_transformer = MARDM_models[model_opt.model_type](
        ae_dim=model_opt.code_dim, cond_mode="text"
    )
    ckpt = torch.load(
        pjoin(
            model_opt.checkpoints_dir,
            model_opt.dataset_name,
            model_opt.name,
            "model",
            which_model,
        ),
        map_location=opt.device,
    )
    model_key = "ema_mardm" if "ema_mardm" in ckpt else "t2m_transformer"
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(
        ckpt[model_key], strict=False
    )
    assert len(unexpected_keys) == 0
    assert all([k.startswith("clip_model.") for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


if __name__ == "__main__":
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 327

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, "model")
    out_dir = pjoin(root_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    out_path = pjoin(out_dir, "%s.log" % opt.ext)

    f = open(pjoin(out_path), "w")

    model_opt_path = pjoin(root_dir, "opt.txt")
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = "ViT-B/32"

    vq_model, vq_opt = load_vq_model(model_opt.vq_name)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.code_dim = vq_opt.code_dim

    dataset_opt_path = "./checkpoints/kitchen_new/Comp_v6_KLD005/opt.txt"

    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    wrapper_opt.motion_type = opt.motion_type
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 52

    eval_val_loader, _ = get_kitchen_motion_loader(
        dataset_opt_path, 32, opt.motion_type, device=opt.device
    )

    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        print("loading checkpoint {}".format(file))
        t2m_transformer = load_trans_model(model_opt, file)
        t2m_transformer.eval()
        vq_model.eval()

        t2m_transformer.to(opt.device)
        vq_model.to(opt.device)

        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mm = []

        repeat_time = 20
        for i in range(repeat_time):
            with torch.no_grad():
                best_fid, best_div, Rprecision, best_matching, best_mm = (
                    eval_t2m.evaluation_mardm_test(
                        eval_val_loader,
                        vq_model,
                        t2m_transformer,
                        i,
                        eval_wrapper=eval_wrapper,
                        time_steps=opt.time_steps,
                        cond_scale=opt.cond_scale,
                        temperature=opt.temperature,
                        cal_mm=True,
                    )
                )
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            mm.append(best_mm)

        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mm = np.array(mm)

        print(f"{file} final result:")
        print(f"{file} final result:", file=f, flush=True)

        msg_final = (
            f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n"
            f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        )
        print(msg_final)
        print(msg_final, file=f, flush=True)

    f.close()


