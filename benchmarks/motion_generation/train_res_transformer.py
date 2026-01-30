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
from os.path import join as pjoin

import numpy as np
import torch
from data.kitchen_dataset import Text2MotionDataset
from models.mask_transformer.transformer import ResidualTransformer
from models.mask_transformer.transformer_trainer import ResidualTransformerTrainer
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from models.vq.model import RVQVAE
from motion_loaders.dataset_motion_loader import get_kitchen_motion_loader
from options.train_option import TrainT2MOptions
from torch.utils.data import DataLoader
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.paramUtil import kit_kinematic_chain, t2m_all_chain, t2m_kinematic_chain
from utils.plot_script import plot_3d_motion


def plot_t2m(data, save_dir, captions, m_lengths):
    data = train_dataset.inv_transform(data)

    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[: m_lengths[i]]
        joint = recover_from_ric(
            torch.from_numpy(joint_data).float(), opt.joints_num
        ).numpy()
        save_path = pjoin(save_dir, "%02d.mp4" % i)
        plot_3d_motion(
            save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius
        )


def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, "opt.txt")
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(
        vq_opt,
        dim_pose,
        vq_opt.nb_code,
        vq_opt.code_dim,
        vq_opt.output_emb_width,
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
            "net_best_fid.tar",
        ),
        map_location=opt.device,
    )
    model_key = "vq_model" if "vq_model" in ckpt else "net"
    vq_model.load_state_dict(ckpt[model_key])
    print(f"Loading VQ Model {opt.vq_name}")
    vq_model.to(opt.device)
    return vq_model, vq_opt


if __name__ == "__main__":
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, "model")
    opt.eval_dir = pjoin(opt.save_root, "animation")
    opt.log_dir = pjoin("./log/res/", opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "t2m":
        opt.data_root = "./dataset/HumanML3D"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = "./checkpoints/t2m/Comp_v6_KLD005/opt.txt"

    elif opt.dataset_name == "kitchen":
        opt.data_root = "dataset/Kitchen"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.joints_num = 22
        opt.max_motion_len = 55
        dim_pose = 263
        fps = 30
        radius = 4
        kinematic_chain = t2m_kinematic_chain
        dataset_opt_path = "./checkpoints/kitchen/Comp_v6_KLD005/opt.txt"

    elif opt.dataset_name == "kitchen_new":
        opt.data_root = "dataset/New_kitchen_data"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.img_dir = pjoin(opt.data_root, "holo_images")
        opt.joints_num = 52
        opt.max_motion_len = 55
        dim_pose = 327
        fps = 30
        opt.radius = 4
        kinematic_chain = t2m_all_chain
        dataset_opt_path = "./checkpoints/kitchen_new/Comp_v6_KLD005/opt.txt"

    elif opt.dataset_name == "kit":  # TODO
        opt.data_root = "./dataset/KIT-ML"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_len = 55
        kinematic_chain = kit_kinematic_chain
        dataset_opt_path = "./checkpoints/kit/Comp_v6_KLD005/opt.txt"

    else:
        raise KeyError("Dataset Does Not Exist")

    opt.text_dir = "dataset/Kitchen/Kitchen/Annotations"

    vq_model, vq_opt = load_vq_model()

    opt.clip_version = "ViT-B/32"

    opt.num_tokens = vq_opt.nb_code
    opt.num_quantizers = vq_opt.num_quantizers

    res_transformer = ResidualTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode="text",
        latent_dim=opt.latent_dim,
        ff_size=opt.ff_size,
        num_layers=opt.n_layers,
        num_heads=opt.n_heads,
        dropout=opt.dropout,
        clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=opt.cond_drop_prob,
        share_weight=opt.share_weight,
        clip_version=opt.clip_version,
        opt=opt,
    )
    all_params = 0
    pc_transformer = sum(
        param.numel() for param in res_transformer.parameters_wo_clip()
    )

    print(res_transformer)
    all_params += pc_transformer

    print("Total parameters of all models: {:.2f}M".format(all_params / 1000_000))

    mean = np.load(pjoin(opt.data_root, "Mean.npy"))
    std = np.load(pjoin(opt.data_root, "Std.npy"))

    train_dataset = Text2MotionDataset(opt, mean, std, "train", opt.motion_type)
    val_dataset = Text2MotionDataset(opt, mean, std, "test", opt.motion_type)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )

    eval_val_loader, _ = get_kitchen_motion_loader(
        dataset_opt_path, opt.batch_size, opt.motion_type, device=opt.device
    )

    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    wrapper_opt.motion_type = opt.motion_type
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    trainer = ResidualTransformerTrainer(opt, res_transformer, vq_model)

    trainer.train(
        train_loader,
        val_loader,
        eval_val_loader,
        eval_wrapper=eval_wrapper,
        plot_eval=plot_t2m,
    )
