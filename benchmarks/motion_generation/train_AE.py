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
from data.kitchen_dataset import KitchenClsMotionDataset, KitchenNewMotionDataset
from models.mardm.AE import AE_models
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from models.vq.vq_trainer import AETokenizerTrainer
from motion_loaders.dataset_motion_loader import get_kitchen_motion_loader
from options.vq_option import arg_parse
from torch.utils.data import DataLoader
from utils import paramUtil
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

os.environ["OMP_NUM_THREADS"] = "1"


def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(
            torch.from_numpy(joint_data).float(), opt.joints_num
        ).numpy()
        save_path = pjoin(save_dir, "%02d.mp4" % (i))
        plot_3d_motion(
            save_path, kinematic_chain, joint, title="None", fps=fps, radius=4
        )


if __name__ == "__main__":
    opt = arg_parse(True)
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, "model")
    opt.meta_dir = pjoin(opt.save_root, "meta")
    opt.eval_dir = pjoin(opt.save_root, "animation")
    opt.log_dir = pjoin("./log/ae/", opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    assert opt.dataset_name == "kitchen_new"
    opt.data_root = "dataset/New_kitchen_data"
    opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
    opt.text_dir = "dataset/Kitchen/Kitchen/Annotations"
    opt.joints_num = 52
    dim_pose = 327
    fps = 30
    opt.radius = 4
    kinematic_chain = paramUtil.t2m_all_chain
    dataset_opt_path = "./checkpoints/kitchen_new/Comp_v6_KLD005/opt.txt"

    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    wrapper_opt.motion_type = opt.motion_type
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    mean = np.load(pjoin(opt.data_root, "Mean.npy"))
    std = np.load(pjoin(opt.data_root, "Std.npy"))

    net = AE_models["AE_Model"](input_width=dim_pose)

    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)

    print("Total parameters of all models: {}M".format(pc_vq / 1000_000))

    trainer = AETokenizerTrainer(opt, vq_model=net)

    if opt.action_type == "all":
        train_dataset = KitchenNewMotionDataset(opt, mean, std, mode="train")
        val_dataset = KitchenNewMotionDataset(opt, mean, std, mode="test")
    else:
        train_dataset = KitchenClsMotionDataset(opt, mean, std, mode="train")
        val_dataset = KitchenClsMotionDataset(opt, mean, std, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        drop_last=False,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        drop_last=False,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    eval_val_loader, _ = get_kitchen_motion_loader(
        dataset_opt_path, int(opt.batch_size / 8), opt.motion_type, device=opt.device
    )
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, None)
