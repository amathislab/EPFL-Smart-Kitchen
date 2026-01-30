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
import re
from argparse import Namespace
from os.path import join as pjoin

from utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip("-").lstrip("+")  
    try:
        reg = re.compile(r"^[-+]?[0-9]+\.[0-9]+$")
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip("-").lstrip("+")  
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, **kwargs):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = (
        "-------------- End ----------------",
        "------------ Options -------------",
        "\n",
    )
    print("Reading", opt_path)
    with open(opt_path, "r") as f:
        for line in f:
            if line.strip() not in skip:
                key, value = line.strip("\n").split(": ")
                if value in ("True", "False"):
                    opt_dict[key] = value == "True"
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    opt_dict["which_epoch"] = "finest"
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, "model")
    opt.meta_dir = pjoin(opt.save_root, "meta")

    if opt.dataset_name == "t2m":
        opt.data_root = "./dataset/HumanML3D/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    elif opt.dataset_name == "kit":
        opt.data_root = "./dataset/KIT-ML/"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "texts")
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    elif opt.dataset_name == "kitchen":
        opt.data_root = "dataset/Kitchen"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.text_dir = pjoin(opt.data_root, "Kitchen/Annotations")
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    elif opt.dataset_name == "kitchen_new":
        opt.data_root = "dataset/New_kitchen_data"
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.img_dir = pjoin(opt.data_root, "holo_images")
        opt.img_dir = pjoin(opt.data_root, "holo_images")
        opt.text_dir = "dataset/Kitchen/Kitchen/Annotations"
        opt.joints_num = 52
        opt.dim_pose = 327
        opt.max_motion_length = 196
        opt.max_motion_frame = 19
        opt.clip_version = "ViT-B/32"
        opt.max_motion_token = 55
    else:
        raise KeyError("Dataset not recognized")
    if not hasattr(opt, "unit_length"):
        opt.unit_length = 4
    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    opt_dict.update(kwargs)  # Overwrite with kwargs params

    return opt
