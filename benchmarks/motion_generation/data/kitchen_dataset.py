#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License.
# A copy is included in LICENSE.
#
import codecs as cs
import copy
import os
import random
from os.path import join as pjoin

import clip
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class KitchenMotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, mode="train"):
        self.opt = opt
        joints_num = opt.joints_num
        self.stride_t = opt.stride_t

        self.data = []
        self.lengths = []
        id_list = os.listdir(pjoin(opt.motion_dir, mode))


        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, mode, name))
                self.lengths.append((motion.shape[0] - opt.window_size) // opt.stride_t)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4 : 4 + (joints_num - 1) * 3] = std[4 : 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] = (
                std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] / 1.0
            )
            # local_velocity (B, seq_len, joint_num*3)
            std[
                4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3
            ] = (
                std[
                    4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3
                ]
                / 1.0
            )
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = (
                std[4 + (joints_num - 1) * 9 + joints_num * 3 :] / opt.feat_bias
            )

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print(
            "Total number of motions {}, snippets {}".format(
                len(self.data), self.cumsum[-1]
            )
        )

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][
            idx * self.stride_t : idx * self.stride_t + self.opt.window_size
        ]
        # make nan to 0
        motion = np.nan_to_num(motion)
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class KitchenNewMotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, mode="train", motion_type="action"):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        index_to_keep = np.loadtxt(
            pjoin(
                opt.data_root,
                "filter_indexes",
                "{}_{}_keep_samples_16_500.txt".format(mode, motion_type),
            ),
            dtype=int,
        )
        id_list = ["{:0>8d}.npy".format(i) for i in index_to_keep]


        self.sample_indexes = []
        for idx, name in tqdm(enumerate(id_list)):
            motion = np.load(pjoin(opt.motion_dir, mode, name))
            if motion.shape[0] < opt.window_size:
                motion_list = []
                for i in range(0, opt.window_size, motion.shape[0] + opt.radius):
                    motion_list.append(
                        np.concatenate(
                            [motion, np.zeros((opt.radius, motion.shape[1]))], axis=0
                        )
                    )
                motion = np.concatenate(motion_list, axis=0)[: opt.window_size]
                self.sample_indexes.append(idx)
            else:
                resammple_times = (motion.shape[0] // opt.window_size) + 1
                self.sample_indexes.extend([idx] * resammple_times)

            self.data.append(motion)

        self.mean = mean
        self.std = std
        print(
            "Total number of motions {}, snippets {}".format(
                len(self.data), len(self.sample_indexes)
            )
        )

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.sample_indexes)

    def __getitem__(self, item):
        motion_id = self.sample_indexes[item]
        motion = self.data[motion_id].copy()

        # random crop
        if motion.shape[0] > self.opt.window_size:
            idx = random.randint(0, motion.shape[0] - self.opt.window_size)
            motion = motion[idx : idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class KitchenClsMotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, mode="train"):
        self.opt = opt
        joints_num = opt.joints_num
        self.window_size = opt.window_size

        self.data = []
        self.lengths = []


        anno_df = pd.read_csv(
            os.path.join(opt.text_dir, "{}_data_new.csv".format(mode))
        )

        # change all the strings in Actions_0 column of anno_df to lower case
        anno_df["Actions_0"] = anno_df["Actions_0"].str.lower()

        # only keep the data in anno_df that the Actions_0 is opt.action_type
        anno_df = anno_df[anno_df["Actions_0"] == opt.action_type]

        for idx, df_item in tqdm(anno_df.iterrows()):

            name = "{}_{}.npy".format(df_item["Participants"], df_item["Session"])
            motion = np.load(pjoin(opt.motion_dir, mode, name))
            start_frame = int(df_item["Start_frame"])
            end_frame = int(df_item["End_frame"])
            if end_frame - start_frame < opt.window_size:
                # extend the motion to the window size by extending the start and end frames
                start_frame = (
                    start_frame - (opt.window_size - (end_frame - start_frame)) // 2
                )
                end_frame = start_frame + opt.window_size
                if start_frame < 0:
                    start_frame = 0
                    end_frame = opt.window_size
                elif end_frame > len(motion):
                    end_frame = len(motion)
                    start_frame = len(motion) - opt.window_size
            selected_motion = copy.deepcopy(motion[start_frame:end_frame])
            # delete motion to save memory
            del motion
            self.data.append(selected_motion)
            flip_name = "{}_{}_M.npy".format(
                df_item["Participants"], df_item["Session"]
            )
            motion = np.load(pjoin(opt.motion_dir, mode, flip_name))
            selected_motion = copy.deepcopy(motion[start_frame:end_frame])
            del motion
            self.data.append(selected_motion)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4 : 4 + (joints_num - 1) * 3] = std[4 : 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] = (
                std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] / 1.0
            )
            # local_velocity (B, seq_len, joint_num*3)
            std[
                4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3
            ] = (
                std[
                    4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3
                ]
                / 1.0
            )
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = (
                std[4 + (joints_num - 1) * 9 + joints_num * 3 :] / opt.feat_bias
            )

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        if len(motion) > self.window_size:
            idx = random.randint(0, len(motion) - self.window_size)
            motion = motion[idx : idx + self.window_size]
        # make nan to 0
        motion = np.nan_to_num(motion)
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion.astype(np.float32)


class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, mode, w_vectorizer, motion_type):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length

        data_dict = {}
        index_to_keep = np.loadtxt(
            pjoin(
                opt.data_root,
                "filter_indexes",
                "{}_{}_keep_samples_16_500.txt".format(mode, motion_type),
            ),
            dtype=int,
        )
        id_list = ["{:0>8d}.npy".format(i) for i in index_to_keep]

        anno_df = pd.read_csv(
            os.path.join(opt.text_dir, "{}_data_new.csv".format(mode))
        )
        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            motion = np.load(pjoin(opt.motion_dir, mode, name))

            if motion_type == "action":
                action = anno_df["Actions_0"][int(name.split(".")[0][1:])].lower()
                tokens = []
                if len(action.split("-")[0].split("_")) == 1:
                    tokens.append(action.split("-")[0] + "/VERB")
                else:
                    tokens.append(action.split("-")[0].split("_")[0] + "/VERB")
                    for adv in action.split("-")[0].split("_")[1:]:
                        tokens.append(adv + "/ADV")

                for noun in action.split("-")[1].split("_"):
                    if noun != "NaN":
                        tokens.append(noun + "/NOUN")
            elif motion_type == "verb":
                action = anno_df["Verbs_1"][int(name.split(".")[0][1:])].lower()
                tokens = []
                if len(action.split("_")) == 1:
                    tokens.append(action + "/VERB")
                else:
                    tokens.append(action.split("_")[0] + "/VERB")
                    for adv in action.split("-")[0].split("_")[1:]:
                        tokens.append(adv + "/ADV")
            else:
                raise KeyError("Motion Type Not Found")

            text_dict = {}
            text_dict["caption"] = action
            text_dict["tokens"] = tokens

            img_path = os.path.join(opt.img_dir, mode, name.split(".")[0] + ".png")
            assert os.path.exists(img_path), "Image path not found: {}".format(img_path)

            data_dict[name] = {
                "motion": motion,
                "length": len(motion),
                "text": text_dict,
                "img_path": img_path,
            }
            new_name_list.append(name)
            length_list.append(len(motion))

        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        _, self.clip_preprocess = clip.load(
            opt.clip_version, device="cpu", jit=False
        )  # Must set jit=False for training

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, m_length, text_data, img_path = (
            data["motion"],
            data["length"],
            data["text"],
            data["img_path"],
        )
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        m_length = min(m_length, self.max_motion_length)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        # load image
        image = self.clip_preprocess(Image.open(img_path))

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            image,
        )


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, mode, motion_type):
        self.opt = opt
        self.max_length = 20
        self.max_motion_length = opt.max_motion_length

        data_dict = {}
        index_to_keep = np.loadtxt(
            pjoin(
                opt.data_root,
                "filter_indexes",
                "{}_{}_keep_samples_16_500.txt".format(mode, motion_type),
            ),
            dtype=int,
        )
        id_list = ["{:0>8d}.npy".format(i) for i in index_to_keep]

        anno_df = pd.read_csv(
            os.path.join(opt.text_dir, "{}_data_new.csv".format(mode))
        )
        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            motion = np.load(pjoin(opt.motion_dir, mode, name))

            if motion_type == "action":
                action = anno_df["Actions_0"][int(name.split(".")[0][1:])].lower()
                tokens = []
                if len(action.split("-")[0].split("_")) == 1:
                    tokens.append(action.split("-")[0] + "/VERB")
                else:
                    tokens.append(action.split("-")[0].split("_")[0] + "/VERB")
                    for adv in action.split("-")[0].split("_")[1:]:
                        tokens.append(adv + "/ADV")

                for noun in action.split("-")[1].split("_"):
                    if noun != "NaN":
                        tokens.append(noun + "/NOUN")
            elif motion_type == "verb":
                action = anno_df["Verbs_1"][int(name.split(".")[0][1:])].lower()
                tokens = []
                if len(action.split("_")) == 1:
                    tokens.append(action + "/VERB")
                else:
                    tokens.append(action.split("_")[0] + "/VERB")
                    for adv in action.split("-")[0].split("_")[1:]:
                        tokens.append(adv + "/ADV")
            else:
                raise KeyError("Motion Type Not Found")

            text_dict = {}
            text_dict["caption"] = action
            text_dict["tokens"] = tokens

            img_path = os.path.join(opt.img_dir, mode, name.split(".")[0] + ".png")
            assert os.path.exists(img_path), "Image path not found: {}".format(img_path)

            data_dict[name] = {
                "motion": motion,
                "length": len(motion),
                "text": text_dict,
                "img_path": img_path,
            }
            new_name_list.append(name)
            length_list.append(len(motion))

        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        _, self.clip_preprocess = clip.load(
            opt.clip_version, device="cpu", jit=False
        )  # Must set jit=False for training

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, m_length, text_data, img_path = (
            data["motion"],
            data["length"],
            data["text"],
            data["img_path"],
        )
        caption, tokens = text_data["caption"], text_data["tokens"]

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        m_length = min(m_length, self.max_motion_length)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        # load image
        image = self.clip_preprocess(Image.open(img_path))

        return caption, motion, m_length, image
