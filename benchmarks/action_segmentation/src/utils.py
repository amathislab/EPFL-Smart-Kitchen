#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License. 
# A copy is included in LICENSE.
#
import pickle
from collections import defaultdict

import numpy as np


def get_keypoint_info(mode="body_pose"):
    keypoint_info = {
        0: dict(name="nose", id=0, color=[51, 153, 255], type="upper", swap=""),
        1: dict(
            name="left_eye", id=1, color=[51, 153, 255], type="upper", swap="right_eye"
        ),
        2: dict(
            name="right_eye", id=2, color=[51, 153, 255], type="upper", swap="left_eye"
        ),
        3: dict(
            name="left_ear", id=3, color=[51, 153, 255], type="upper", swap="right_ear"
        ),
        4: dict(
            name="right_ear", id=4, color=[51, 153, 255], type="upper", swap="left_ear"
        ),
        5: dict(
            name="left_shoulder",
            id=5,
            color=[0, 255, 0],
            type="upper",
            swap="right_shoulder",
        ),
        6: dict(
            name="right_shoulder",
            id=6,
            color=[255, 128, 0],
            type="upper",
            swap="left_shoulder",
        ),
        7: dict(
            name="left_elbow", id=7, color=[0, 255, 0], type="upper", swap="right_elbow"
        ),
        8: dict(
            name="right_elbow",
            id=8,
            color=[255, 128, 0],
            type="upper",
            swap="left_elbow",
        ),
        9: dict(
            name="left_wrist", id=9, color=[0, 255, 0], type="upper", swap="right_wrist"
        ),
        10: dict(
            name="right_wrist",
            id=10,
            color=[255, 128, 0],
            type="upper",
            swap="left_wrist",
        ),
        11: dict(
            name="left_hip", id=11, color=[0, 255, 0], type="lower", swap="right_hip"
        ),
        12: dict(
            name="right_hip", id=12, color=[255, 128, 0], type="lower", swap="left_hip"
        ),
        13: dict(
            name="left_knee", id=13, color=[0, 255, 0], type="lower", swap="right_knee"
        ),
        14: dict(
            name="right_knee",
            id=14,
            color=[255, 128, 0],
            type="lower",
            swap="left_knee",
        ),
        15: dict(
            name="left_ankle",
            id=15,
            color=[0, 255, 0],
            type="lower",
            swap="right_ankle",
        ),
        16: dict(
            name="right_ankle",
            id=16,
            color=[255, 128, 0],
            type="lower",
            swap="left_ankle",
        ),
    }

    hand_info = {i: dict(name=f"hand_{i}", id=i) for i in range(17, 17 + 42)}
    both_info = keypoint_info.copy()
    both_info.update(hand_info)
    eye_info = {100 + i: dict(name=f"eye_gaze_{i}") for i in range(10)}
    holo_info = {i: dict(name=f"holo_hand_{i}", id=i) for i in range(200, 252)}
    return keypoint_info, hand_info, both_info, eye_info, holo_info


def get_hand_bone_list():
    hand_bone_list = (
        np.array(
            [
                [17, 18],
                [18, 19],
                [19, 20],
                [20, 21],
                [17, 22],
                [22, 23],
                [23, 24],
                [24, 25],
                [17, 26],
                [26, 27],
                [27, 28],
                [28, 29],
                [17, 30],
                [30, 31],
                [31, 32],
                [32, 33],
                [17, 34],
                [34, 35],
                [35, 36],
                [36, 37],  # left hand
                [38, 39],
                [39, 40],
                [40, 41],
                [41, 42],
                [38, 43],
                [43, 44],
                [44, 45],
                [45, 46],
                [38, 47],
                [47, 48],
                [48, 49],
                [49, 50],
                [38, 51],
                [51, 52],
                [52, 53],
                [53, 54],
                [38, 55],
                [55, 56],
                [56, 57],
                [57, 58],
            ]
        )
        - 17
    )  # right hand

    hololen2mano = np.array(
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
        + list(
            np.array(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    7,
                    8,
                    9,
                    10,
                    12,
                    13,
                    14,
                    15,
                    17,
                    18,
                    19,
                    20,
                    22,
                    23,
                    24,
                    25,
                ]
            )
            + 26
        )
    )

    return hand_bone_list, hololen2mano


def reformat_params(params):
    res = defaultdict(lambda: defaultdict(lambda: {}))
    for k, v in params.items():
        big_key, small_key = k.split("/")[0], "/".join(k.split("/")[1:])
        if len(small_key.split("/")) == 1:
            res[big_key][small_key] = v
        else:
            group, key = small_key.split("/")
            res[big_key][group][key] = v
    return res


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def binarize_data(data: list, max_frame: int = -1) -> np.array:
    """Convert file in start, stop, confusion in to binary matrix of size (N_classes, N_timepoints)"""
    data = data[3][0]
    all_annot = [arr for arr in data if len(arr)]
    all_annot = np.concatenate(all_annot, axis=0)
    max_frame = np.max(all_annot) if max_frame == -1 else max_frame
    label_array = np.zeros((len(data), max_frame))
    for k, arr in enumerate(data):
        for start, stop, confusion in arr:
            if confusion < 0.5:
                label_array[k, start:stop] = 1
    return label_array
