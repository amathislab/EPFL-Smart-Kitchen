#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License. 
# A copy is included in LICENSE.
#
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from benchmarks.action_segmentation.src.get_parameters import load_config


def convert_data(filepath: str, output_folder: str, fps: int = 30):
    """
    Convert the annotation data from JSON format to a format compatible with D2A.
    The function reads the JSON file, extracts the relevant information, and saves it in a D2A pickle file.
    """

    filename = (
        os.path.basename(os.path.dirname(filepath))
        + "_"
        + os.path.basename(filepath).split(".")[0]
    )
    infos = {
        "datetime": str(datetime.now()),
        "annotator": "andy",
        "video_file": filename + ".mp4",
    }

    data = pd.read_json(filepath)
    data = data["annotations"].values.tolist()
    data = pd.DataFrame(data)
    data["start"] = data["start"].apply(lambda x: x * fps).astype(int)
    data["end"] = data["end"].apply(lambda x: x * fps).astype(int)
    data["Activities"] = data["Activities"].apply(
        lambda x: x + "_annot" if x == "other" else x
    )
    act_list_ind = data["Activities"].unique()

    beh_data = []
    for activity in act_list_ind:
        data_act = data[data["Activities"] == activity]
        lab_arr = data_act[["start", "end"]].values
        lab_arr = np.concatenate(
            (lab_arr, np.zeros((lab_arr.shape[0], 1), dtype=int)), axis=1
        )
        beh_data.append(lab_arr.astype(np.int32))

    formated_data = (
        infos,
        act_list_ind,
        [filename],
        [beh_data],
    )

    pickle.dump(
        formated_data,
        open(
            os.path.join(output_folder, f"{filename}_activities.pickle"),
            "wb",
        ),
    )


def get_paths(annot_dir: str):
    """ "
    "Get all the paths to the annotation files in the given directory."
    """
    paths = []
    for p in os.listdir(annot_dir):
        if not os.path.isdir(os.path.join(annot_dir, p)):
            continue
        for s in os.listdir(os.path.join(annot_dir, p)):
            activity_path = os.path.join(
                annot_dir, p, s, "annotations", "activity_annotations.json"
            )
            if os.path.exists(activity_path):
                paths.append(os.path.join(annot_dir, p, s))
    return paths


if __name__ == "__main__":

    splits = ["train", "test"]
    configs = load_config()
    output_path = configs["data_path"]
    for split in splits:
        original_data_path = os.path.join(configs["original_data_path"], split)
        paths = get_paths(original_data_path)
        output_folder_label = os.path.join(output_path, "D2A_converted_label_activity")
        os.makedirs(output_folder_label, exist_ok=True)
        for path in paths:
            convert_data(path, output_folder_label)
