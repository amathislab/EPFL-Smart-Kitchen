#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License. 
# A copy is included in LICENSE.
#
import argparse
import ast
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmarks.action_segmentation.src.get_parameters import load_config
from benchmarks.action_segmentation.src.utils import get_keypoint_info
from tqdm import tqdm

BODY_WO_ARM_IND = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16]
NUM_EYE_GAZES = 10


def parse_arguments() -> argparse.Namespace:
    """Parse arguments for the script"""
    parser = argparse.ArgumentParser(description="Convert action and pose data")
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="verbs_norm",
        help="Type of experiment",
    )
    parser.add_argument(
        "--pose_mode",
        type=str,
        default="body_hand_pose",
        choices=[
            "body_hand_eye_pose",
            "holo_hand_pose",
        ],
        help="Mode for input data type",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for splitting the data",
    )
    parser.add_argument(
        "--no_pose",
        action="store_true",
        help="Flag to indicate if pose data should be converted",
    )
    parser.add_argument(
        "--no_annotations",
        action="store_true",
        help="Flag to indicate if behavior data should be converted",
    )
    parser.add_argument(
        "--no_filtering",
        action="store_true",
        help="Flag to indicate if filtering should be applied",
    )
    parser.add_argument(
        "--dummy_pose",
        action="store_true",
        help="Flag to create dummy pose data instead of using real pose data",
    )
    parser.add_argument(
        "--no_age",
        action="store_true",
        help="Flag to indicate if age should be included",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Split to use for the conversion (train or test)",
    )
    arguments = parser.parse_args()
    return arguments


def load_json(path: str) -> pd.DataFrame:
    """Load json file and return as pandas dataframe"""
    data = pd.read_json(path)
    return data


def load_excel(path: str) -> pd.DataFrame:
    """Load excel file and return as pandas dataframe"""
    return pd.read_excel(path)


def load_csv(path: str) -> pd.DataFrame:
    """Load csv file and return as pandas dataframe"""
    return pd.read_csv(path)

    # Function implementation goes here


def get_overall_likelihood(kpt_conf: np.ndarray, l2_dist: np.ndarray, mode: str):
    """Get overall likelihood for each kpt"""
    kpt_conf = [kpt_conf.replace("nan", "0") for kpt_conf in kpt_conf]
    kpt_conf = np.array(list(map(ast.literal_eval, kpt_conf)))
    l2_treshold = {"body": 0.09, "hand": 0.06}.get(mode)
    l2_dist = l2_dist < l2_treshold
    if mode == "body":
        final_conf = kpt_conf * l2_dist[:, None]
    elif mode == "hand":
        final_conf = np.concatenate(
            [
                kpt_conf[:, :21] * l2_dist[:, 0][:, None],
                kpt_conf[:, 21:] * l2_dist[:, 1][:, None],
            ],
            axis=1,
        )
    return final_conf


def get_pose_array(
    input_pose: np.array, kpt_conf, l2_dist, num_keypoints: int
) -> np.array:
    """Convert pose data from string to numpy array
    inputs:
        input_pose : list of strings
        num_keypoints : int; number of keypoints
    outputs:
        arr_pose : numpy array of shape (num_frames, num_keypoints, 4)"""
    mode = "body" if num_keypoints == 17 else "hand"
    overall_likelihood = get_overall_likelihood(kpt_conf, l2_dist, mode)
    input_pose = [pose.replace("nan", "80000000") for pose in input_pose]
    arr_pose = list(map(ast.literal_eval, input_pose))
    # arr_pose = list(map(np.fromstring, input_pose))
    arr_pose = np.concatenate(arr_pose, axis=0)
    arr_pose[np.abs(arr_pose) > 100000] = np.nan
    # arr_pose = np.concatenate([arr_pose, np.ones((arr_pose.shape[0], 1))], axis=1)
    arr_pose = np.concatenate([arr_pose, overall_likelihood.reshape(-1, 1)], axis=1)
    arr_pose = np.reshape(arr_pose, (-1, num_keypoints, arr_pose.shape[-1]))
    return arr_pose


def save_dataframe_pose(filename, keypoint_name, pose_data_val, output_path):
    columnindex = pd.MultiIndex.from_product(
        [
            ["andy"],
            [filename],
            keypoint_name,
            ["x", "y", "z", "likelihood"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df_list = pd.DataFrame(pose_data_val, columns=columnindex)
    df_list.to_hdf(output_path, key="tracks", format="table", mode="w")


def get_pose_data_val(
    body_data, hands_data, eye_data, hololens_pose_data_val, mode="body_pose"
):
    """Get the pose data for the given mode"""
    if mode == "hand_pose" or mode == "centered_hand_pose":
        return hands_data
    elif mode == "body_pose":
        return body_data
    elif mode == "body_hand_pose":
        return np.concatenate((body_data, hands_data), axis=1)
    elif mode == "eye_pose":
        return eye_data
    elif mode == "body_eye_pose":
        return np.concatenate((body_data, eye_data), axis=1)
    elif mode == "hand_eye_pose":
        return np.concatenate((hands_data, eye_data), axis=1)
    elif mode == "body_hand_eye_pose":
        return np.concatenate((body_data, hands_data, eye_data), axis=1)
    elif mode == "holo_hand_pose":
        return hololens_pose_data_val
    elif mode == "body_wo_arm":
        # Remove arms
        return body_data[:, BODY_WO_ARM_IND, :]


def convert_and_save(
    action_data: pd.DataFrame,
    body_pose_data: pd.DataFrame,
    filename: str,
    output_folder_label: str,
    output_folder_pose: str,
    action_list: list,
    recipe: str,
    hand_pose_data: pd.DataFrame = None,
    eye_pose_data: pd.DataFrame = None,
    hololens_pose_data: pd.DataFrame = None,
    mode: str = "body_pose",
    only_verbs: bool = False,
    only_nouns: bool = False,
    batch_size: int = None,
    norm: bool = False,
    convert_pose: bool = True,
    convert_annotations: bool = True,
    age_data: pd.DataFrame = None,
    num_frames_age: int = None,
) -> None:
    """
    Convert action data and pose data and save as pickle file and DLC file respectively.

    Args:
        action_data (pd.DataFrame): Dataframe containing action data
        body_pose_data (pd.DataFrame): Dataframe containing body pose data
        filename (str): Name of the file
        output_folder_label (str): Output folder for action data
        output_folder_pose (str): Output folder for pose data
        action_list (list): List of actions
        recipe (str): Recipe name
        hand_pose_data (pd.DataFrame): Dataframe containing hand pose data
        eye_pose_data (pd.DataFrame): Dataframe containing eye pose data
        hololens_pose_data (pd.DataFrame): Dataframe containing hololens pose data
        mode (str): Mode for input data type
        only_verbs (bool): Flag to indicate if only verbs should be considered
        only_nouns (bool): Flag to indicate if only nouns should be considered
        batch_size (int): Batch size for splitting the data
        norm (bool): Flag to indicate if data should be reprojected
        convert_pose (bool): Flag to indicate if pose data should be converted
        convert_annotations (bool): Flag to indicate if action data should be converted
    Returns:
        None
    """

    if convert_annotations:
        assert not (only_verbs and only_nouns)
        # Convert Behavior data
        action_data["Start"] = (action_data["Start"].values * 30).astype(np.int32)
        action_data["End"] = (action_data["End"].values * 30).astype(np.int32)

    indices = np.arange(len(action_list))
    if batch_size is not None:
        indices = np.split(indices, np.where(indices % batch_size == 0)[0][1:], axis=0)
    else:
        indices = indices[np.newaxis, :]

    action_list = np.array(list(action_list))
    for k, ind in enumerate(indices):
        if convert_annotations:

            beh_data = []
            action_list_ind = action_list[ind]
            for action in action_list_ind:
                v_n = action.split("__")
                if len(v_n) == 1:
                    verb = v_n[0]
                    noun = ""
                elif len(v_n) == 2:
                    verb = v_n[0]
                    noun = v_n[1]
                else:
                    raise ValueError("WTH")

                act_seq = action_data[action_data["Verbs"] == verb]
                act_seq.loc[:, "Confusion"] = (
                    act_seq.loc[:, "Confusion"].fillna(1).astype(np.float32).values
                )
                if not (only_verbs or only_nouns):
                    act_seq = act_seq[act_seq["Nouns"] == noun]
                if only_nouns:
                    act_seq = action_data[action_data["Nouns"] == noun]
                beh_data.append(
                    np.array(
                        [act_seq["Start"], act_seq["End"], act_seq["Confusion"]]
                    ).T.astype(np.int32)
                )

            if only_verbs:
                action_list_ind = np.array(
                    [action.split("__")[0] for action in action_list_ind]
                )
            elif only_nouns:
                action_list_ind = np.array(
                    [action.split("__")[1] for action in action_list_ind]
                )

            new_beh_data = beh_data
            if only_nouns or only_verbs:
                action_list_ind, indices_new = np.unique(
                    action_list_ind, return_inverse=True
                )
                new_beh_data = []
                for k in range(len(action_list_ind)):
                    new_beh_data.append(
                        np.concatenate(
                            [beh_data[a] for a, i in enumerate(indices_new) if i == k],
                            axis=0,
                        )
                    )

            # Filter repetitions
            new_beh_data = [np.unique(action, axis=0) for action in new_beh_data]

            infos = {
                "datetime": str(datetime.now()),
                "annotator": "andy",
                "video_file": filename,
                "recipe": recipe,
            }

            if not age_data is None:
                # assert not num_frames_age is None, "Number of frames for age is not provided"
                participant = filename.split("_")[0]
                age = int(
                    age_data[age_data["Folder name"] == participant]["Age"].values[0]
                )
                # action_list_ind = np.append(action_list_ind, "age")
                # new_beh_data.append(np.array([[0, num_frames_age, 1]], dtype = np.int32))
                infos["age"] = age

            formated_data = (
                infos,
                action_list_ind,
                [filename],
                [new_beh_data],
            )  # Add lists for number of actions and number of individuals

            if batch_size is None:
                pickle.dump(
                    formated_data,
                    open(
                        os.path.join(output_folder_label, f"{filename}_labels.pickle"),
                        "wb",
                    ),
                )
            else:
                output_folder_batch = os.path.join(output_folder_label, f"batch_{k}")
                os.makedirs(output_folder_batch, exist_ok=True)
                pickle.dump(
                    formated_data,
                    open(
                        os.path.join(output_folder_batch, f"{filename}_labels.pickle"),
                        "wb",
                    ),
                )

        # Convert pose data
        if convert_pose:
            body_info, hand_info, both_info, eye_info, holo_info = get_keypoint_info(
                mode=mode
            )
            switcher = {
                "body_hand_eye_pose": (lambda d: d.update(eye_info) or d)(
                    both_info.copy()
                ),
                "holo_hand_pose": holo_info,
            }
            keypoint_name = [kpt["name"] for kpt in switcher.get(mode).values()]
            num_keypoints = len(keypoint_name)
            norm_ext = "_norm" if norm else ""
            if batch_size is None:
                output_path = os.path.join(
                    output_folder_pose, f"{filename}_{mode}{norm_ext}.h5"
                )
            else:
                output_path = os.path.join(
                    output_folder_pose, f"batch_{k}", f"{filename}_{mode}{norm_ext}.h5"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if not os.path.exists(output_path):
                # frame_id = pose_data["rgb_frameid"]
                body_pose_data_val = body_pose_data["kp3ds"].values
                body_conf = body_pose_data["kp3ds_conf"].values
                body_l2 = body_pose_data["l2_dist"].values
                body_pose_data_val = get_pose_array(
                    body_pose_data_val, body_conf, body_l2, len(body_info)
                )
                hands_pose_data_val = None
                eye_pose_data_val = None
                hololens_pose_data_val = None
                if "hand" in mode:
                    hands_pose_data_val = hand_pose_data["kp3ds"].values
                    hand_conf = hand_pose_data["kp3ds_conf"].values
                    hand_l2_left = hand_pose_data["l2_dist_left"].values[:, None]
                    hand_l2_right = hand_pose_data["l2_dist_right"].values[:, None]
                    hand_l2 = np.concatenate([hand_l2_left, hand_l2_right], axis=1)
                    hands_pose_data_val = get_pose_array(
                        hands_pose_data_val, hand_conf, hand_l2, len(hand_info)
                    )
                if "eye" in mode:
                    eye_pose_data_val = eye_pose_data["eyes"]
                    eye_pose_data_val = np.array(
                        list(map(ast.literal_eval, eye_pose_data_val))
                    )
                    ref_pose = np.mean(body_pose_data_val[:, 1:3, :], axis=1)
                    # eye_pose_data_val = ref_pose + eye_pose_data_val[:, 3:7]
                    # eye_pose_data_val = eye_pose_data_val[:, np.newaxis, :]
                    eye_pose_data_val = np.concatenate(
                        [
                            ref_pose[:, None, :]
                            + i * eye_pose_data_val[:, 3:7][:, None, :]
                            for i in range(NUM_EYE_GAZES)
                        ],
                        axis=1,
                    )

                if "holo_hand_pose" in mode:
                    hololens_pose_data_val = []
                    for i, h in enumerate(["hololefts", "holorights"]):
                        hpd = hololens_pose_data[h].values
                        hpd = np.array(list(map(ast.literal_eval, hpd)))
                        if not i:
                            offset_pose = hpd[:, 0, :3]  # Remove the wrist offset
                            ref_pose = body_pose_data_val[
                                :, 9, :3
                            ]  # take the body wrist as reference
                        hpd[:, :, :3] = (
                            hpd[:, :, :3]
                            - np.repeat(
                                offset_pose[:, np.newaxis, :], hpd.shape[1], axis=1
                            )
                            + np.repeat(
                                ref_pose[:, np.newaxis, :], hpd.shape[1], axis=1
                            )
                        )
                        hololens_pose_data_val.append(hpd)
                    hololens_pose_data_val = np.concatenate(
                        hololens_pose_data_val, axis=1
                    )

                pose_data_val = get_pose_data_val(
                    body_pose_data_val,
                    hands_pose_data_val,
                    eye_pose_data_val,
                    hololens_pose_data_val,
                    mode=mode,
                )
                if norm:
                    pose_data_val = normalize_data(body_pose_data_val, pose_data_val)

                pose_data_val = np.reshape(
                    pose_data_val, (-1, num_keypoints * pose_data_val.shape[-1])
                )
                save_dataframe_pose(filename, keypoint_name, pose_data_val, output_path)


def generate_video(poses: np.ndarray, output_dir: str, output_file: str = "tmp.mp4"):
    """Poses for N frames"""
    os.makedirs(output_dir, exist_ok=True)
    for frame, pose in tqdm(enumerate(poses), total=len(poses)):
        plot_pose(pose, output_path=os.path.join(output_dir, f"img_{frame:03}.png"))
    os.system(
        f"ffmpeg -y -framerate 30 -i {output_dir}/img_%03d.png -c:v libx264 -r 30 {output_file}"
    )


def plot_pose(pose, output_path="temp.png"):
    """Pose at frame t"""
    f = plt.figure()
    ax = f.add_subplot(projection="3d")
    colors = plt.cm.turbo(np.linspace(0, 1, pose.shape[0]))
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=colors)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.savefig(output_path)
    plt.close()


def normalize_data(pose_data: np.array, pose_to_apply_to: np.array) -> np.array:
    """reproject and rescale pose data
    inputs:
        pose_data : (num_frames, num_body_parts, 4)"""
    assert len(pose_data.shape) == 3
    pose = pose_data[:, :, :3]
    pose_app = pose_to_apply_to[:, :, :3]
    num_frames, _, num_dim = pose.shape
    num_frames_app, num_body_parts_app, _ = pose_app.shape

    likelihood = pose_to_apply_to[:, :, -1]
    head_ind = [1, 2, 3, 4, 5, 6] + [11, 12]
    right_ind = [2, 4, 6, 12]
    left_ind = [1, 3, 5, 11]

    # Get center of mass
    center_of_mass = np.mean(pose, axis=1)
    # Get head position
    head_pos = np.mean(pose[:, head_ind, :], axis=1)
    # Get First vector
    ortho_0 = (head_pos.flatten() - center_of_mass.flatten()).reshape(
        (num_frames, num_dim)
    )
    # Get right and left position
    left_pos = np.mean(pose[:, left_ind, :], axis=1)
    right_pos = np.mean(pose[:, right_ind, :], axis=1)
    # Get Second vector
    vec_1 = (right_pos.flatten() - left_pos.flatten()).reshape((num_frames, num_dim))
    ortho_1 = np.cross(ortho_0, vec_1, axis=1)
    ortho_2 = np.cross(ortho_0, ortho_1, axis=1)
    # Normalize vectors
    norm_vec = []
    for i, vec in zip(range(3), [ortho_1, ortho_2, ortho_0]):
        norm_vec.append(vec / np.linalg.norm(vec, axis=1)[:, None])
    norm_vec.append(center_of_mass)
    norm_vec = np.swapaxes(np.array(norm_vec), 0, 1)
    norm_vec = np.swapaxes(
        np.concatenate(
            (norm_vec, np.array([[0, 0, 0, 1]] * num_frames)[:, :, None]), axis=2
        ),
        1,
        2,
    )
    # Take inverse
    M = fast_inverse(norm_vec)

    # Apply transform
    transform_pose = np.swapaxes(
        np.array(
            list(
                map(
                    np.dot,
                    M,
                    np.swapaxes(
                        np.concatenate(
                            (
                                pose_app,
                                np.ones((num_frames_app, num_body_parts_app, 1)),
                            ),
                            axis=-1,
                        ),
                        1,
                        2,
                    ),
                )
            )
        ),
        1,
        2,
    )
    transform_pose[:, :, -1] = likelihood
    return transform_pose


def fast_inverse(A: np.array) -> np.array:
    """Compute inverse of a batch of matrices"""
    identity = np.identity(A.shape[2], dtype=A.dtype)
    return np.array([np.linalg.solve(x, identity) for x in A])


def get_action_list(path: str, recipe_list: list) -> dict:
    """Find list of actions accross participants for a given recipe"""
    action_list = dict.fromkeys(np.unique(list(recipe_list.keys())))

    for key in action_list.keys():
        voc = []
        for session_name in recipe_list[key]:
            participant = session_name.split("_")[0]
            session = "_".join(session_name.split("_")[1:])
            data = load_excel(
                # os.path.join(path, participant, session, "annotations_basic_AI.xlsx")
                os.path.join(
                    path,
                    participant,
                    session,
                    "annotations",
                    "actions_annotations.xlsx",
                )
            )
            data["Nouns"] = data["Nouns"].fillna(" ")
            sess_vocabulary = (
                data["Verbs"] + np.array(["__"] * len(data)) + data["Nouns"]
            ).tolist()
            voc.append(np.unique(sess_vocabulary))
        action_list[key] = set().union(*voc)

    return action_list


def get_recipe_list(path: str) -> dict:
    recipes = ["Ratatouille", "Omelet", "Pad_Thai", "Risotto"]
    recipe_list = {r: [] for r in recipes}
    for participant in os.listdir(path):
        if participant in [".", ".."] or participant.endswith(".py"):
            continue
        for session in os.listdir(os.path.join(path, participant)):
            if session in [".", ".."]:
                continue
            filename = os.path.join(
                path, participant, session, "annotations", "actions_annotations.xlsx"
            )
            if os.path.exists(filename):
                excel_data = load_excel(filename)
                found = False
                for recipe in [
                    "Ratatouille",
                    "Omelet",
                    "Pad_Thai",
                    "Risotto",
                ]:
                    if recipe in excel_data["Nouns"].values:
                        recipe_list[recipe].append(participant + "_" + session)
                        found = True
                    elif recipe == "Pad_Thai":
                        if "Tofu" in excel_data["Nouns"].values:
                            recipe_list[recipe].append(participant + "_" + session)
                            found = True
                    elif recipe == "Omelet":
                        if (
                            "Eggs" in excel_data["Nouns"].values
                            and "Tomatoes" in excel_data["Nouns"].values
                        ):
                            recipe_list[recipe].append(participant + "_" + session)
                            found = True
                if not found:
                    print(
                        "Recipe not found for: ",
                        participant + "_" + os.path.splitext(session)[0],
                    )
                    exit()
    return recipe_list


def create_dummy_pose(
    body_pose_data: pd.DataFrame,
    filename: str,
    output_folder_pose: str,
    num_keypoints: int = 3,
):
    num_frames = body_pose_data["kp3ds"].values.shape[0]
    # dummy_arr = np.ones((num_frames, num_keypoints, 4))
    dummy_arr = np.random.rand(num_frames, num_keypoints, 4)
    dummy_arr[:, :, :3] = dummy_arr[:, :, :3] * 0.8
    dummy_arr[:, :, 3] = 0
    dummy_arr = np.reshape(dummy_arr, (dummy_arr.shape[0], -1))
    keypoint_name = [f"kpt_{i}" for i in range(num_keypoints)]
    output_path = os.path.join(output_folder_pose, f"{filename}_dummy_pose.h5")
    save_dataframe_pose(filename, keypoint_name, dummy_arr, output_path)


if __name__ == "__main__":

    args = parse_arguments()
    experiment_type = args.experiment_type
    pose_mode = args.pose_mode
    batch_size = args.batch_size
    do_pose = not args.no_pose
    add_age = not args.no_age
    do_annotations = not args.no_annotations
    filtering = not args.no_filtering
    dummy_pose = args.dummy_pose
    split = args.split

    pose_mode = "body_pose" if dummy_pose else pose_mode

    if experiment_type.startswith("batch"):
        batch_size = int(experiment_type.split("_")[1])
    only_nouns = experiment_type.startswith("nouns")
    only_verbs = experiment_type.startswith("verbs")
    norm = experiment_type.endswith("norm")
    norm_ext = "norm" if norm else ""

    # Get recipes
    paths = load_config()

    age_data = None
    num_frames_age = None
    if add_age:
        age_path = paths["age_path"]
        age_data = load_csv(age_path)

    filtered_list_actions = None
    filt_ext = ""
    if filtering:
        filtered_list_actions = paths["list_of_actions"]
        filt_ext = "_filt"

    path_to_data = os.path.join(paths["original_data_path"], split)

    output_folder_label = os.path.join(
        paths["data_path"],
        f"D2A_converted_label_{experiment_type.split('_norm', maxsplit=1)[0]}{filt_ext}",
    )
    output_folder_pose = os.path.join(
        paths["data_path"], f"D2A_converted_pose_{norm_ext}"
    )

    os.makedirs(output_folder_label, exist_ok=True)
    os.makedirs(output_folder_pose, exist_ok=True)

    recipe_list = get_recipe_list(path_to_data)
    action_list = get_action_list(path_to_data, recipe_list)

    for recipe in recipe_list.keys():
        action_list_recipe = action_list[recipe]
        if not filtered_list_actions is None:
            action_list_recipe = [
                value for value in action_list_recipe if value in filtered_list_actions
            ]
        for session_name in tqdm(recipe_list[recipe]):
            participant = session_name.split("_")[0]
            session = "_".join(session_name.split("_")[1:])
            excel_data = (
                load_excel(
                    os.path.join(
                        path_to_data,
                        participant,
                        session,
                        "annotations",
                        "actions_annotation.xlsx",
                    )
                )
                if do_annotations
                else None
            )
            body_pose_data = None
            hand_pose_data = None
            eye_pose_data = None
            holohand_pose_data = None
            convert_pose = do_pose

            # Get number of frames from pose for age
            if num_frames_age is None and not age_data is None:
                body_pose_path = os.path.join(
                    path_to_data,
                    participant,
                    session,
                    "pose_3d",
                    "pose3d_smpl.csv",
                )
                body_pose_data = load_csv(body_pose_path)
                num_frames_age = len(body_pose_data)
                body_pose_data = None

            if convert_pose:
                body_pose_path = os.path.join(
                    path_to_data,
                    participant,
                    session,
                    "pose_3d",
                    "pose3d_smpl.csv",
                )
                hand_pose_path = os.path.join(
                    path_to_data,
                    participant,
                    session,
                    "pose_3d",
                    "pose3d_mano.csv",
                )

                body_pose_data = (
                    load_csv(body_pose_path) if os.path.exists(body_pose_path) else None
                )
                if "hand" in pose_mode and not body_pose_data is None:
                    hand_pose_data = (
                        load_csv(hand_pose_path)
                        if os.path.exists(hand_pose_path)
                        else None
                    )
                if "eye" in pose_mode and not body_pose_data is None:
                    eye_pose_path = os.path.join(
                        path_to_data,
                        participant,
                        session,
                        "meta_data",
                        "holo_data_wpose.csv",
                    )
                    eye_pose_data = None
                    eye_pose_data = load_csv(eye_pose_path)

                if pose_mode == "holo_hand_pose" and not body_pose_data is None:
                    holo_hand_path = os.path.join(
                        path_to_data,
                        participant,
                        session,
                        "meta_data",
                        "holo_data_wpose.csv",
                    )
                    holohand_pose_data = (
                        load_csv(holo_hand_path)
                        if os.path.exists(holo_hand_path)
                        else None
                    )

            convert_pose = False if body_pose_data is None else convert_pose
            convert_pose = (
                False
                if (hand_pose_data is None) and ("hand" in pose_mode)
                else convert_pose
            )
            convert_pose = (
                False
                if (eye_pose_data is None) and ("eye" in pose_mode)
                else convert_pose
            )
            convert_pose = (
                False
                if (holohand_pose_data is None) and ("holo_hand" in pose_mode)
                else convert_pose
            )
            filename = participant + "_" + session
            if dummy_pose:
                if convert_pose:
                    create_dummy_pose(body_pose_data, filename, output_folder_pose)
                else:
                    print(f"No pose file for {participant} {session}")
            else:
                convert_and_save(
                    excel_data,
                    body_pose_data,
                    filename,
                    output_folder_label,
                    output_folder_pose,
                    action_list_recipe,
                    recipe,
                    hand_pose_data=hand_pose_data,
                    eye_pose_data=eye_pose_data,
                    hololens_pose_data=holohand_pose_data,
                    only_verbs=only_verbs,
                    only_nouns=only_nouns,
                    batch_size=batch_size,
                    norm=norm,
                    convert_pose=convert_pose,
                    convert_annotations=do_annotations,
                    mode=pose_mode,
                    age_data=age_data,
                    num_frames_age=num_frames_age,
                )
