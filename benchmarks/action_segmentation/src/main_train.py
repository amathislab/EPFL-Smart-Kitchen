#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License. 
# A copy is included in LICENSE.
#
import argparse
import ast
import os

import pandas as pd
from dlc2action.project import Project
from get_parameters import get_parameters, load_config
from utils import reformat_params


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DLC2Action Script")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-creation of project folders",
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="verbs_norm",
        help="Type of experiment",
    )
    parser.add_argument(
        "--pose_type",
        type=str,
        default="body",
        choices=[
            "body",
            "hands",
            "body_hands",
            "eyes",
            "body_eyes",
            "body_hands_eyes",
            "holo_hands",
        ],
        help="Type of pose data",
    )
    parser.add_argument(
        "--experiment_number",
        type=int,
        default=1,
        help="Experiment number",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="None",
        choices=[
            "None",
            "videomae_egocentric",
        ],
        help="Chose features to add to training (e.g i3d features, omnivore features, videomae features)",
    )
    parser.add_argument(
        "--hp_search",
        action="store_true",
        help="Run hyperparameter search",
    )
    parser.add_argument(
        "--manual_search",
        action="store_true",
        help="Run hyperparameter search, manual search has priority over hyperparameter search",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for training"
    )

    parser.add_argument(
        "--split_filename",
        type=str,
        default="split_file_85_25_fixed_test.txt",
        help="Split filename",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="c2f_tcn",
        choices=["c2f_tcn", "c2f_transformer", "edtcn", "ms_tcn3", "all"],
        help="Model to train",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run DLC2Action experiments for the ESK dataset.
    The function trains and evaluates the models on the ESK dataset."""

    # Extract arguments
    force = args.force
    experiment_type = args.experiment_type
    pose_type = args.pose_type
    feature_type = args.features
    hp_search = args.hp_search
    manual_search = args.manual_search
    batch_size = args.batch_size
    lr = args.learning_rate
    split_filename = args.split_filename
    hp_search = (
        False if manual_search else hp_search
    )  # Manual search has priority of hyperparameter search

    experiment_number = args.experiment_number
    feat_ext = {
        "None": "",
        "videomae_egocentric": "_ego_videomae",
    }.get(feature_type)
    project_name_base = (
        f"ESK_{experiment_type}_{pose_type}{feat_ext}_{experiment_number}"
    )
    paths = load_config()
    experiment_path = paths["data_path"]

    data_type = "esk_track" if feature_type == "None" else "features"
    annotation_type = "dlc"

    # Create project folder
    project_path = os.path.join(experiment_path, "D2A_projects")
    os.makedirs(project_path, exist_ok=True)

    # Define extensions
    pose_ext = {
        "holo_hands": "_holo_hand_pose",
    }.get(pose_type, "_body_hand_eye_pose")

    norm_ext1 = f"_norm" if "_norm" in experiment_type else ""
    norm_ext2 = f"{pose_ext}_norm" if ("_norm" in experiment_type) else pose_ext
    norm_ext2 = f"{pose_ext}_norm" if ("_norm" in experiment_type) else pose_ext

    if manual_search:
        if "nouns" in experiment_type:
            manual_search_path = "./benchmarks/action_segmentation/searches/manual_search_nouns.csv"
        elif "verbs" in experiment_type:
            manual_search_path = "./benchmarks/action_segmentation/searches/manual_search_verbs.csv"
        else:
            raise NotImplementedError(
                f"Manual search not implemented for this experiment type : {experiment_type}"
            )
        manual_search_df = pd.read_csv(manual_search_path)

    if experiment_type.startswith("batch"):
        batch_path = os.path.join(
            experiment_path,
            f"D2A_converted_label_{experiment_type.split('_norm', maxsplit = 1)[0]}",
        )
        data_paths = [
            os.path.join(batch_path, f"batch_{i}")
            for i in range(len(os.listdir(batch_path)))
        ]
        pose_paths = [
            os.path.join(experiment_path, f"D2A_converted_pose{norm_ext1}")
        ] * len(data_paths)
        project_names = [
            project_name_base + f"_batch_{i}"
            for i in range(len(os.listdir(batch_path)))
        ]
    else:
        label_paths = [
            os.path.join(
                experiment_path,
                f"D2A_converted_label_{experiment_type.split('_norm', maxsplit = 1)[0]}",
            )
        ]
        pose_paths = [os.path.join(experiment_path, f"D2A_converted_pose{norm_ext1}")]
        if "videomae" in feature_type:
            pose_paths = [os.path.join(experiment_path, "videomae_features")]
        project_names = [project_name_base]

    split_fold = "benchmarks/action_segmentation/data"
    for project_name, label_path, pose_path in zip(
        project_names, label_paths, pose_paths
    ):
        if force:
            os.system(f"rm -r {os.path.join(project_path, project_name)}")

        project = Project(
            project_name,
            data_path=pose_path,
            annotation_path=label_path,
            projects_path=project_path,
            data_type=data_type,
            annotation_type=annotation_type,
        )

        param = get_parameters()
        param["general"]["dim"] = 3
        param["general"]["num_cpus"] = 12
        param["training"]["to_ram"] = False
        param["data"]["data_suffix"] = f"{norm_ext2}.h5"
        param["training"]["batch_size"] = batch_size
        param["training"]["lr"] = lr
        param["training"]["split_path"] = os.path.join(split_fold, split_filename)

        # new addition
        param["data"]["keypoint_type"] = pose_type

        if feature_type == "videomae_egocentric":
            param["data"]["feature_suffix"] = f"_only_videomae.npy"
        if pose_type == "eyes":
            param["features"]["keys"] = ["coords"]

        print("Update parameters ... ")
        project.update_parameters(param)

        # Update some parameters
        gpu = args.gpu
        models = args.models
        if models == "all":
            model_names = [
                "c2f_tcn",
                "c2f_transformer",
                "edtcn",
                "ms_tcn3",
            ]
        else:
            model_names = [models]
        for model in model_names:
            if ("body_hands_eyes" in pose_type and experiment_type == "nouns_norm") or (
                "body_hands_eyes" in pose_type and feature_type == "videomae_egocentric"
            ):
                print("Here")
                project.update_parameters(
                    {
                        "features": {
                            "keys": [
                                "coords",
                                "center",
                                "speed_direction",
                                "speed_value",
                                "acc_joints",
                                "angle_speeds",
                            ]
                        }
                    }
                )
            # Hyperparameter search
            search_name = None
            if hp_search:
                search_name = f"search_{model}"
                project.run_default_hyperparameter_search(
                    search_name=search_name,
                    model_name=model,
                    metric="f1",
                    best_n=3,
                    direction="maximize",
                    force=args.force,
                )

            project.update_parameters(
                {
                    "general": {"model_name": model},
                    "training": {"device": f"cuda:{gpu}"},
                }
            )

            if manual_search:
                params = manual_search_df[
                    manual_search_df["Models"] == f"search_{model}"
                ]["best_params"].tolist()[0]
                params = manual_search_df[
                    manual_search_df["Models"] == f"search_{model}"
                ]["best_params"].tolist()[0]
                params = ast.literal_eval(params)
                params = reformat_params(params)
                project.update_parameters(params)

            episode_name = f"exp_{model}_split_{split_filename.split('.')[0]}"

            num_epochs = 150
            project.run_episode(
                episode_name,
                force=True,
                load_search=search_name,
                parameters_update={
                    "training": {"num_epochs": num_epochs},
                },
                n_seeds=1,
            )


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
