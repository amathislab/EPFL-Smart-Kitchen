#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License.
# A copy is included in LICENSE.
#

# import seaborn as sns
import argparse
import ast
import os
import os.path as osp
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dlc2action.project import Project
from get_parameters import get_parameters, load_config
from utils import binarize_data, load_pickle, reformat_params

MODEL_LIST = [
    "c2f_tcn",
    "c2f_transformer",
    "edtcn",
    "ms_tcn3",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DLC2Action models on ESK dataset"
    )
    parser.add_argument(
        "--project_name", type=str, required=True, help="Name of the project"
    )
    parser.add_argument(
        "--run_predictions", action="store_true", help="Run predictions"
    )
    parser.add_argument("--plot_pred", action="store_true", help="Plot predictions")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate predictions")
    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--average",
        type=str,
        default="macro",
        choices=["macro", "micro"],
        help="Averaging method for metrics",
    )
    parser.add_argument(
        "--split_filename",
        type=str,
        default="split_file_85_25_fixed_test.txt",
        help="Split file name",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Model(s) to use, comma separated or 'all'",
    )
    parser.add_argument(
        "--load_pretrained",
        action="store_true",
        help="Whether to load pretrained weights, specific DLC2Action projects will be created to load the pretrained weights",
    )
    return parser.parse_args()


def plot_ethograms(
    data: dict,
    behaviors: list,
    start: int = 0,
    end: int = -1,
    mode: str = "prediction",
    cmap: str = "Oranges",
    save: str = None,
    individual: str = None,
    plot_all: bool = False,
) -> None:
    """Plot ethograms from start to end time (in frames), mode can be prediction or ground truth depending on the data format."""
    if mode == "prediction":
        assert (
            individual is not None
        ), "Individual name must be provided for prediction mode"
        best_pred = data[individual].numpy() > 0.5  # Threshold the predictions
    elif mode == "ground_truth":
        best_pred = binarize_data(data, max_frame=end)
    else:
        raise NotImplementedError
    end = best_pred.shape[1] if end < 0 else end
    indices = np.where(np.sum(best_pred[:, start:end], axis=1) > 0)[0]
    if plot_all:
        indices = np.arange(best_pred.shape[0])
    f, ax = plt.subplots(figsize=(12, 5 + len(indices)))
    ax.imshow(
        best_pred[indices, start:end], aspect="auto", cmap=cmap, interpolation="nearest"
    )
    labels = [behaviors[i] for i in indices]
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Ethograms")
    ax.set_xticks(
        np.linspace(0, end - start, 6), np.linspace(start, end, 6).astype(np.int32)
    )
    # ax.set_xlim(start, end)

    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()


def save_results(
    results: dict, project_path: str, project_name: str, mode: str
) -> None:
    """Save the evaluation results to a csv file."""
    output_path = f"{project_path}/{project_name}/results/evaluation_results_{mode}.csv"
    if osp.exists(output_path):
        df = pd.read_csv(output_path)
        df.update(pd.DataFrame(results))
    else:
        df = pd.DataFrame(results)
    df.to_csv(output_path)
    print(results)


def prepare_pretrained_project(
    project_name: str,
    data_path: str,
    data_type: str,
    annotation_type: str,
    episode_names: list,
    split_path: str,
    pose_ext:str,
    gpu: int = 0,
) -> Project:
    """Prepare a DLC2Action Project for loading pretrained weights."""

    "ESK_verbs_norm_body_hands_eyes_ego_videomae_205"
    experiment_type = project_name.split("_norm")[0].split("ESK_")[1]
    pose_type = "_".join(project_name.split(experiment_type + "_norm" + "_")[-1].split("_")[:-1])

    project_path = osp.join(data_path, "D2A_projects_pretrained")
    checkpoint_path = osp.join(data_path, "D2A_checkpoints")
    data_path_pose = osp.join(data_path, "D2A_converted_pose_norm")
    annotation_path = osp.join(data_path, f"D2A_converted_label_{experiment_type}")

    project = Project(
        project_name,
        data_path=data_path_pose,
        annotation_path=annotation_path,
        projects_path=project_path,
        data_type=data_type,
        annotation_type=annotation_type,
    )

    # copy checkpoints into episode folder
    for episode_name in episode_names:
        model_name = [i for i in MODEL_LIST if i in episode_name][0]
        src_checkpoint = osp.join(
            checkpoint_path,
            experiment_type,
            pose_type,
            f"{model_name}.pt",
        )
        

        params = get_pretrained_params(
                pose_ext, split_path, pose_type, project_name
            )
        project.update_parameters(params)
        project.update_parameters(
            {
                "general": {"model_name": model_name},
                "training": {"device": f"cuda:{gpu}"},
            }
        )

        if "nouns" in experiment_type:
            manual_search_path = "./benchmarks/action_segmentation/searches/manual_search_nouns.csv"
        elif "verbs" in experiment_type:
            manual_search_path = "./benchmarks/action_segmentation/searches/manual_search_verbs.csv"
        elif "activity" in experiment_type:
            manual_search_path = "./benchmarks/action_segmentation/searches/manual_search_activity.csv"

        else:
            raise NotImplementedError(
                f"Manual search not implemented for this experiment type : {experiment_type}"
            )
        manual_search_df = pd.read_csv(manual_search_path)
        params = manual_search_df[
            manual_search_df["Models"] == f"search_{model_name}"
        ]["best_params"].tolist()[0]
        params = manual_search_df[
            manual_search_df["Models"] == f"search_{model_name}"
        ]["best_params"].tolist()[0]
        params = ast.literal_eval(params)
        params = reformat_params(params)
        project.update_parameters(params)

        project.run_episode(
            episode_name=episode_name,
            load_search=None,
            parameters_update={
                "training": {"device": f"cuda:{gpu}", "num_epochs": 1}
            },
            force=True,
            n_seeds=1,
        )

        dst_checkpoint_dir = osp.join(
            project.project_path, "results", "model", episode_name
        )
        os.makedirs(dst_checkpoint_dir, exist_ok=True)
        dst_checkpoint = osp.join(dst_checkpoint_dir, f"{model_name}.pt")
        if not osp.exists(dst_checkpoint):
            shutil.copy(src_checkpoint, dst_checkpoint)
            print(f"Copied checkpoint from {src_checkpoint} to {dst_checkpoint}")

        # Modify the meta data to relate to the checkpoint path
        meta_path = osp.join(project.project_path, "meta", f"episodes.pickle")
        meta_data = pd.read_pickle(meta_path)
        if episode_name in meta_data.index:
            training = meta_data["training"]
            training.loc[episode_name, "checkpoint_path"] = dst_checkpoint
            meta_data["training"] = training
        else:
            raise ValueError(f"Episode {episode_name} not found in meta data. Something went wrong with the 1epoch run.")

    return project


def get_pretrained_params(pose_ext, split_path, pose_type, project_name) -> dict:

    param = get_parameters()
    param["general"]["dim"] = 3
    param["general"]["num_cpus"] = 12
    param["training"]["to_ram"] = False
    param["data"]["data_suffix"] = f"{pose_ext}_norm.h5"
    param["training"]["batch_size"] = 256
    param["training"]["split_path"] = split_path

    # new addition
    param["data"]["keypoint_type"] = pose_type

    if "videomae" in project_name:
        param["data"]["feature_suffix"] = f"_only_videomae.npy"
    if pose_type == "eyes":
        param["features"]["keys"] = ["coords"]

    print("Update parameters ... ")
    return param


def main(
    project_name: str,
    run_predictions: bool = True,
    plot_pred: bool = True,
    evaluate: bool = True,
    mode: str = "val",
    average: str = "macro",
    split_filename: str = "split_file_85_25_fixed_test.txt",
    gpu: int = 0,
    models: str = "all",
    load_pretrained: bool = False,
) -> None:
    """Run DLC2Action experiments for the ESK dataset.
    The function makes predictions and evaluates the models on the ESK dataset."""

    config = load_config()

    save_path = osp.join("projects", project_name)
    plot_path = osp.join(save_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    label_path = osp.join(config["data_path"], "D2A_converted_label_verbs_filt")
    project_path = osp.join(config["data_path"], "D2A_projects") if not load_pretrained else osp.join(
        config["data_path"], "D2A_projects_pretrained"
    )

    data_type = "esk_track" if "videomae" not in project_name else "features"
    annotation_type = "dlc"
    batch_size = 1
    # Update some parameters
    if models == "all":
        model_names = MODEL_LIST
    else:
        model_names = [models]

    if osp.isdir(osp.join(project_path, project_name)) or load_pretrained:

        pose_type = "_".join(project_name.split("_")[3:-1])
        pose_ext = {
            "holo_hands": "_holo_hand_pose",
        }.get(pose_type, "_body_hand_eye_pose")

        split_dir = "benchmarks/action_segmentation/data"
        split_path = osp.join(split_dir, split_filename)
    

        episode_names = [
            f"exp_{model}_split_{split_filename.split('.')[0]}" for model in model_names
        ]

        if load_pretrained:
            project =  prepare_pretrained_project(
                project_name,
                config["data_path"],
                data_type,
                annotation_type,
                episode_names,
                split_path,
                pose_ext,
            )
        else:
            project = Project(
                project_name,
                data_path=None,
                annotation_path=None,
                projects_path=project_path,
                data_type=data_type,
                annotation_type=annotation_type,
            )


        results = {}
        for episode_name in episode_names:
            # try:
            prediction_name = (
                f"prediction_{episode_name}_{mode}_{split_filename.split('.')[0]}"
            )
            if run_predictions:
                project.update_parameters(
                    {
                        "training": {
                            "split_path": split_path,
                        },
                    }
                )
                project.run_prediction(
                    prediction_name=prediction_name,
                    parameters_update={
                        "training": {
                            "device": f"cuda:{gpu}",
                            "batch_size": batch_size,
                            "split_path": split_path,
                        },
                        "data": {
                            "data_suffix": (pose_ext + "_norm.h5"),
                            "feature_suffix": (
                                "_only_videomae.npy"
                                if "videomae" in project_name
                                else None
                            ),
                        },
                    },
                    episode_names=[episode_name],
                    augment_n=0,
                    mode=mode,
                    force=True,
                )
            path = osp.join(
                project.project_path, "results", "predictions", prediction_name
            )
            prediction_paths = [osp.join(path, name) for name in os.listdir(path)]

            # Load predictions
            for prediction_path in prediction_paths:
                predictions = load_pickle(prediction_path)
                behaviors = project.get_behavior_dictionary(episode_name)
                # Plot predictions

                if plot_pred:
                    individual = osp.basename(prediction_path).split("_prediction")[0]
                    plot_ethograms(
                        predictions,
                        behaviors,
                        individual=individual,
                        start=25000,
                        end=30000,
                        mode="prediction",
                        save=osp.join(
                            plot_path,
                            osp.basename(prediction_path) + "_plot.png",
                        ),
                        plot_all=True,
                    )
                    # Load labels
                    ground_truth_path = osp.join(
                        label_path, f"{individual}_labels.pickle"
                    )
                    ground_truth = load_pickle(ground_truth_path)
                    behaviors = ground_truth[1]
                    # Plot labels
                    plot_ethograms(
                        ground_truth,
                        behaviors,
                        individual=individual,
                        start=25000,
                        end=30000,
                        mode="ground_truth",
                        save=osp.join(plot_path, f"{individual}_groundtruth"),
                        plot_all=True,
                    )
            if evaluate:
                if average == "macro":
                    metric_functions = [
                        "f1",
                        "recall",
                        "precision",
                        "segmental_f1",
                        "edit_distance",
                    ]
                    params = {"general": {"metric_functions": metric_functions}, "training": {"batch_size": batch_size}}

                else:
                    metric_functions = ["f1", "recall", "precision", "segmental_f1"]
                    params = {
                        "general": {"metric_functions": metric_functions},
                        "metrics": {i: {"average": average} for i in metric_functions},
                        "training": {"batch_size": batch_size},
                    }

                results[episode_name] = project.evaluate_prediction(
                    prediction_name=f"prediction_{episode_name}_{mode}_{split_filename.split('.')[0]}",
                    mode=mode,
                    parameters_update=params,
                )[1]
                if average == "micro":
                    results[episode_name] = {
                        k: [t.item() for t in v]
                        for k, v in results[episode_name].items()
                    }
        if evaluate:
            save_results(
                results, project_path, project_name, "_".join([models, mode, average])
            )

    else:
        raise FileNotFoundError(f"Project {project_name} not found in {project_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        project_name=args.project_name,
        run_predictions=args.run_predictions,
        plot_pred=args.plot_pred,
        evaluate=args.evaluate,
        mode=args.mode,
        average=args.average,
        split_filename=args.split_filename,
        gpu=args.gpu,
        models=args.models,
        load_pretrained=args.load_pretrained,
    )
