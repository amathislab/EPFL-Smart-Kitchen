#
# Copyright 2025-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under the MIT License. 
# A copy is included in LICENSE.
#
import yaml


def get_parameters():
    dictionary = {
        "general": {
            "model_name": "ms_tcn3",  # str; model name (run project.help("model") for more info)
            "metric_functions": [
                "f1"
            ],  # list; a list of metric names (run project.help("metric") for more info)
            "ignored_clips": None,  # list; a list of string clip ids (agent names) to be ignored
            "len_segment": 50,  # int; the length of segments (in frames) to cut the videos into
            "overlap": 10,  # int; the overlap (in frames) between neighboring segments
            "interactive": False,  # bool; if true, annotations are assigned and features are computed for pairs of clips (animals)
            "exclusive": False,
            "num_cpus": 12,
        },
        "data": {
            "data_suffix": None,  # str; the data suffix (the data files should be named {video_id}{data_suffix}, e.g. video1_suffix.pickle, where video1 is the video is and _suffix.pickle is the suffix
            "feature_suffix": None,  # str; suffix of files with additional features saved as a dictionary (files should be named {video_id}{feature_suffix} and places at data path)
            "annotation_suffix": "_labels.pickle",  # str | set, optional the suffix or the set of suffices such that the annotation files are named {video_id}{annotation_suffix}, e.g, video1_suffix.pickle where video1 is the video id and _suffix.pickle is the suffix
            "canvas_shape": [
                3,
                3,
                3,
            ],  # list; [x, y] size of the canvas where the coordinates are defined
            "ignored_bodyparts": None,  # list; a list of nodypart names to ignore
            "likelihood_threshold": 0,  # float; minimum likelihood to load (bodyparts with lower likelihood will be treated as unknown)
            "behaviors": None,  # list; the list of behaviors to put in the annotation (if null, if will be inferred from the data; !!PLEASE SET IT MANUALLY if different files can have different behavior sets!!)
            "filter_annotated": True,  # bool; if true, the samples that do not have any labels will be filtered
            "filter_background": False,  # bool, if true, only the unlabeled frames that are close to annotated frames will be labeled as background
            "visibility_min_score": 0,  # float, the minimum visibility score for visibility filtering (from 0 ro 1)
            "visibility_min_frac": 0,  # float, the minimum fraction of visible frames for visibility filtering
            "default_agent_name": "individual0",
        },
        "training": {
            "lr": 0.0001,  # float; learning rate
            "device": "cuda:0",  # str; device
            "num_epochs": 10,  # int; number of epochs
            "to_ram": False,  # bool; transfer the dataset to RAM for training (preferred if the dataset fits in working memory)
            "batch_size": 350,  # int; batch size
            "normalize": True,  # bool; if true, normalization statistics will be computed on the training set and applied to all data
            "temporal_subsampling_size": 0.85,  # float; this fraction of frames in each segment is randomly sampled at training time
            "parallel": False,  # bool; if true, the model will be trained on all gpus visible in the system (use os.environ[“CUDA_VISIBLE_DEVICES”] =“{indices}” to exclude gpus in this mode)
            "val_frac": 0.2,  # float; fraction of dataset to use as validation
            "test_frac": 0,  # float; fraction of dataset to use as test
            "partition_method": "file",  # str; the train/test/val partitioning method (for more info run project.help("partition_method"))
            "split_path": "benchmarks/action_segmentation/data/split_file_85_25_fixed_test.txt",
        },
        "losses": {
            "ms_tcn": {
                "focal": True,  # bool; if True, focal loss will be used
                "gamma": 2,  # float; the gamma parameter of focal loss
                "alpha": 0.001,  # float; the weight of consistency loss
            },
        },
        "metrics": {
            "f1": {
                "average": "macro",  # ['macro', 'micro', 'none']; averaging method for classes
                "ignored_classes": "None",  # set; a set of class ids to ignore in calculation
                "threshold_value": 0.5,  # float; the probability threshold for positive samples
            },
            "precision": {
                "average": "macro",  # ['macro', 'micro', 'none']; averaging method for classes
                "ignored_classes": "None",  # set; a set of class ids to ignore in calculation
                "threshold_value": 0.5,  # float; the probability threshold for positive samples
            },
        },
        "model": {
            "num_f_maps": 128,  # int; number of maps
            "feature_dim": None,  # int; if not null, intermediate features are generated with this dimension and then passed to a 2-layer MLP for classification (useful for SSL)
        },
        "features": {
            "keys": [
                "coords",
                "coord_diff",
                "center",
                "intra_distance",
                "speed_direction",
                "speed_value",
                "acc_joints",
                "angle_speeds",
            ],  # set; a list of names of the features to extract (a subset of available keys; run project.help("features") for more info)
            # "keys": ["coords"], # set; a list of names of the features to extract (a subset of available keys; run project.help("features") for more info)
            "averaging_window": 1,  # int; if >1, features are averaged with a moving window of this size (in frames)
            "distance_pairs": None,  # list; a list of bodypart name tuples (e.g. `[("tail", "nose")]`) to compute distances for when `"intra_distance"` is in `keys` (by default all distances are computed)
            "angle_pairs": None,  # list; a list of bodypart name tuples (e.g. `[("ear1", "nose", "ear2")]`) for the angle between `"ear1"--"nose"` and `"nose"--"ear2"` lines) to compute angle cosines for when `"angles"` is in `keys` (by default no angles are computed)
            "zone_vertices": None,  # dict; a dictionary of bodypart name tuples of any length >= 3 that define zones for `"zone_bools"`and `"zone_distances"` features; keys should be zone names and values should be tuples that define the polygons (e.g. `{"main_area": ("x_min", "x_max", "y_max", "y_min"))}`)
            "zone_bools": None,  # list; a list of zone and bodypart name tuples to compute binary identifiers for (1 if an animal is within the polygon or 0 if it's outside) (e.g. `[("main_area", "nose")]`); the zones should be defined in the `zone_vertices` parameter; this is only computed if `"zone_bools"` is in `keys`
            "zone_distances": None,  # list; a list of zone and bodypart name tuples to compute distances for (distance from the bodypart to the closest of the boundaries) (e.g. `[("main_area", "nose")]`); the zones should be defined in the `zone_vertices` parameter; this is only computed if `"zone_distances"` is in `keys`
            "area_vertices": None,  # list; a list of bodypart name tuples of any length >= 3 (e.g. `[("ear1", "nose", "ear2", "spine1")]`) that define polygons to compute areas for when `"areas"` is in `keys` (by default no areas are computed)
        },
        "augmentations": {
            "augmentations": [
                "rotate",
                "switch",
                "add_noise",
                "shift",
                "zoom",
            ],  # set; a set of augmentations (from 'rotate', 'real_lens', 'add_noise', 'shift', 'zoom', 'mirror', 'switch')
            "rotation_limits": [
                -1.57,
                1.57,
            ],  # list; list of rotation angle limits in radians ([low, high])
            "mirror_dim": {
                0
            },  # set; set of dimensions that can be mirrored (0 for x, 1 for y, 2 for z)
            "noise_std": 0.003,  # float; standard deviation of noise
            "zoom_limits": [0.5, 1.5],  # list; list of float zoom limits ([low, high])
            "masking_probability": 0.1,  # float; the probability of masking a joint
        },
    }

    return dictionary


def load_config():
    with open("benchmarks/action_segmentation/config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
    return data
