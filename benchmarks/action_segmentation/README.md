# EPFL-Smart-Kitchen: Action segmentation benchmark

## Installation

![DLC2Action](../../media/DLC2Action.png)

[DLC2Action](https://github.com/amathislab/DLC2action) is a python-based toolbox to for skeleton-based action segmentation tasks. You can install DLC2Action by running the following:

```
pip install dlc2action
```

## Data preparation


### With HuggingFace
The action segmentation benchmark is available on [HuggingFace](https://huggingface.co/datasets/amathislab/ESK_action_segmentation). Therefore you can easily download the DLC2Action-compatible data using the following script:
```
bash benchmarks/action_segmnetation/scripts/download_from_hf.sh [DATA_DIR]
```
Please modify the `data_path` to the datasets in `config.yaml` before training.

### From scratch
1. Download the original data from Zenodo (see [download table](../../README.md#-find-our-datasets))
2. Modify the `original_data_path` in the `config.yaml` to where you downloaded the data, and `data_path` to where you would like the processed data to be.
3. Convert the dataset to DLC2Action format by running the conversion script
    ```
    bash benchmarks/action_segmentation/scripts/launch_data_conversion.sh
    ```
 
## Usage

### Training
All models from the benchmark can be trained using the training script
```
bash benchmarks/action_segmentation/scripts/launch_training.sh
```
Specifically, this runs `benchmarks/action_segmentation/src/main_train.py` which has the following parameters:
```
python benchmarks/action_segmentation/src/main_train.py \
    --experiment_type [EXPERIMENT TYPE] \ # Type of annotations (verbs_norm, nouns_norm,, activity_norm)
    --pose_type [POSE MODALITY] \ # Input type (body, hands, eyes, or any combination (e.g body_hands, body_hands_eyes))
    --features videomae_egocentric \ # Set for using videomae features or remove
    --experiment_number 0001 \ # Project identification
    --batch_size 512 \ # Batch size
    --manual_search \ # Set to use hyperparameter previously found in benchmarks/action_segmentation/searches
    --models [MODEL] \ # Model to use or "all"
    --gpu 0 \ # GPU ID
```

### Evaluation
All models from the benchmark can be evaluated using the evaluation script
```
bash benchmarks/action_segmentation/scripts/launch_evaluate.sh
```

Specifically, this runs `benchmarks/action_segmentation/src/main_evaluate.py` which has the following parameters:
```
python benchmarks/action_segmentation/src/main_evaluate.py \
    --project_name "ESK_[EXPERIMENT TYPE]_[POSE MODALITY]_[VIDEOMAE EXT]_0001"
    --mode [TEST/VALIDATION] \ # whether to use the test set or the validation set
    --average [MACRO/MICRO] \# whether to average over classes or not
    --models [MODEL] \ # Model to use or "all"
    --gpu 0 \ # GPU ID
    --run_predictions \ # Set to run predictions
    --evaluate \ # Set to run evaluation 
```

## Reference 

```
@article {kozlova2025DLC2Action,
	author = {Kozlova, Elizaveta and Bonnetto, Andy and Mathis, Alexander},
	title = {DLC2Action: A Deep Learning-based Toolbox for Automated Behavior Segmentation},
	elocation-id = {2025.09.27.678941},
	year = {2025},
	doi = {10.1101/2025.09.27.678941},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/09/28/2025.09.27.678941},
	eprint = {https://www.biorxiv.org/content/early/2025/09/28/2025.09.27.678941.full.pdf},
	journal = {bioRxiv}
}
```

