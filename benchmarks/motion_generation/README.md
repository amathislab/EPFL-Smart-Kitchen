# 🎭 EPFL-Smart-Kitchen: Motion Generation Benchmark

Welcome to the motion generation benchmark for the EPFL-Smart-Kitchen dataset! This benchmark provides a comprehensive framework for evaluating text-to-motion generation models on naturalistic cooking activities captured in the EPFL-Smart-Kitchen.

## 📋 Overview

This codebase enables you to reproduce the results from the motion generation benchmark presented in our paper. We leverage state-of-the-art motion generation models including MARDM, T2M-GPT, and MoMask, adapted for our dense 3D pose annotations and semantic cooking action descriptions.

### ✨ Key Features

- 🎯 **Text-to-motion generation** for cooking activities
- 📊 **Benchmark evaluation scripts** for standardized comparison
- 🔄 **Multiple baseline models** (MARDM, T2M-GPT, MoMask)
- 📈 **Comprehensive metrics** and evaluation tools
- 🤸 **Full-body motion sequences** with hand and body kinematics

## 🚀 Quick Start

### Dataset preparation

Download the EPFL-Smart-Kitchen action recognition dataset from Hugging Face:

```bash
bash benchmarks/motion_generation/download_from_hf.sh
```

## 📦 What’s in this folder
This directory contains split archives for motion data and (optionally) pretrained checkpoints:

- `motion_data.z01`, `motion_data.z02`, …: multipart archive with motion training/eval data
- `checkpoints.z01` (optional): multipart archive with example pretrained weights

You’ll first reconstruct and unzip these archives locally. And you can see the following folders:
```
ESK_motion_generation
├── motion_data
|   ├── new_joint_vecs
|   └── holo_images
├── evaluators
|   ├── Comp_v6_KLD005
|   └── Decomp_SP001_SM001_H512
|   └── text_mot_match_[TEXT_TYPE]
├── checkpoints
|   ├── fullbody_[TOKENIZER]
|   └── fullbody_[BASELINE]
|   ├── fullbody_image_[TOKENIZER]
|   └── fullbody_image_[BASELINE]
└── README.md
```

### 🔓 Extract the archives (Linux)

```bash
# Reconstruct and extract motion data
cat motion_data.z* > motion_data.zip
unzip motion_data.zip -d motion_data

unzip evaluators.zip -d evaluators

# (Optional) Reconstruct and extract pretrained checkpoints
cat checkpoints.z* > checkpoints.zip
unzip checkpoints.zip -d checkpoints
```


### 📦 Installation

Install the required packages with the following command:

```bash
pip install -r requirements.txt
```

### 🛠️ Training and Evaluation

All results can be reproduced by running the following script:

```bash
bash run.sh
```

The script includes:
- Data preprocessing pipelines
- Training scripts with optimized hyperparameters
- Evaluation and inference code
- Metric computation for motion quality assessment

## 🏆 Results

Our benchmark evaluates motion generation quality using standard metrics including FID, diversity, and motion-text alignment scores. For detailed results, please refer to our paper.

## 🙏 Acknowledgements

We sincerely thank the authors of [MARDM](https://github.com/ChenFengYe/motion-latent-diffusion), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), and [MoMask](https://github.com/EricGuo5513/momask-codes) for open-sourcing their code, which forms the foundation of our motion generation pipeline.

### 📚 Important Notes

Note that our code depends on other libraries, including:
- [SMPL](https://smpl.is.tue.mpg.de/)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)

Each of these libraries has its own respective license that must also be followed.