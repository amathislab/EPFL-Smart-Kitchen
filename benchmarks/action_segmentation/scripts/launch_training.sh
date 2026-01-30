#!/bin/bash

for experiment_type in nouns_norm verbs_norm activity_norm; do
    for model in c2f_tcn c2f_transformer ms_tcn3 edtcn; do
        for pose_type in body hands body_hands eyes body_hands_eyes; do
            # Launch training
            python benchmarks/action_segmentation/src/main_train.py \
            --experiment_type $experiment_type \
            --pose_type $pose_type \
            --experiment_number 0001 \
            --batch_size 512 \
            --manual_search \
            --models $model \
            --gpu 0 \
            --force

        done
        # Launch training with videomae features and body_hands_eyes pose type
        python benchmarks/action_segmentation/src/main_train.py \
        --experiment_type $experiment_type \
        --pose_type body_hands_eyes \
        --features videomae_egocentric \
        --experiment_number 0001 \
        --batch_size 512 \
        --manual_search \
        --models $model \
        --gpu 0
        # Launch training with videomae features only
        python benchmarks/action_segmentation/src/main_train.py \
        --experiment_type $experiment_type \
        --pose_type dummy \
        --features videomae_egocentric \
        --experiment_number 0001 \
        --batch_size 512 \
        --manual_search \
        --models $model \
        --gpu 0

    done    
done
