#!/bin/bash
for experiment_type in nouns_norm verbs_norm activity_norm; do
    for model in c2f_tcn c2f_transformer ms_tcn3 edtcn; do
        # Launch evaluation for each pose type
        for pose_type in body hands body_hands eyes body_hands_eyes; do
            project_name="ESK_${experiment_type}_${pose_type}_0001"
            python benchmarks/action_segmentation/src/main_evaluate.py \
                --project_name $project_name \
                --mode test \
                --average macro \
                --models $model \
                --gpu 0 \
                --run_predictions \
                --evaluate \
                --load_pretrained \

            done
        # # Launch evaluation with videomae features and body_hands_eyes pose type
        python benchmarks/action_segmentation/src/main_evaluate.py \
            --project_name "ESK_${experiment_type}_body_hands_eyes_ego_videomae_0001" \
            --mode test \
            --average macro \
            --models $model \
            --gpu 0 \
            --run_predictions \
            --load_pretrained \
            --evaluate \

        # # Launch evaluation with videomae features only
        python benchmarks/action_segmentation/src/main_evaluate.py \
            --project_name "ESK_${experiment_type}_dummy_ego_videomae_0001" \
            --mode test \
            --average macro \
            --models $model \
            --gpu 0 \
            --run_predictions \
            --evaluate \
            --load_pretrained \
    
    done
done




