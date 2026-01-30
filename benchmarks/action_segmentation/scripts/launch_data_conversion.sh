#bin/bash!

# Convert pose data
for pose_mode in body_hand_eye_pose holo_hand_pose; do
    python benchmarks/action_segmentation/data/convert2d2a.py --experiment_type verbs_norm --pose_mode "$pose_mode" --no_annotations
done

# Convert action annotation data
python benchmarks/action_segmentation/data/convert2d2a.py --experiment_type nouns_norm --no_pose
python benchmarks/action_segmentation/data/convert2d2a.py --experiment_type verbs_norm --no_pose
# Convert activity annotation data
python benchmarks/action_segmentation/data/convert2d2a_activity.py 