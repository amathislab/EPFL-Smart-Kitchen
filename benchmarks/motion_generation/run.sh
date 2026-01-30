python train_vq.py --name FID_rvq_verb --gpu_id 0 --dataset_name kitchen --batch_size 2048 --motion_type verb --num_quantizers 6  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 > FID_kitchen_vq_verb_train.out 2>&1

echo "Finish training VQ"

python eval_t2m_vq.py --gpu_id 0 --name FID_rvq_verb --dataset_name kitchen --batch_size 2048 --motion_type verb --ext rvq_nq6 > FID_kitchen_vq_verb_test.out 2>&1

echo "Finish evaluation VQ"


python train_t2m_transformer.py --name FID_mtrans_verb --gpu_id 0 --dataset_name kitchen --batch_size 256 --motion_type verb --vq_name FID_rvq_verb > FID_kitchen_mtrans_verb_train.out 2>&1

echo "Finish training M Transformer"

python train_res_transformer.py --name FID_rtrans_verb  --gpu_id 0 --dataset_name kitchen --batch_size 256 --motion_type verb --vq_name FID_rvq_verb --cond_drop_prob 0.2 --share_weight > FID_kitchen_rtrans_verb_train.out 2>&1

echo "Finish training R Transformer"

python eval_t2m_trans_res.py --res_name FID_rtrans_verb --dataset_name kitchen --name FID_mtrans_verb --motion_type verb --gpu_id 0 --cond_scale 4 --time_steps 10 --ext evaluation > FID_kitchen_verb_eval_again.out 2>&1

echo "Finish evaluation the trained M Transformer and Residual Transformer"


python eval_t2m_trans_res.py --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw --dataset_name kitchen --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns --motion_type verb --gpu_id 0 --cond_scale 4 --time_steps 10 --ext evaluation > momask_kitchen_verb_eval.out 2>&1