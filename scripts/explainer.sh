CUDA_VISIBLE_DEVICES=3 \
CUDA_LAUNCH_BLOCKING=1 \
python train_explainer.py \
--model_id 16 \
--trial 11 \
--explainer_type pgexplainer \
--explanation_type model \
--n_matches 20 \
--n_epochs 30 \
--batch_size 512 \
--lr 0.002 \
--print_freq 5 \
--seed 100