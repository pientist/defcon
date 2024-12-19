CUDA_VISIBLE_DEVICES=3,4,5 \
python train.py \
--trial 220 \
--target dest_conceding \
--model spiel \
--in_dim 12 \
--pi_dim 64 \
--rnn_dim 128 \
--rnn_layers 2 \
--n_epochs 100 \
--batch_size 1536 \
--lambda_l1 0.0001 \
--start_lr 0.0001 \
--min_lr 1e-6 \
--print_freq 10 \
--seed 100 \
--cuda