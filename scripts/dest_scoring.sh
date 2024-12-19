CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
--trial 430 \
--target dest_scoring \
--model spiel \
--seq_len 30 \
--pi_dim 64 \
--rnn_dim 128 \
--rnn_layers 2 \
--n_epochs 100 \
--batch_size 256 \
--lambda_l1 0.0001 \
--start_lr 0.0001 \
--min_lr 1e-6 \
--print_freq 50 \
--seed 100 \
--cuda