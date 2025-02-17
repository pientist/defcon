CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python train.py \
--trial 50 \
--model gat \
--target scoring \
--xy_only \
--possessor_aware \
--keeper_aware \
--ball_z_aware \
--poss_vel_aware \
--use_xg \
--residual \
--sparsify delaunay \
--edge_in_dim 2 \
--node_emb_dim 128 \
--graph_emb_dim 128 \
--gnn_layers 2 \
--gnn_heads 4 \
--skip_conn \
--n_epochs 200 \
--batch_size 512 \
--lambda_l1 1e-6 \
--start_lr 0.0001 \
--min_lr 1e-5 \
--print_freq 50 \
--seed 100 \
--cuda