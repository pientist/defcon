CUDA_VISIBLE_DEVICES=3 \
CUDA_LAUNCH_BLOCKING=1 \
python train.py \
--trial 02 \
--model gat \
--target oppo_agn_intent \
--min_duration 0.5 \
--possessor_aware \
--keeper_aware \
--drop_opponents \
--sparsify delaunay \
--edge_in_dim 2 \
--node_emb_dim 128 \
--graph_emb_dim 128 \
--gnn_layers 2 \
--gnn_heads 4 \
--skip_conn \
--n_epochs 100 \
--batch_size 512 \
--lambda_l1 0.0001 \
--start_lr 0.002 \
--min_lr 1e-5 \
--print_freq 50 \
--seed 100 \
--cuda