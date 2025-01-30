CUDA_VISIBLE_DEVICES=3 python search_ddd17.py \
 --batch-size 2 --dataset ddd17 \
 --filter_multiplier 8 --resize 256 --crop_size 321 --is_resize 0 \
 --num_layer 6 --checkname snn_c2b_planb_randomseed --block_multiplier 3 --step 3 \
 --alpha_epoch 10 --epochs 21 \
 --randomseed --num_randomseed 6 --seed 9 \
 --lr 0.005 --initial_channels 5 \
 --timestep 4 --sequence 3 --burning_time 1 \
  --is_allsnn 1 \