# best dsec_images  57.77
CUDA_VISIBLE_DEVICES=0 python evaluate_dsec_all_snn.py \
--net_arch logs/retrain/retrain_best_model/encoder_best_model/network_path_space.npy \
--cell_arch logs/retrain/retrain_best_model/encoder_best_model/genotype.npy \
--net_path logs/retrain/retrain_best_model/encoder_best_model/network_path.npy \
--dataset dsec_images --batch_size 1 --burning_time 1 --sequence 3 --resize 256 --crop_size 321 --block_multiplier 3 --step 3 \
--exp dsec_images_exp4_SSAM_alif \
--filter_multiplier 32 --epochs 101 \
--num_layer 6  --timestep 4 --initial_channels 8 \
--base_lr 0.001 --is_resize 0 --is_allsnn 1 --aps_channel 3 \
--spade_type 1 --spade_snn 1 \
--multi_gamma 0 --spade_bn 0 \
--spade_v3_type 12 --h_channel 64 \
--SSAM_type 3 --tau_SSAM 0.2 \
# 100th epoch is the best

# best dsec  53.04
CUDA_VISIBLE_DEVICES=7 python evaluate_dsec_all_snn.py \
--net_arch logs/retrain/retrain_best_model/encoder_best_model/network_path_space.npy \
--cell_arch logs/retrain/retrain_best_model/encoder_best_model/genotype.npy \
--net_path logs/retrain/retrain_best_model/encoder_best_model/network_path.npy \
--dataset dsec --batch_size 1 --burning_time 1 --sequence 5 --resize 256 --crop_size 321 --block_multiplier 3 --step 3 \
--exp dsec_exp3_alif_vth0_3 \
--filter_multiplier 32 --epochs 102 \
--num_layer 6  --timestep 6 --initial_channels 5 \
--base_lr 0.001 --is_resize 0 --is_allsnn 1 --aps_channel 0 \
# 100th epoch is the best