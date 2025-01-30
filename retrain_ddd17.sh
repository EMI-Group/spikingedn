# alif thresh 0.1-1 internal 0.1  
# retrain pure event best result
CUDA_VISIBLE_DEVICES=0 python retrain_evaluate_ddd17.py \
--net_arch logs/retrain/retrain_best_model/encoder_best_model/network_path_space.npy \
--cell_arch logs/retrain/retrain_best_model/encoder_best_model/genotype.npy \
--net_path logs/retrain/retrain_best_model/encoder_best_model/network_path.npy \
--dataset ddd17 --batch_size 2 --burning_time 1 --sequence 5 --resize 256 --crop_size 321 --block_multiplier 3 --step 3 \
--exp exp1_dvs_alif_0_5_test \
--filter_multiplier 32 --epochs 101 \
--num_layer 6  --timestep 6 --initial_channels 5 \
--base_lr 0.001 --is_resize 0 --aps_channel 0 \


# retrain frame + event best result  (100 epoch)
CUDA_VISIBLE_DEVICES=0 python retrain_evaluate_ddd17.py \
--net_arch logs/retrain/retrain_best_model/encoder_best_model/network_path_space.npy \
--cell_arch logs/retrain/retrain_best_model/encoder_best_model/genotype.npy \
--net_path logs/retrain/retrain_best_model/encoder_best_model/network_path.npy \
--dataset ddd17_images --batch_size 2 --burning_time 1 --sequence 3 --resize 256 --crop_size 321 --block_multiplier 3 --step 3 \
--exp exp2_ssam \
--filter_multiplier 32 --epochs 101 \
--num_layer 6  --timestep 4 --initial_channels 6 \
--base_lr 0.001 --is_resize 0 --is_allsnn 1 --aps_channel 1 \
--spade_type 1 --spade_snn 1 \
--multi_gamma 0 --spade_bn 0 \
--spade_v3_type 12 --h_channel 64 \
--SSAM_type 3 --tau_SSAM 0 \