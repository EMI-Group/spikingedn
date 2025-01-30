# best results event only  51.39% miou
CUDA_VISIBLE_DEVICES=7 python evaluate_ddd17.py \
--net_arch logs/retrain/retrain_best_model/encoder_best_model/network_path_space.npy \
--cell_arch logs/retrain/retrain_best_model/encoder_best_model/genotype.npy \
--net_path logs/retrain/retrain_best_model/encoder_best_model/network_path.npy \
--dataset ddd17 --batch_size 1 --burning_time 1 --sequence 3 --resize 256 --crop_size 321 --block_multiplier 3 --step 3 \
--exp exp1_alif_vth0_5 \
--filter_multiplier 32 --epochs 101 \
--num_layer 6  --timestep 4 --initial_channels 5 \
--base_lr 0.001 --is_resize 0 --is_allsnn 1 --aps_channel 0 \
# --resume /media/HDD2/personal_files/zhangrui/retrain_data/data/deeplab_autodeeplab_ddd17_v3_exp13_tau0_1_L6_e100_lr3_f32_all_snn_convbr_RS485_epoch20_dvs_6class_stem0_111_alif_vth0_5_xu_epoch41.pth


# best results ssam (e+f)  72.57% miou
CUDA_VISIBLE_DEVICES=1 python evaluate_ddd17.py \
--net_arch logs/retrain/retrain_best_model/encoder_best_model/network_path_space.npy \
--cell_arch logs/retrain/retrain_best_model/encoder_best_model/genotype.npy \
--net_path logs/retrain/retrain_best_model/encoder_best_model/network_path.npy \
--dataset ddd17_images --batch_size 1 --burning_time 1 --sequence 3 --resize 256 --crop_size 321 --block_multiplier 3 --step 3 \
--exp exp2_SSAM_alif \
--filter_multiplier 32 --epochs 101 \
--num_layer 6  --timestep 4 --initial_channels 6 \
--base_lr 0.001 --is_resize 0 --is_allsnn 1 --aps_channel 1 \
--spade_type 1 --spade_snn 1 \
--multi_gamma 0 --spade_bn 0 \
--spade_v3_type 12 --h_channel 64 \
--SSAM_type 3 --tau_SSAM 0 \
