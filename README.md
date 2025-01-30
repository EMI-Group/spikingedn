# SpikingEDN
This code is a demo of our TNNLS 2024 paper "Accurate and Efficient Event-based Semantic Segmentation Using Adaptive Spiking Encoder-Decoder Network".

# Dataset
To proceed, please download the DDD17/DSEC-SEMANTIC dataset on your own.

# Environment
```
1. Python 3.8.*
2. CUDA 10.0
3. PyTorch 
4. TorchVision 
5. fitlog
```

# Install
Create a  virtual environment and activate it.
```shell
conda create -n SpikingEDN python=3.8
conda activate SpikingEDN
```
The code has been tested with PyTorch 1.6 and Cuda 10.2.
```shell
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch
conda install matplotlib path.py tqdm
conda install tensorboard tensorboardX
conda install scipy scikit-image opencv
```

# Code for SpikingEDN
We provide retrain and evaluate code for DDD17/DSEC-SEMANTIC. The best model is provided on '/logs/retrain/retrain_best_model/encoder_best_model'

## Retrain
For retrain procedure, execute: \
  `bash retrain_ddd17.sh`

## Evaluate
For evaluate procedure, execute: \
  `bash evaluate_best_model_ddd17.sh`
  `bash evaluate_dsec.sh`

# Paper Reference
```

@article{zhang2023accurate,
  title={Accurate and efficient event-based semantic segmentation using adaptive spiking encoder-decoder network},
  author={Zhang, Rui and Leng, Luziwei and Che, Kaiwei and Zhang, Hu and Cheng, Jie and Guo, Qinghai and Liao, Jiangxing and Cheng, Ran},
  journal={arXiv preprint arXiv:2304.11857},
  year={2023}
}

```

Our code is developed based on the code from papers "Differentiable hierarchical and surrogate gradient search for spiking neural networks" and "Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation"  

code:  
https://github.com/Huawei-BIC/SpikeDHS
https://github.com/NoamRosenberg/autodeeplab  


## License
This open-source project is not an official Huawei product, and Huawei is not expected to provide support for this project.