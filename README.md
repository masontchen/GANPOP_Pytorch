# GANPOP
GANPOP: Generative Adversarial Network Prediction of Optical Properties from Single Snapshot Wide-field Images

If you use this code, please cite:

Coming soon...

<img src="https://github.com/masontchen/GANPOP_Pytorch/blob/master/imgs/Fig_1.jpg" width="1200"/>   <img src="https://github.com/masontchen/GANPOP_Pytorch/blob/master/imgs/Figure2.jpg" width="1800"/> 

## Setup

### Prerequisites

- Linux (Tested on Ubuntu 16.04)
- NVIDIA GPU (Tested on Nvidia P100 using Google Cloud)
- CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)
- Pytorch>=0.4.0
- torchvision>=0.2.1
- dominate>=2.3.1
- visdom>=0.1.8.3
- scipy

### Dataset Organization

All image pairs must be 256x256 and paired together in 256x512 images. '.png' and '.jpg' files are acceptable. Data needs to be arranged in the following order:

```bash
SOMEPATH # Some arbitrary path
└── Datasets # Datasets folder
      └── XYZ_Dataset # Active dataset
            ├── test
            └── train
```

### Training

To train a model:
```
python train.py --dataroot <datapath> --name GANPOP  --gpu_ids 0 --display_id 0 
--lambda_L1 60 --niter 100 --niter_decay 100 --pool_size 64 --loadSize 256 --fineSize 256 --gan_mode lsgan --lr 0.0002 --model pix2pix --which_netG fusion
```
- To view epoch-wise intermediate training results, `./checkpoints/GANPOP/web/index.html`
- `--niter` number of epochs with constant learning rate `--niter_decay` number of epochs with linearly decaying learning rate

<img src="https://github.com/masontchen/GANPOP_Pytorch/blob/master/imgs/Network.jpg" width="1350"/> 

### Pre-trained Models

Coming soon...

### Testing

To test the model:
```
python test.py --dataroot <datapath> --name GANPOP --gpu_ids 0 --display_id 0 
--loadSize 256 --fineSize 256 --model pix2pix
```
- The test results will be saved to a html file here: `./results/GANPOP/test_latest/index.html`.

### Issues

- Please open new threads or report issues to mason@jhmi.edu

## License
© [Durr Lab](https://durr.jhu.edu) - This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
- This code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [FusionNet_Pytorch] (https://github.com/GunhoChoi/FusionNet_Pytorch)
* Subsidized computing resources were provided by Google Cloud.

## Reference
If you find our work useful in your research please consider citing our paper:
```
Coming soon...
```
