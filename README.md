# GANPOP
Code, dataset, and trained models for "GANPOP: Generative Adversarial Network Prediction of Optical Properties from Single Snapshot Wide-field Images"

If you use this code, please cite:

Chen, Mason T., et al. "GANPOP: Generative Adversarial Network Prediction of Optical Properties from Single Snapshot Wide-field Images." arXiv preprint arXiv:1906.05360 (2019).


<img src="https://github.com/masontchen/GANPOP_Pytorch/blob/master/imgs/Fig_1.jpg" width="512"/> 

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
GANPOP_Pytorch # Path to all the code
└── Datasets # Datasets folder
      └── XYZ_Dataset # Name of your dataset
            ├── test
            └── train
```
<img src="https://github.com/masontchen/GANPOP_Pytorch/blob/master/imgs/Figure2.jpg" width="512"/>

### Training

To train a model:
```
python train.py --dataroot <datapath> --name <experiment_name>  --gpu_ids 0 --display_id 0 
--lambda_L1 60 --niter 100 --niter_decay 100 --pool_size 64 --loadSize 256 --fineSize 256 --gan_mode lsgan --lr 0.0002 --model pix2pix --which_model_netG fusion
```
- To view epoch-wise intermediate training results, `./checkpoints/<experiment_name>/web/index.html`
- `--niter` number of epochs with constant learning rate `--niter_decay` number of epochs with linearly decaying learning rate

<img src="https://github.com/masontchen/GANPOP_Pytorch/blob/master/imgs/Network.jpg" width="512"/> 

### Pre-trained Models

Example pre-trained models for each experiment can be downloaded [here](https://drive.google.com/drive/folders/1Qyh3k0MTiSJqTVIJnZ1KNFERv8NWPkR3?usp=sharing). 
- "AC" and "DC" specify the type of input images, and "corr" stands for profilometry-corrected experiment. 
- These models are all trained on human esophagus samples 1-6, human hands and feet 1-6, and 6 phantoms. 
- Test patches are available under `dataset` folder, including human esophagus 7-8, hands and feet 7-8, 4 ex-vivo pigs, 1 live pig, and 12 phantoms. To validate the models, please save the downloaded subfolders with models under `checkpoints` and follow the directions in the next section ("Testing").

### Testing

To test the model:
```
python test.py --dataroot <datapath> --name <experiment_name> --gpu_ids 0 --display_id 0 
--loadSize 256 --fineSize 256 --model pix2pix --which_model_netG fusion
```
- The test results will be saved to a html file here: `./results/<experiment_name>/test_latest/index.html`.

### Dataset

The full-image dataset can be downloaded [here]. (https://drive.google.com/drive/folders/1o_hIv5xmkO1_jD34Jo6JD0V1kXm5SdiM?usp=sharing)

### Issues

- Please open new threads or report issues to mason@jhmi.edu

## License
© [Durr Lab](https://durr.jhu.edu) - This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
- This code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [FusionNet_Pytorch](https://github.com/GunhoChoi/FusionNet_Pytorch)
* Subsidized computing resources were provided by Google Cloud.

## Reference
If you find our work useful in your research please consider citing our paper:
```
@article{chen2019ganpop,
  title={GANPOP: Generative Adversarial Network Prediction of Optical Properties from Single Snapshot Wide-field Images},
  author={Chen, Mason T and Mahmood, Faisal and Sweer, Jordan A and Durr, Nicholas J},
  journal={arXiv preprint arXiv:1906.05360},
  year={2019}
}
```
