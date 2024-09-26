# Brownian Bridge Diffusion Models
***
#### [BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models](https://arxiv.org/abs/2205.07680)
https://arxiv.org/abs/2205.07680

**Bo Li, Kai-Tao Xue, Bin Liu, Yu-Kun Lai**

![img](resources/BBDM_architecture.png)

## Requirements
```commandline
conda env create -f environment.yml
conda activate BBDM

# install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Mujoco Installation
wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco200_linux.zip
mkdir -p ~/.mujoco
unzip mujoco200_linux.zip -d ~/.mujoco
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
rm mujoco200_linux.zip

wget https://www.roboti.us/file/mjkey.txt -O ~/.mujoco/mjkey.txt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
# follow https://github.com/openai/mujoco-py/issues/627
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
export CPATH=$CONDA_PREFIX/include
python3 -m pip install patchelf
python3 -m pip install Cython==0.29.36 numpy==1.21.5 mujoco_py==2.0.2.3
#
# ./lib/x86_64-linux-gnu/libGL.so.1
# cd envs/lib then ln -s libOSMesa32.so libOSMesa.so
# solve libGL the same as in the above link

# Design-Bench Installation
python3 -m pip install design-bench==2.0.12
python3 -m pip install robel==0.1.2 morphing_agents==1.5.1 transforms3d --no-dependencies
python3 -m pip install botorch==0.6.4 gpytorch==1.6.0
python3 -m pip install gym==0.12.5

# Download Design-Bench Offline Datasets: 
python3 -m pip install gdown
python3 -m pip uninstall charset-normalizer
python3 -m pip install charset-normalizer
gdown 'https://drive.google.com/uc?id=1_ITQSRrO4SV0EaW2FTfCYryrla2-IXdP'
unzip design_bench_data.zip
rm -rf design_bench_data.zip
mv -v design_bench_data $CONDA_PREFIX/lib/python3.9/site-packages
python3 -m pip install tensorflow==2.11.0
python3 -m pip install wandb
# python3 -m pip uninstall numpy
python3 -m pip install numpy==1.22.4
python3 -m pip install omegaconf
python3 -m pip install einops
export PYTHONPATH=/home/user03/miniconda3/envs/BBDM/lib/python3.9/site-packages/:/cm/shared/apps/jupyter/15.3.0/lib64/python3.9/site-packages/:/cm/shared/apps/jupyter/15.3.0/lib/python3.9/site-packages/

```

## Data preparation
### Paired translation task
For datasets that have paired image data, the path should be formatted as:
```yaml
your_dataset_path/train/A  # training reference
your_dataset_path/train/B  # training ground truth
your_dataset_path/val/A  # validating reference
your_dataset_path/val/B  # validating ground truth
your_dataset_path/test/A  # testing reference
your_dataset_path/test/B  # testing ground truth
```
After that, the dataset configuration should be specified in config file as:
```yaml
dataset_name: 'your_dataset_name'
dataset_type: 'custom_aligned'
dataset_config:
  dataset_path: 'your_dataset_path'
```

### Colorization and Inpainting
For colorization and inpainting tasks, the references may be generated from ground truth. The path should be formatted as:
```yaml
your_dataset_path/train  # training ground truth
your_dataset_path/val  # validating ground truth
your_dataset_path/test  # testing ground truth
```

#### Colorization
For generalization, the gray image and ground truth are all in RGB format in colorization task. You can use our dataset type or implement your own.
```yaml
dataset_name: 'your_dataset_name'
dataset_type: 'custom_colorization or implement_your_dataset_type'
dataset_config:
  dataset_path: 'your_dataset_path'
```

#### Inpainting
We randomly mask 25%-50% of the ground truth. You can use our dataset type or implement your own.
```yaml
dataset_name: 'your_dataset_name'
dataset_type: 'custom_inpainting or implement_your_dataset_type'
dataset_config:
  dataset_path: 'your_dataset_path'
```

## Train and Test
### Specify your configuration file
Modify the configuration file based on our templates in <font color=violet><b>configs/Template-*.yaml</b></font>  
The template of BBDM in pixel space are named <font color=violet><b>Template-BBDM.yaml</b></font> that can be found in **configs/** and <font color=violet><b>Template-LBBDM-f4.yaml Template-LBBDM-f8.yaml Template-LBBDM-f16.yaml</b></font> are templates for latent space BBDM with latent depth of 4/8/16. 

Don't forget to specify your VQGAN checkpoint path and dataset path.
### Specity your training and tesing shell
Specity your shell file based on our templates in <font color=violet><b>configs/Template-shell.sh</b></font>

If you wish to train from the beginning
```commandline
python3 main.py --config configs/Template_LBBDM_f4.yaml --train --sample_at_start --save_top --gpu_ids 0 
```

If you wish to continue training, specify the model checkpoint path and optimizer checkpoint path in the train part.
```commandline
python3 main.py --config configs/Template_LBBDM_f4.yaml --train --sample_at_start --save_top --gpu_ids 0 
--resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt
```

If you wish to sample the whole test dataset to evaluate metrics
```commandline
python3 main.py --config configs/Template_LBBDM_f4.yaml --sample_to_eval --gpu_ids 0 --resume_model path/to/model_ckpt
```

Note that optimizer checkpoint is not needed in test and specifying checkpoint path in commandline has higher priority than specifying in configuration file.

For distributed training, just modify the configuration of **--gpu_ids** with your specified gpus. 
```commandline
python3 main.py --config configs/Template_LBBDM_f4.yaml --sample_to_eval --gpu_ids 0,1,2,3 --resume_model path/to/model_ckpt
```

### Run
```commandline
sh shell/your_shell.sh
```

## Pretrained Models
For simplicity, we re-trained all of the models based on the same VQGAN model from LDM.

The pre-trained VQGAN models provided by LDM can be directly used for all tasks.  
https://github.com/CompVis/latent-diffusion#bibtex

The VQGAN checkpoint VQ-4,8,16 we used are listed as follows and they all can be found in the above LDM mainpage:

VQGAN-4: https://ommer-lab.com/files/latent-diffusion/vq-f4.zip  
VQGAN-8: https://ommer-lab.com/files/latent-diffusion/vq-f8.zip  
VQGAN-16: https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1

All of our models can be found here.
https://pan.baidu.com/s/1xmuAHrBt9rhj7vMu5HIhvA?pwd=hubb

## Acknowledgement
Our code is implemented based on Latent Diffusion Model and VQGAN

[Latent Diffusion Models](https://github.com/CompVis/latent-diffusion#bibtex)  
[VQGAN](https://github.com/CompVis/taming-transformers)

## Citation
```
@inproceedings{li2023bbdm,
  title={BBDM: Image-to-image translation with Brownian bridge diffusion models},
  author={Li, Bo and Xue, Kaitao and Liu, Bin and Lai, Yu-Kun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1952--1961},
  year={2023}
}
```
