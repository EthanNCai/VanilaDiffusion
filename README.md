# SimplestDiffusion

This repository contains implementations of Diffusion models, using the MNIST dataset as a demonstration dataset.

Implemented components:

* DDPM - Unconditional training and inference code
* DDPM - Conditional training and inference code
* Rectified Flow Matching - Training code
* Rectified Flow Matching - ODE inference code
* Rectified Flow Matching - SDE inference code

\* All implementations are conditional generation unless otherwise specified.

## Environment

```
pip install -r requirements.txt
```

## Dataset

We'll be using MINST, It’s a very small dataset and will be downloaded automatically when you run the code.


## DDPM-Unconditionaled generation

please first cd to the `ddpm` dir

### Training

```
python train.py
```

adjustable params:

```
options:
  -h, --help            show this help message and exit
  --img_size IMG_SIZE   Image size (default: 64)
  --channels CHANNELS   Number of image channels (default: 1)
  --patch_size PATCH_SIZE
                        Patch size for DiT (default: 2)
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Training batch size
  --lr LR               Learning rate
  --seed SEED           Random seed for reproducibility
  --save_dir SAVE_DIR   Directory to save model checkpoints
  --save_freq SAVE_FREQ
                        Save model every N epochs
```

### Infer

```
python infer.py --ckpt /path/to/your/ckpt.pt --output /path/to/your/output.jpg
```

## DDPM-Conditionaled generation
CFG (classifier free guidance) training and sampling + condition injection by cross attention

please first cd to the `ddpm` dir
 
### Training

```
python train_cfg.py
```

Adjustable params:
```
options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Training batch size
  --lr LR               Learning rate
  --img_size IMG_SIZE   Image size (default: 32)
  --patch_size PATCH_SIZE
                        Patch size for ViT (default: 4)
  --save_dir SAVE_DIR   Directory to save model checkpoints
  --seed SEED           Random seed for reproducibility
  --save_freq SAVE_FREQ
                        Save model every N epochs
  --lr_decay_step LR_DECAY_STEP
                        LR decay step size
  --lr_decay_rate LR_DECAY_RATE
                        LR decay rate
```

### Infer

to generate a picture of handwritten "four", use command:

```
python infer_cfg.py --ckpt /path/to/your/ckpt.pt --output /path/to/your/output.jpg --label 4
```


## Flowmatching Training

please first cd to the flowmatching dir

and train by
```
python fm_train.py
```

adjustable params:
```
Train a diffusion model on MNIST dataset

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Training batch size
  --lr LR               Learning rate
  --img_size IMG_SIZE   Image size (default: 32)
  --patch_size PATCH_SIZE
                        Patch size for ViT (default: 4)
  --save_dir SAVE_DIR   Directory to save model checkpoints
  --seed SEED           Random seed for reproducibility
  --save_freq SAVE_FREQ
                        Save model every N epochs
  --lr_decay_step LR_DECAY_STEP
                        LR decay step size
  --lr_decay_rate LR_DECAY_RATE
                        LR decay rate
```

## Flowmatching Infer

__Important note__ : ODE and SDE are differents sampling method, and it's okay to use the same weights trained by fm_train.py 

###  ODE Infer

please first cd to the flowmatching dir

and infer using ode method to generate a picture of handwritten "four" by command:

```
python fm_ode_infer.py --ckpt /path/to/your/ckpt.pt --output /path/to/your/output.jpg --label 4
```

adjustable params:

```
options:
  -h, --help            show this help message and exit
  --ckpt CKPT           Path to the checkpoint file
  --output OUTPUT       Path to save the image
  --guidance_scale GUIDANCE_SCALE
                        CFG guidance scale (default: 3.0)
  --label LABEL         Label to generate (0-9, default: 0)
  --img_size IMG_SIZE   Image size (default: 32)
  --channels CHANNELS   Number of image channels (default: 1)
  --seed SEED           Random seed for reproducibility
  --fm_steps FM_STEPS   steps for flow matching
```


###  SDE Infer

please first cd to the flowmatching dir

and infer using sde method to generate a picture of handwritten "four" by command:

__Important note__: the parameter called `--sde_noise_scale` controls the randomness of generation, if you increase this value, please make sure the parameter `--fm_steps` are increased as well. 

```
python fm_sde_infer.py --ckpt /path/to/your/ckpt.pt --output /path/to/your/output.jpg --label 4
```

adjustable params:
```
options:
  -h, --help            show this help message and exit
  --ckpt CKPT           Path to the checkpoint file
  --output OUTPUT       Path to save the image
  --guidance_scale GUIDANCE_SCALE
                        CFG guidance scale (default: 3.0)
  --label LABEL         Label to generate (0-9, default: 0)
  --img_size IMG_SIZE   Image size (default: 32)
  --channels CHANNELS   Number of image channels (default: 1)
  --seed SEED           Random seed for reproducibility
  --fm_steps FM_STEPS   steps for flow matching
  --sde_noise_scale SDE_NOISE_SCALE
                        SDE noise scale parameter a (default: 0.1)
```


## References

[通俗易懂理解Flow Matching](https://zhuanlan.zhihu.com/p/16113190076)

[扩散模型 | 2.SDE精讲](https://zhuanlan.zhihu.com/p/677154173)
