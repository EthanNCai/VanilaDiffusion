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

to do .....

## Flowmatching ODE Infer

to do .....

## Flowmatching SDE Infer

to do .....


## References

[通俗易懂理解Flow Matching](https://zhuanlan.zhihu.com/p/16113190076)

[扩散模型 | 2.SDE精讲](https://zhuanlan.zhihu.com/p/677154173)
