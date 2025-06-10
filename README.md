# SimplestDiffusion

这个仓库包含了Diffusion的代码实现，使用MINST数据集作为演示数据集

实现的内容有：

* 一个Diffusion噪声调度器
* A- __最基本的无条件DiT噪声预测模型__
* B- __使用CrossAttention注入条件的有条件DiT噪声预测模型 (推理使用Classifier free guidance)__

## 环境安装

```
pip install -r requirements.txt
```

## MINST数据集

数据集方面不需要特别注意，这是个很小的数据集，在运行代码之后会自动下载。

## A-最基本的无条件DiT噪声预测模型
 
### 训练

```
python train.py
```

可以调整的参数有

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

### 推理

```
python infer.py --ckpt /path/to/your/ckpt.pt --output /path/to/your/output.jpg
```

## B-使用CrossAttention注入条件的有条件DiT噪声预测模型 (推理使用Classifier free guidance)

 
### 训练

单卡训练
```
python train_cfg.py
```

多卡训练
```
torchrun --nproc_per_node=1 --master_port=12355 train_cfg_ddp.py
```

可调整的参数有
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

### 推理

生成一张4的手写图片，可以使用命令

```
python infer_cfg.py --ckpt /path/to/your/ckpt.pt --output /path/to/your/output.jpg --label 4
```

## 额外的笔记

### 噪声调度器

噪声调度器主要实现两个函数，对应着扩散中的前向和后向过程。

#### 对于前向过程

```python
'''
设
原图（也可以说是：t=0时刻的加噪图）= x0
t时刻的加噪图 = xt
前向时间步序号 = t 

这里我们实现的函数就可以表示为:
xt = f(x0, t）
即下面的这个函数
'''
def noise_image(self, x0, t):
    ...
    return xt, noise

```


#### 对于后向过程

```python
'''
设
t时刻的加噪图 = xt 
t-1时刻的加噪图 = x_{t-1}
前向时间步序号 = t 
预测的t时间步加入的噪声 = e，

这里我们实现的函数就可以表示为:
x_{t-1} = f(xt, t, e）
即下面的这个函数
'''
def sample_prev_image_distribution(self, x_t, t, pred_noise):
    ...
    return x_t-1
```

### CFG(Classifier free)训练和推理

#### cfg的训练

cfg 的训练很简单，我们只要保证一件事情：
> 模型既有无条件生成能力，也要有有条件生成能力。

我们达成这个目的的方法是：在训练的时候随机把准备注入扩散过程的条件 张量 c 随机drop掉。

#### cfg的推理

cfg推理的情况下，对于条件c，我们在每一步降噪时同时做无条件降噪和有条件降噪。这样可以获得两个预测的噪声。

通过这两个噪声，使无条件噪声“指向”有条件噪声，就可以获得一个根据条件c引导的降噪方向。最终获得一个融合有条件和无条件的最终预测噪声。


代码非常简单，如下：

```python
# 分别推理一次有条件和无条件
pred_noise_uncond = model(x_t, t_tensor, uncond_condition)
pred_noise_cond = model(x_t, t_tensor, condition)

# CFG 融合
pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
```
## Cross Attention是怎么把条件c融合进扩散的？

假设我们有一个隐tensor h，Shape 为 (B, N1, dim)
然后我们有一个条件tensor c, Shape为 (B, N2, dim)
我们可以用h来做为query，c作为key 和 value
那么最终将会输出一个形状为(B,N1,dim)的tensor hc，

经过注意力计算后，h中的每个query向量会基于c中的key和value计算加权平均，可以看作，h已经被c注入了一些（条件）信息。