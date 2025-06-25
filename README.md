# SimplestDiffusion

这个仓库包含了Diffusion的代码实现，使用MINST数据集作为演示数据集

实现的内容有：

* DDPM-无条件的训练和推理代码
* DDPM-有条件的训练和推理代码
* rectified flow matching-训练代码
* rectified flow matching-ODE推理代码
* rectified flow matching-SDE推理代码

 \* 没有特别说明的都是有条件生成


## 环境安装

```
pip install -r requirements.txt
```

## MINST数据集

数据集方面不需要特别注意，这是个很小的数据集，在运行代码之后会自动下载。

## DDPM-无条件生成

请首先cd到DDPM目录里

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

## DDPM-有条件生成
原理：CFG训练和采样+cross attention注入条件

请首先cd到DDPM目录里
 
### 训练

```
python train_cfg.py
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


## Flowmatching 训练

to do .....

## Flow Matching ODE 推理

to do .....

## Flow Matching SDE 推理

to do .....



# 其他乱七八糟的笔记

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

### CFG(Classifier free)训练和推理的细节

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

## Flowmatching 笔记


[这是一个好笔记](https://zhuanlan.zhihu.com/p/16113190076)

[这是一个好可视化笔记](https://zhuanlan.zhihu.com/p/677154173)


### Flow matching 训练

Flowmatching的美德在于

* 全部都是确定的东西！！因此没有任何概率建模（例如分布）
* 比DDPM简洁100倍


x是图片，x可以在空间里面移动，假设t时移动的位置是 $x(t)$

并且我们还知道 x(t) 其实是在 x_0 即原图 和 x_1 即完全噪声 之间的一个值（甚至可以看作加权平均）所以可以得到下面这个式子

$$ x(t) = t \times x_1  + (1-t) \times x_0 $$

对其求导就得到了一个超简单的常微分方程

$$
\frac{dx(t)}{dt} = x_1 - x_0
$$

而我们的模型预测的目标就是它，也就是$\frac{dx(t)}{dt}$

所以假设 x_0 到 x_1 是一个线性的过程的话，那么每一步的变化率恒定为 $x_1 - x_0$ 这太棒了，因为每一步的ground truth都是恒定的。

所以loss就可以设计为


$$ MSE(v(x_t,t) , x_1 - x_0 )$$

其中v就表示神经网络用 $x_t$ 和 $t$ 预测出来的"x此时的加速度"，也可以称之为流场， 也可以称之为x(t) 的导数 




### Fun Facts(我猜的)

* ODE 是确定过程，因此其整个过程都不需要概率建模，但是SDE就不一样，引入了随机量之后，每一个状态都变成了一个分布了, 需要大量概率建模，理解难度直线上升。
* ODE，对于同样一个初始化噪声图，降噪出来的结果都一样，对于SDE，同一个噪声图最终会降噪出多样化的结果（这是RL所喜欢的东西）
* 对于SDE，我们需要理解$p_t(x_t)$梯度方向的含义是什么，请看我的纸质笔记,在那个黄色的本子上！