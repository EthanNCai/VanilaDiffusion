# VanillaDiffusion

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

```

### 推理

```
```


## B-使用CrossAttention注入条件的有条件DiT噪声预测模型 (推理使用Classifier free guidance)

 
### 训练




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
# 传入 x0 和 前向时间步序号 t， 得到x0 在 t时刻的加噪图
def noise_image(self, x0, t):
    ...
    return xt, noise

```
这段代码对应的公式是：
给定干净图像 $ x_0 $ 和时间步 $ t $，我们通过下式得到加噪图像 $ x_t $：

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中：

- $ \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s $，是到当前时间步的累计保留率（噪声扩散控制系数）
- $ \epsilon $ 是标准正态分布的高斯噪声

#### 对于后向过程

```python
# 传入 xt 和 前向时间步序号 t，以及预测的这一步添加的噪声。得到由xt降噪而来的xt-1的噪声图
def sample_prev_image_distribution(self, x_t, t, pred_noise):
    ...
    return x_t-1
```
这个过程对应前向过程采样公式

我们要从条件分布 $ p_\theta(x_{t-1} \mid x_t) $ 中采样：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right) + \sigma_t \cdot z
$$

其中：

- $ \alpha_t = 1 - \beta_t $
- $ \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s $，即累计保留率
- $ \epsilon_\theta(x_t, t) $ 是预测的噪声
- $ z \sim \mathcal{N}(0, I) $，是标准正态分布噪声
- $ \sigma_t = \sqrt{\beta_t} $，是采样时添加的噪声系数
- 若 $ t = 0 $，则不添加噪声项：$ z = 0 $


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

__简单来说：__ 在DiT的每一个Transfromer block里面，我们都用此时的隐变量 $dim$ 作为 $Query$， 然后 $c$ 作为 $Key$ 和 $Value$， 这样做一个Cross-Attention，就能把 $c$ 注入隐变量。

__细节来说：__ 在每一个 Transformer Block 中，DiT 执行如下 Cross-Attention 操作：

设：

- $ \mathbf{z} \in \mathbb{R}^{N \times d} $：图像的隐藏变量（Query）
- $ \mathbf{c} \in \mathbb{R}^{M \times d} $：条件信息（Key 和 Value）
- $ W_Q, W_K, W_V \in \mathbb{R}^{d \times d} $：投影矩阵

则：

$$
Q = \mathbf{z} W_Q \\
K = \mathbf{c} W_K \\
V = \mathbf{c} W_V \\
\text{Attention}(\mathbf{z}, \mathbf{c}) = \text{softmax} \left( \frac{Q K^\top}{\sqrt{d}} \right) V
$$

最终的输出是（下面这一步是残差连接）：

$$
\mathbf{z}' = \mathbf{z} + \text{Attention}(\mathbf{z}, \mathbf{c})
$$

其中：

- $ \text{softmax} \left( \frac{Q K^\top}{\sqrt{d}} \right) $：计算 Query 与 Key 的相似度
- Residual 连接保证模型训练稳定