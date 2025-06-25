## By Junzhi, Cai

import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class NoiseScheduler():

    def __init__(self,
                 timesteps=300,
                 alpha_start=0.0001,
                 alpha_end=0.035,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        # 首先计算出来一些可以被提前计算的东西

        self.betas = torch.linspace(alpha_start, alpha_end, timesteps)
        self.alphas = 1. - self.betas.to(device)
        self.sqrt_alphas = torch.sqrt(self.alphas).to(device)
        self.sqrt_one_minus_alphas = torch.sqrt(1. - self.alphas).to(device)
        
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)
        self.timesteps = timesteps
        
    def sample_prev_image_distribution(self, x_t, t, pred_noise):
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        
        # 支持 batch 的 t（确保是长整型）
        if t.dim() == 0:
            t = t[None]
        
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)              # α_t
        beta_t = (1.0 - alpha_t)                                # β_t
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)  # ȧ_t = ∏ α_t
        eps = 1e-8

        # 去噪部分：预测 x_{t-1} 的均值 μ
        x_mean = (1. / torch.sqrt(alpha_t + eps)) * (
            x_t - ((beta_t / torch.sqrt(1 - alpha_bar_t + eps)) * pred_noise)
        )

        # 加采样噪声（除非 t==0，不再加）
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)  # batch 中哪些是 t>0
        x_prev = x_mean + nonzero_mask * torch.sqrt(beta_t) * noise

        return x_prev

    def noise_image(self, x0, t):
        # x0 is images with shape [B, C, H, W]
        # t is denoising timesteps with shape [B]
        
        # Reshape to [B, 1, 1, 1] for proper broadcasting with [B, C, H, W]
        sqrt_alphas_cumprod_ = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_ = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x0)  # Generate Gaussian noise with same shape as x0
        
        xt = x0 * sqrt_alphas_cumprod_ + noise * sqrt_one_minus_alphas_cumprod_
        
        return xt, noise


if __name__ == "__main__":
    # 加载图片
    img_path = './sample.jpg'
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Fix 1: Change 300 to 299 to stay within valid index range
    timesteps = torch.tensor([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 299])
    
    img_tensors = transform(img) # 添加批次维度
    img_tensors = img_tensors.unsqueeze(0).repeat(len(timesteps),1,1,1) 
    
    # 初始化噪声调度器
    noise_scheduler = NoiseScheduler()

    # 生成不同时间步的带噪图像
    assert timesteps.shape[0] == img_tensors.shape[0]
    
    noised_img_tensors, pred_noise  = noise_scheduler.noise_image(img_tensors, timesteps)    

    # 可视化加噪
    plt.figure(figsize=(22, 4))  # 调整宽度以适应11张图片
    
    for i in range(len(timesteps)):
        # Fix 2: Use len(timesteps) to determine number of columns
        plt.subplot(1, len(timesteps), i+1)
        
        # 将tensor转换为可显示的图像格式
        img_to_plot = noised_img_tensors[i].permute(1, 2, 0).clip(0, 1).cpu().numpy()
        
        plt.imshow(img_to_plot)
        plt.title(f"t={timesteps[i].item()}")
        plt.axis('off')  # 隐藏坐标轴
    
    plt.tight_layout()  # 调整子图之间的间距
    # 保存图像
    plt.savefig('./noise_scheduler_sample.jpg')
    plt.show()