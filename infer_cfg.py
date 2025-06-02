import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from model_dit_cfg_xattention import SimpleDiTConditional
from noise_scheduler import NoiseScheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using a trained diffusion model with Classifier-Free Guidance')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the image')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='CFG guidance scale (default: 3.0)')
    return parser.parse_args()

@torch.no_grad()
def infer_cfg(model, noise_scheduler: NoiseScheduler, x_t, condition, guidance_scale):
    """
    Classifier-Free Guidance 推理 + 使用封装好的 denoise_image
    """
    model.eval()
    device = x_t.device
    timesteps = list(range(noise_scheduler.timesteps - 1, -1, -1))
    batch_size = x_t.shape[0]

    uncond_condition = torch.zeros_like(condition).to(device)

    for t in timesteps:
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # 分别推理一次有条件和无条件
        pred_noise_uncond = model(x_t, t_tensor, uncond_condition)
        pred_noise_cond = model(x_t, t_tensor, condition)

        # CFG 融合
        pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

        # 使用 noise_scheduler 执行采样一步
        x_t = noise_scheduler.sample_prev_image_distribution(x_t, t_tensor, pred_noise)

    return x_t


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型和噪声调度器
    model = SimpleDiTConditional(img_size=(64, 64), image_channels=1, patch_size=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    noise_scheduler = NoiseScheduler(device=device)

    # 生成初始噪声图像
    x_t = torch.randn((1, 1, 64, 64), device=device)

    # 生成一个全是-1的条件向量
    condition = torch.full((1, 1), -1, device=device)

    # 进行带CFG的推理
    with torch.no_grad():
        output = infer_cfg(model, noise_scheduler, x_t, condition, args.guidance_scale)

    # 归一化并保存图片
    img = output[0, 0].cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()
