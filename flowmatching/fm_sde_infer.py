import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from model_dit import SimpleDiTConditional
from PIL import Image

# Define condition vectors for digits 0-9 (与原代码相同)
conditions_mappings = {
    '0': torch.tensor((1,0,0,0,0,0,0,0,0,0)).float().unsqueeze(0),
    '1': torch.tensor((0,1,0,0,0,0,0,0,0,0)).float().unsqueeze(0),
    '2': torch.tensor((0,0,1,0,0,0,0,0,0,0)).float().unsqueeze(0),
    '3': torch.tensor((0,0,0,1,0,0,0,0,0,0)).float().unsqueeze(0),
    '4': torch.tensor((0,0,0,0,1,0,0,0,0,0)).float().unsqueeze(0),
    '5': torch.tensor((0,0,0,0,0,1,0,0,0,0)).float().unsqueeze(0),
    '6': torch.tensor((0,0,0,0,0,0,1,0,0,0)).float().unsqueeze(0),
    '7': torch.tensor((0,0,0,0,0,0,0,1,0,0)).float().unsqueeze(0),
    '8': torch.tensor((0,0,0,0,0,0,0,0,1,0)).float().unsqueeze(0),
    '9': torch.tensor((0,0,0,0,0,0,0,0,0,1)).float().unsqueeze(0),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using a trained diffusion model with Classifier-Free Guidance')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output', type=str, default='fm_output_sde.jpg', help='Path to save the image')
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='CFG guidance scale (default: 3.0)')
    parser.add_argument('--label', type=str, default='0', help='Label to generate (0-9, default: 0)')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 32)')
    parser.add_argument('--channels', type=int, default=1, help='Number of image channels (default: 1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--fm_steps', type=int, default=250, help='steps for flow matching')
    parser.add_argument('--sde_noise_scale', type=float, default=0.4, help='SDE noise scale parameter a (default: 0.1)')
    return parser.parse_args()

def infer_flow_matching_cfg(model, x_0, condition, guidance_scale=3.0, fm_steps=250, sde_noise_scale=0.1):
    """
    sde_noise_scale 是论文中提到的超参a
    """
    model.eval()
    device = x_0.device
    x = x_0.clone()

    B = x.shape[0]
    dt = 1.0 / fm_steps  # Time step size
    uncond_condition = torch.zeros_like(condition).to(device)

    with torch.no_grad():
        for i in range(fm_steps):
            
            t_val = i / fm_steps
            t = torch.full((B,), t_val, dtype=torch.float, device=device)  
            
            # CFG 逻辑
            v_uncond = model(x, t, uncond_condition, force_drop_condition=True)
            v_cond = model(x, t, condition)

            # CFG 逻辑
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            
            # FlowGRPO 公式9的下面注解部分给出的噪声scaler因子计算方法 
            # sigma_t = a * sqrt(t/(1-t))
            sigma_t = sde_noise_scale * torch.sqrt(t / (1.0 - t + 1e-5))
            
            # FlowGRPO 公式9 中间部分
            drift_correction = (sigma_t**2 / (2 * t_val + 1e-5)) * (x + (1 - t_val) * v) * dt
            
            # FlowGRPO 公式9 右半部分
            noise = torch.randn_like(x) * sigma_t.view(-1, 1, 1, 1) * torch.sqrt(torch.tensor(dt).to(device))
            
            # FlowGRPO 公式9 总体
            x = x + v * dt + drift_correction + noise

    return x


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.label in conditions_mappings:
        condition = conditions_mappings[args.label].to(device)
    else:
        print(f"Warning: Label '{args.label}' not found in mappings, using default label '0'")
        condition = conditions_mappings['0'].to(device)

    model = SimpleDiTConditional(
        img_size=(args.img_size, args.img_size),
        image_channels=args.channels,
        patch_size=4,
        condition_dim=10
    ).to(device)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    x_t = torch.randn((1, args.channels, args.img_size, args.img_size), device=device)

    with torch.no_grad():
        output = infer_flow_matching_cfg(
            model, 
            x_t, 
            condition, 
            args.guidance_scale, 
            fm_steps=args.fm_steps,
            sde_noise_scale=args.sde_noise_scale
        )

    img_array = output.cpu().detach().numpy()
    img = img_array[0, 0] 
    
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Save the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Generated image for label {args.label} saved to {args.output}")
if __name__ == "__main__":
    main()