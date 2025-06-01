import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from model_unet import SimpleUnet
from model_dit import SimpleDiT
from noise_scheduler import NoiseScheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using a trained diffusion model')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the image')
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    model = SimpleDiT(img_size=(64, 64), image_channels=1, patch_size=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()  # Set model to evaluation mode
    noise_scheduler = NoiseScheduler(device=device)
    x_t = torch.randn((1, 1, 64, 64)).to(device)
    with torch.no_grad():
        for t in reversed(range(0, noise_scheduler.timesteps)):
            t_tensor = torch.tensor([t]).to(device)
            pred_noise = model(x_t, t_tensor)
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            alpha = noise_scheduler.alphas[t]
            beta = 1 - alpha
            eps = 1e-8
            x_t = (1 / torch.sqrt(alpha + eps)) * (x_t - ((beta) / (torch.sqrt(1 - noise_scheduler.alphas_cumprod[t] + eps))) * pred_noise)
            
            if t > 0:
                x_t = x_t + torch.sqrt(beta) * noise
    
    img_array = x_t.cpu().detach().numpy()
    img = img_array[0, 0]  # Extract the image (shape becomes 64x64)
    
    img = (img - img.min()) / (img.max() - img.min())
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Uncommented to hide axis
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()