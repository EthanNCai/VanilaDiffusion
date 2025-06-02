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
    model = SimpleDiT(img_size=(64, 64), image_channels=1, patch_size=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduler(device=device)

    # Start from pure noise
    x_t = torch.randn((1, 1, 64, 64), device=device)

    with torch.no_grad():
        for t in reversed(range(noise_scheduler.timesteps)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            pred_noise = model(x_t, t_tensor)
            x_t = noise_scheduler.sample_prev_image_distribution(x_t, t_tensor, pred_noise)

    # Convert to numpy and normalize to [0,1]
    img_array = x_t.cpu().detach().numpy()
    img = img_array[0, 0]  # (64, 64)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()
