import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from model_dit import SimpleDiT
from noise_scheduler import NoiseScheduler
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using a trained diffusion model')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the image')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 32)')
    parser.add_argument('--channels', type=int, default=1, help='Number of image channels (default: 1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    return parser.parse_args()

@torch.no_grad()
def infer(model, noise_scheduler: NoiseScheduler, x_t):
    """
    Standard diffusion inference process
    """
    model.eval()
    device = x_t.device
    timesteps = list(range(noise_scheduler.timesteps - 1, -1, -1))
    batch_size = x_t.shape[0]  # Always 1 in this simplified version

    for t in timesteps:
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.int)
        
        # Get model prediction
        pred_noise = model(x_t, t_tensor)
        
        # Sample one step using noise_scheduler
        x_t = noise_scheduler.sample_prev_image_distribution(x_t, t_tensor, pred_noise)

    return x_t


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Initialize model and noise scheduler
    model = SimpleDiT(
        img_size=(args.img_size, args.img_size),
        image_channels=args.channels,
        patch_size=4
    ).to(device)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    noise_scheduler = NoiseScheduler(device=device)

    # Generate initial noise image
    x_t = torch.randn((1, args.channels, args.img_size, args.img_size), device=device)

    # Perform inference
    with torch.no_grad():
        output = infer(model, noise_scheduler, x_t)

    # Extract and save the generated image
    img_array = output.cpu().detach().numpy()
    img = img_array[0, 0]  # Extract first channel of the first (and only) image
    
    # Normalize the image for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Save the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Generated image saved to {args.output}")


if __name__ == "__main__":
    main()