import torch
import argparse
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_dit import SimpleDiTConditional
from noise_scheduler import NoiseScheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model on MNIST dataset')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 64)')
    parser.add_argument('--channels', type=int, default=1, help='Number of image channels (default: 1)')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for DiT (default: 2)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='output', help='Directory to save model checkpoints')
    parser.add_argument('--save_freq', type=int, default=5, help='Save model every N epochs')
    
    return parser.parse_args()

def setup_data_loader(args):
    """Setup MNIST dataset with proper transforms"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST standard normalization
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    return train_loader

def train(args):
    """Main training function"""
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = setup_data_loader(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    
    model = SimpleDiTConditional(
        img_size=(args.img_size, args.img_size),
        image_channels=args.channels,
        patch_size=args.patch_size
    ).to(device)
    
    noise_scheduler = NoiseScheduler(device=device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    os.makedirs(args.save_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(args.save_dir, date_str)
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, _) in progress_bar:
            images = images.to(device)
            
            optimizer.zero_grad()
            
            timesteps = torch.randint(0, noise_scheduler.timesteps, (images.shape[0],), device=device)
            noisy_images, noise = noise_scheduler.noise_image(images, timesteps)
            
            pred_noise = model(noisy_images, timesteps)
            loss = F.l1_loss(noise, pred_noise)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} completed | Average Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            model_filename = f"dit_epoch_{epoch+1}.pt"
            checkpoint_path = os.path.join(save_path, model_filename)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Model checkpoint saved to {checkpoint_path}")
    
    print(f"✓ Training complete!")
    return model

if __name__ == "__main__":
    args = parse_args()
    trained_model = train(args)