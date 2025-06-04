import torch
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_dit_cfg_xattention import SimpleDiTConditional
from noise_scheduler import NoiseScheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model on MNIST dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 32)')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for ViT (default: 4)')
    parser.add_argument('--save_dir', type=str, default='output', help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_freq', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='LR decay step size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='LR decay rate')
    return parser.parse_args()

def setup_data_loader(args):
    """Setup MNIST dataset with proper transforms"""
    # Setup transform for single-channel grayscale images (MNIST)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalization for single-channel images
    ])
    
    # Download and prepare dataset
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Create data loader
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    return train_loader

def train(args):
    """Main training function"""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup data loader
    train_loader = setup_data_loader(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Initialize model
    model = SimpleDiTConditional(
        img_size=(args.img_size, args.img_size),
        image_channels=1,  # MNIST is single channel
        patch_size=args.patch_size,
        condition_dim=10,  # 10 classes for MNIST digits
    ).to(device)
    
    # Initialize noise scheduler
    noise_scheduler = NoiseScheduler(device=device)
    
    # Initialize optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    
    # Create directory for saving models
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(args.save_dir, date_str)
    os.makedirs(save_path, exist_ok=True)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        # Create progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (images, labels) in progress_bar:
            # Convert labels to one-hot encoding
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
            images = images.to(device)
            
            # Training step
            optimizer.zero_grad()
            
            # Apply noise at random timesteps
            timesteps = torch.randint(0, noise_scheduler.timesteps, (images.shape[0],), device=device)
            noisy_images, noise = noise_scheduler.noise_image(images, timesteps)
            
            # Get model prediction
            pred_noise = model(noisy_images, timesteps, labels)
            loss = F.l1_loss(noise, pred_noise)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f}")
        
        # Save model at specified frequency
        if (epoch + 1) % args.save_freq == 0:
            model_filename = f"dit_xatt_epoch_{epoch+1}.pt"
            checkpoint_path = os.path.join(save_path, model_filename)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(save_path, "diffusion_dit_xatt_mnist_model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Training complete! Final model saved to {final_model_path}")
    
    return model

if __name__ == "__main__":
    args = parse_args()
    trained_model = train(args)