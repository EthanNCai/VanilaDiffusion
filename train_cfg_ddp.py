import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from tqdm import tqdm
from datetime import datetime
import argparse
import torch.nn.functional as F

from model_dit_cfg_xattention import SimpleDiTConditional
from noise_scheduler import NoiseScheduler

def setup(rank, world_size, args):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Set seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

def train_process(rank, world_size, args):
    """
    Main training function for each distributed process.
    """
    # Initialize distributed environment
    setup(rank, world_size, args)
    device = torch.device(f"cuda:{rank}")
    
    # Setup MNIST dataset with proper transforms for single-channel images
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # For single-channel images
    ])
    
    # Download dataset only on rank 0 to avoid file conflicts
    dataset = datasets.MNIST(root='./data', train=True, download=(rank == 0), transform=transform)
    
    # Create distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = SimpleDiTConditional(
        img_size=(args.img_size, args.img_size),
        image_channels=1,  # MNIST is single channel
        patch_size=4,
        condition_dim=10,  # 10 classes for MNIST digits
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Initialize noise scheduler
    noise_scheduler = NoiseScheduler(device=device)
    
    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=(rank == 0))
    
    # Training loop
    for epoch in range(args.epochs):
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        
        # Create progress bar (only on rank 0)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        
        for images, labels in progress_bar:
            # Convert labels to one-hot encoding
            labels = F.one_hot(labels, num_classes=10).float().to(device)
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
            
            # Update progress bar (only on rank 0)
            if rank == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print epoch summary and save model (only on rank 0)
        if rank == 0:
            print(f"\n✅ Epoch {epoch+1}/{args.epochs} | Average Loss: {avg_loss:.6f}")
            
            # Save model checkpoint
            if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
                date_str = datetime.now().strftime("%Y-%m-%d")
                folder_path = os.path.join(args.save_dir, date_str)
                os.makedirs(folder_path, exist_ok=True)
                save_path = os.path.join(folder_path, f"dit_xatt_ddp_epoch_{epoch+1}.pt")
                
                # Save only the model parameters without DDP wrapper
                torch.save(model.module.state_dict(), save_path)
                print(f"✓ Model saved to {save_path}")
    
    # Clean up distributed environment
    cleanup()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a diffusion model on MNIST dataset with DDP')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=32, help='Image size (default: 32)')
    parser.add_argument('--save_dir', type=str, default='output', help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_freq', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--port', type=str, default='12355', help='Port for distributed training')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Get the number of available GPUs
    world_size = torch.cuda.device_count()
    
    # Start distributed training process
    if world_size > 1:
        print(f"Training with {world_size} GPUs using DistributedDataParallel")
        mp.spawn(train_process, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("Only one GPU detected, training with a single GPU")
        train_process(0, 1, args)