import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import argparse

from model_dit_cfg_xattention import SimpleDiTConditional
from noise_scheduler import NoiseScheduler
from cat_and_dog_dataset import CatDogDataset

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, epochs):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    dataset = CatDogDataset("./PetImages", transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

    model = SimpleDiTConditional(img_size=(128, 128), image_channels=3, patch_size=4,condition_dim=2).to(device)
    model = DDP(model, device_ids=[rank])

    noise_scheduler = NoiseScheduler(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"[GPU {rank}] Epoch {epoch+1}/{epochs}", disable=(rank != 0))

        for images, labels in progress_bar:
            labels = torch.nn.functional.one_hot(labels, num_classes=2).float().to(device)
            images = images.to(device)

            optimizer.zero_grad()
            timesteps = torch.randint(0, noise_scheduler.timesteps, (images.size(0),), device=device)
            noisy_images, noise = noise_scheduler.noise_image(images, timesteps)
            pred_noise = model(noisy_images, timesteps, labels)
            loss = torch.nn.functional.l1_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"\n Epoch {epoch+1} completed | Avg Loss: {avg_loss:.6f}")
            date_str = datetime.now().strftime("%Y-%m-%d")
            folder_path = os.path.join("output", date_str)
            os.makedirs(folder_path, exist_ok=True)
            save_path = os.path.join(folder_path, f"diffusion_dit_xatt_model_cat_and_dog_epoch_{epoch+1}.pt")
            torch.save(model.module.state_dict(), save_path)

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size, args.epochs), nprocs=world_size, join=True)