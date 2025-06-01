import torch
from model_unet import SimpleUnet
# from model_dit import SimpleDiT
from model_dit_cfg_xattention import SimpleDiTConditional
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from loss import get_loss
from noise_scheduler import NoiseScheduler
import torch.nn.functional as F
from torch.optim import Adam
from cat_and_dog_dataset import CatDogDataset
from torch.utils.data import DataLoader
from torchvision import transforms
transform = transforms.Compose([
transforms.Resize((64, 64)),
transforms.ToTensor(),
transforms.Normalize([0.5]*3, [0.5]*3)  # 适用于 RGB 图像
])

dataset = CatDogDataset("./PetImages", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
model = SimpleDiTConditional(
    img_size=(64, 64),
    image_channels = 3,
    patch_size=2,
)
noise_scheduler = NoiseScheduler(device="cuda" if torch.cuda.is_available() else "cpu")

from tqdm import tqdm  # 确保已导入 tqdm

def train(model, train_loader, noise_scheduler, num_epochs=50, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Move model to device
    model = model.to(device)
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (images, label) in progress_bar:
            label = label.unsqueeze(1).float().to(device)
            images = images.to(device)
            optimizer.zero_grad()
            timesteps = torch.randint(0, noise_scheduler.timesteps, (images.shape[0],), device=device)
            noisy_images, noise = noise_scheduler.noise_image(images, timesteps)
            pred_noise = model(noisy_images, timesteps, label)
            loss = F.l1_loss(noise, pred_noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}/{num_epochs} completed | Average Loss: {avg_loss:.6f}")
        
        torch.save(model.state_dict(), f"output/diffusion_dit_2_model_epoch_{epoch+1}.pt")
    
    return model


# Call the training function
if __name__ == "__main__":
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train(model, dataloader, noise_scheduler, num_epochs=100)
    
# train()