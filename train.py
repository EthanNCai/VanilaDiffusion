import torch
from model_unet import SimpleUnet
from model_dit import SimpleDiT
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from loss import get_loss
from noise_scheduler import NoiseScheduler
import torch.nn.functional as F
from torch.optim import Adam

transform = transforms.Compose([
    transforms.ToTensor(),                        
    transforms.Resize((64, 64)),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# 2. 加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. 包装成 DataLoader（批量读取）
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = SimpleDiT(
    img_size=(64, 64),
    image_channels = 1
)
noise_scheduler = NoiseScheduler(device="cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, noise_scheduler, num_epochs=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Move model to device
    model = model.to(device)
    
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            timesteps = torch.randint(0, noise_scheduler.timesteps, (images.shape[0],), device=device)
            noisy_images, noise = noise_scheduler.noise_image(images, timesteps)
            pred_noise = model(noisy_images, timesteps)
            loss = F.l1_loss(noise, pred_noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.6f}")
        
        torch.save(model.state_dict(), f"output/diffusion_dit_model_epoch_{epoch+1}.pt")
    return model

# Call the training function
if __name__ == "__main__":
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train(model, train_loader, noise_scheduler, num_epochs=100)
    
# train()