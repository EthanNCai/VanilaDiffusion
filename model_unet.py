import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
import math


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, 
                 image_channels=1,  # Changed default from 3 to 1
                 down_channels=(64, 128, 256, 512, 1024),
                 up_channels=(1024, 512, 256, 128, 64),
                 out_dim=None,  # Changed default to None, will default to image_channels
                 time_emb_dim=32,
                 img_size=(32, 32)):
        super().__init__()
        
        # Store parameters
        self.image_channels = image_channels
        self.out_dim = out_dim if out_dim is not None else image_channels  # Default output channels to input channels
        self.time_emb_dim = time_emb_dim
        
        # Check if image size is valid
        # With 4 downsample operations, dimensions must be divisible by 2^4=16
        self.img_height, self.img_width = img_size
        min_dim = 2**(len(down_channels)-1)
        
        if self.img_height % min_dim != 0 or self.img_width % min_dim != 0:
            # Adjust to the nearest valid size
            self.img_height = (self.img_height // min_dim) * min_dim
            self.img_width = (self.img_width // min_dim) * min_dim
            print(f"Warning: Image dimensions adjusted to ({self.img_height}, {self.img_width})")
        
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], 
                                    time_emb_dim) 
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], 
                                        time_emb_dim, up=True) 
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv2d(up_channels[-1], self.out_dim, 1)

    def forward(self, x, timestep):
        # Check input shape
        _, c, h, w = x.shape
        if c != self.image_channels:
            raise ValueError(f"Input should have {self.image_channels} channels, but got {c}")
        if h != self.img_height or w != self.img_width:
            print(f"Warning: Input size ({h},{w}) doesn't match configured size ({self.img_height},{self.img_width})")
            # You could also resize here: x = F.interpolate(x, size=(self.img_height, self.img_width))
        
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


if __name__ == "__main__":
    # Initialize the model with custom parameters and image size
    
    # Example 1: Default 1-channel model
    channels = 1  # Default is now 1 channel instead of 3
    model = SimpleUnet(
        image_channels=channels,
        img_size=(64, 64)
    )
    print("\nExample 1: Single-channel model")
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    print(f"Configured image size: {model.img_height}x{model.img_width}")
    print(f"Input channels: {model.image_channels}, Output channels: {model.out_dim}")
    
    # Create random input tensors matching the configured size
    batch_size = 2
    
    # Random image tensor with shape [batch_size, channels, height, width]
    x = torch.randn(batch_size, channels, 64, 64)
    
    # Random timestep tensor with shape [batch_size]
    timestep = torch.randint(0, 1000, (batch_size,))
    
    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {timestep.shape}")
    
    # Forward pass
    output = model(x, timestep)
    
    print(f"Output shape: {output.shape}")
    
    # Example 2: RGB model with 3 channels
    channels = 3
    model_rgb = SimpleUnet(
        image_channels=channels, 
        img_size=(64, 64)
    )
    print("\nExample 2: RGB model with 3 channels")
    print("Num params: ", sum(p.numel() for p in model_rgb.parameters()))
    print(f"Input channels: {model_rgb.image_channels}, Output channels: {model_rgb.out_dim}")
    
    # Create RGB input
    x_rgb = torch.randn(batch_size, channels, 64, 64)
    output_rgb = model_rgb(x_rgb, timestep)
    print(f"Input shape: {x_rgb.shape}")
    print(f"Output shape: {output_rgb.shape}")
    
    # Example 3: Model with different input and output channels
    in_ch, out_ch = 1, 2  # Example: 1 channel in, 2 channels out
    model_custom = SimpleUnet(
        image_channels=in_ch,
        out_dim=out_ch,
        img_size=(64, 64)
    )
    print("\nExample 3: Custom channel configuration")
    print(f"Input channels: {model_custom.image_channels}, Output channels: {model_custom.out_dim}")
    
    # Create custom input
    x_custom = torch.randn(batch_size, in_ch, 64, 64)
    output_custom = model_custom(x_custom, timestep)
    print(f"Input shape: {x_custom.shape}")
    print(f"Output shape: {output_custom.shape}")