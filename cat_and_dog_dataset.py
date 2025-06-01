import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: str，指向 PetImages 文件夹的路径
        transform: torchvision.transforms，用于图像预处理
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        for label, class_name in enumerate(["Cat", "Dog"]):
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                if os.path.isfile(filepath) and filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Cannot open image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # 适用于 RGB 图像
    ])

    dataset = CatDogDataset("./PetImages", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 试试看
    for images, labels in dataloader:
        print(images.shape)  # torch.Size([32, 3, 128, 128])
        print(labels[:5])    # tensor([0, 1, 0, 1, 0]) 之类的
        break

