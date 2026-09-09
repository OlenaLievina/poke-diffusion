import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class PokemonDataset(Dataset):
    """Custom class for loading and preprocessing Pokemon images for DDPM models"""
    def __init__(self, data_dir, image_size=64):
        self.data_dir = data_dir
        self.image_size = image_size


        self.image_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
        ]

        # Image preprocessing pipeline 
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load image from disk by index"""
        img_path = self.image_paths[idx]

        # Convert image to RGB to discard  transparency from PNGs
        img = Image.open(img_path).convert('RGB')

        processed_image = self.transform(img)
        return processed_image


def get_dataloader(dataset, batch_size=64, num_workers=4):
    """Creates a DataLoader for batching and shuffling the dataset"""

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        drop_last=True, # Drop incomplete final batch to ensure stable batch shapes
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == '__main__':
      # Check block to verify the complete data pipeline before training
    try:
        dataset = PokemonDataset(data_dir='./data/Pokemon Dataset')
        loader = get_dataloader(dataset, batch_size=4)
        batch = next(iter(loader))
        print(f'Image batch shape: {batch.shape}')
    except Exception as e:
        print(f'Dataset verification failed: {e}')