import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

LABEL_MAP = {
    "floor": 0, "wall": 1, "human": 2, "exit": 3,
    "coin": 4, "gem": 5, "key": 6, "door": 7,
    "lava": 8, "box": 9, "boots": 10, "shield": 11, "ghost": 12
}

class TileDataset(Dataset):
    def __init__(self, tiles):
        self.tiles = tiles
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        img, label = self.tiles[idx]
        # Resize to fixed size, normalize to [0,1]
        img = np.array(Image.fromarray(img).resize((32, 32)))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, LABEL_MAP[label]

class PreprocessedTileDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.images = data['images']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]