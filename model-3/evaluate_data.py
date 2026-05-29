import torch
import numpy as np
from PIL import Image
from create_label import create_labels, create_labels_evaluate
from dataset import LABEL_MAP
from torchvision import transforms

# Create and process tiles
tiles = create_labels_evaluate()

augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    transforms.ToTensor(),
])

normal = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
])

processed = []
for img, label in tiles:
    # Keep original
    processed.append((normal(img), LABEL_MAP[label]))
    processed.append((augment(img), LABEL_MAP[label]))

images = torch.stack([p[0] for p in processed])
labels = torch.tensor([p[1] for p in processed])

torch.save({'images': images, 'labels': labels}, 'processed_evaluate_tiles_96.pt')
print(f"Saved {len(labels)} tiles")