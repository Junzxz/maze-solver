import torch
import time
from torch.utils.data import DataLoader
from dataset import PreprocessedTileDataset, LABEL_MAP
from models.model_6 import get_model

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def evaluate_model(eval_data_path='processed_evaluate_tiles.pt'):
    start = time.time()
    model = get_model() 
    dataset = PreprocessedTileDataset(eval_data_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    total = len(all_labels)
    correct = (all_preds == all_labels).sum().item()
    print(f"Overall: {correct}/{total} ({correct/total:.4f})")

    for idx in range(len(LABEL_MAP)):
        mask = all_labels == idx    
        if mask.sum() == 0:
            continue
        c = (all_preds[mask] == idx).sum().item()
        t = mask.sum().item()
        print(f"  {REVERSE_LABEL_MAP[idx]:>8s}: {c}/{t} ({c/t:.4f})")

    print(f"\nTime: {time.time() - start:.2f}s")
    
def evaluate_225_tiles(eval_data_path='processed_evaluate_tiles_96.pt'):
    start = time.time()
    model = get_model()

    # Simulate 225 tiles like final.py does
    tiles = [torch.randn(4, 96, 96) for _ in range(225)]

    with torch.no_grad():
        preds = model(torch.stack(tiles)).argmax(dim=1)

    print(f"Time: {time.time() - start:.2f}s")

evaluate_model()