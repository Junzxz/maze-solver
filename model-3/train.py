from create_label import create_labels
from dataset import TileDataset, PreprocessedTileDataset
from torch.utils.data import Dataset, DataLoader
from customCNN import customCNN

import torch
import torch.nn as nn

dataset = PreprocessedTileDataset('processed_tiles.pt')
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = customCNN()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
criterion = nn.CrossEntropyLoss()

losses = []

model.train()

# for i in range(50):
#     epoch_loss = 0.0
#     for x_batch, y_batch in loader:
#         """ YOUR CODE HERE """
#         optimizer.zero_grad()
#         output = model(x_batch)
#         loss = criterion(output, y_batch)
#         loss.backward()
#         optimizer.step()
#         """ YOUR CODE END HERE """
#         epoch_loss += loss.item()
#     print(f"Epoch {i+1}/50, Loss: {epoch_loss:.4f}")
#     losses.append(epoch_loss)

for i in range(50):
    epoch_loss = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    print(f"Epoch {i+1}/50, Loss: {epoch_loss:.4f}")

    
from utils import generate_torch_loader_snippet

example_input = torch.randn(1, 4, 64, 64)
snippet = generate_torch_loader_snippet(model, example_inputs=example_input)
print(snippet)
print(len(snippet) / 1_000_000, "MB") 