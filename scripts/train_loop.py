# scripts/train_loop.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from models.unet_conditional import ConditionalUNet
from models.diffusion_conditional import ConditionalDiffusion
from utils.dataset import CloudSequenceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = CloudSequenceDataset("data/samples", input_frames=2, output_frames=4)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Init model
unet = ConditionalUNet(in_channels=3, out_channels=1).to(device)
diffusion = ConditionalDiffusion(model=unet).to(device)

optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
epochs = 1000

# Training loop
for epoch in range(epochs):
    for input_seq, target_seq in loader:
        input_seq = input_seq.to(device)         # [B, 2, 1, H, W]
        target_seq = target_seq.to(device)       # [B, 4, 1, H, W]

        total_loss = 0
        for i in range(target_seq.shape[1]):
            target = target_seq[:, i]            # [B, 1, H, W]
            cond = input_seq.permute(0, 1, 2, 3, 4).reshape(1, -1, input_seq.shape[-2], input_seq.shape[-1])
            t = torch.randint(0, diffusion.timesteps, (1,), device=device).long()

            loss = diffusion.p_losses(target, cond, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Loss: {total_loss:.4f}")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(unet.state_dict(), "checkpoints/unet_cloud.pth")
            print(f"Saved model at epoch {epoch}")

# Save checkpoint
os.makedirs("checkpoints", exist_ok=True)
torch.save(unet.state_dict(), "checkpoints/unet_cloud.pth")
print("Model saved to checkpoints/unet_cloud.pth")
