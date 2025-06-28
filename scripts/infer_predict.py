# scripts/infer_predict.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision.transforms import ToPILImage
from models.unet_conditional import ConditionalUNet
from models.diffusion_conditional import ConditionalDiffusion
from utils.dataset import CloudSequenceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ConditionalUNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("checkpoints/unet_cloud.pth", map_location=device))
model.eval()

diffusion = ConditionalDiffusion(model).to(device)

# Load 2 input frames
dataset = CloudSequenceDataset("data/samples", input_frames=2, output_frames=4)
input_seq, _ = dataset[0]               # shape: [2, 1, H, W]
input_seq = input_seq.unsqueeze(0)      # shape: [1, 2, 1, H, W]
cond = input_seq.reshape(1, -1, input_seq.shape[-2], input_seq.shape[-1]).to(device)

# Predict next 4 frames
os.makedirs("outputs", exist_ok=True)
to_img = ToPILImage()

for i in range(4):
    noise = torch.randn(1, 1, cond.shape[2], cond.shape[3]).to(device)
    t = torch.randint(0, diffusion.timesteps, (1,), device=device).long()

    # Concatenate conditioning + noisy frame
    model_input = torch.cat([cond, noise], dim=1)
    with torch.no_grad():
        denoised = model(model_input)

    # Unnormalize
    pred = denoised.squeeze(0).squeeze(0).cpu().clamp(-1, 1)
    pred = (pred + 1) / 2  # back to [0, 1]
    img = to_img(pred)
    img.save(f"outputs/predicted_{i+1}.png")
    print(f"Saved outputs/predicted_{i+1}.png")
