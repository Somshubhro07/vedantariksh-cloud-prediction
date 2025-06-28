# scripts/test_loader.py

from utils.dataset import CloudSequenceDataset
import matplotlib.pyplot as plt

dataset = CloudSequenceDataset(root_dir="data/samples", input_frames=2, output_frames=4)
input_seq, target_seq = dataset[0]

print("Input shape:", input_seq.shape)     # [2, 1, H, W]
print("Target shape:", target_seq.shape)   # [4, 1, H, W]

# Show one input and target
plt.subplot(1, 2, 1)
plt.title("Input[1]")
plt.imshow(input_seq[1].squeeze().numpy(), cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Target[0]")
plt.imshow(target_seq[0].squeeze().numpy(), cmap='gray')
plt.show()
