# %%
from dlmbl_unet import UNet
from classifier.model import DenseModel
from classifier.data import ColoredMNIST
import torch
from torch import nn
import json
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from train_gan import Generator

# %%
with open("checkpoints/stargan/losses.json", "r") as f:
    losses = json.load(f)

for key, value in losses.items():
    plt.plot(value, label=key)
plt.legend()

# %%
# Create the model
unet = UNet(depth=2, in_channels=6, out_channels=3, final_activation=nn.Sigmoid())
style_encoder = DenseModel(input_shape=(3, 28, 28), num_classes=3)
# Load model weights
weights = torch.load("checkpoints/stargan/checkpoint_25.pth")
unet.load_state_dict(weights["unet"])
style_encoder.load_state_dict(weights["style_mapping"])  # Change this to style encoder
generator = Generator(unet, style_encoder)

# %% Plotting an example
# Load the data
mnist = ColoredMNIST("../data", download=True, train=False)

# Load one image from the dataset
x, y = mnist[0]
# Load one image from each other class
results = {}
for i in range(len(mnist.classes)):
    if i == y:
        continue
    index = np.where(mnist.targets == i)[0][0]
    style = mnist[index][0]
    # Generate the images
    generated = generator(x.unsqueeze(0), style.unsqueeze(0))
    results[i] = (style, generated)
# %%
# Plot the images
source_style = mnist.classes[y]

fig, axes = plt.subplots(2, 4, figsize=(12, 3))
for i, (style, generated) in results.items():
    axes[0, i].imshow(style.permute(1, 2, 0))
    axes[0, i].set_title(mnist.classes[i])
    axes[0, i].axis("off")
    axes[1, i].imshow(generated[0].detach().permute(1, 2, 0))
    axes[1, i].set_title(f"{mnist.classes[i]}")
    axes[1, i].axis("off")

# Plot real
axes[1, y].imshow(x.permute(1, 2, 0))
axes[1, y].set_title(source_style)
axes[1, y].axis("off")
axes[0, y].axis("off")

# %%
# TODO get prototype images for each class
# TODO convert every image in the dataset + classify result
# TODO plot a confusion matrix
