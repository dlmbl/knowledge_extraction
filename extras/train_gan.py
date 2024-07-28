from dlmbl_unet import UNet
from classifier.model import DenseModel
from classifier.data import ColoredMNIST
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
import json
from pathlib import Path

class Generator(nn.Module):
    def __init__(self, generator, style_mapping):
        super().__init__()
        self.generator = generator
        self.style_mapping = style_mapping

    def forward(self, x, y):
        """
        x: torch.Tensor
            The source image
        y: torch.Tensor
            The style image
        """
        style = self.style_mapping(y)
        # Concatenate the style vector with the input image
        style = style.unsqueeze(-1).unsqueeze(-1)
        style = style.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, style], dim=1)
        return self.generator(x)


def set_requires_grad(module, value=True):
    """Sets `requires_grad` on a `module`'s parameters to `value`"""
    for param in module.parameters():
        param.requires_grad = value


def exponential_moving_average(model, ema_model, beta=0.999):
    """Update the EMA model's parameters with an exponential moving average"""
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(beta).add_((1 - beta) * param.data)


def copy_parameters(source_model, target_model):
    """Copy the parameters of a model to another model"""
    for param, target_param in zip(
        source_model.parameters(), target_model.parameters()
    ):
        target_param.data.copy_(param.data)


if __name__ == "__main__":
    save_dir = Path("checkpoints/stargan")
    save_dir.mkdir(parents=True, exist_ok=True)
    mnist = ColoredMNIST("../data", download=True, train=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet(depth=2, in_channels=6, out_channels=3, final_activation=nn.Sigmoid())
    unet_ema = deepcopy(unet)
    discriminator = DenseModel(input_shape=(3, 28, 28), num_classes=4)
    style_mapping = DenseModel(input_shape=(3, 28, 28), num_classes=3)
    generator = Generator(unet, style_mapping=style_mapping)

    # all models on the GPU
    generator = generator.to(device)
    unet_ema = unet_ema.to(device)
    discriminator = discriminator.to(device)

    cycle_loss_fn = nn.L1Loss()
    class_loss_fn = nn.CrossEntropyLoss()

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-6)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)

    dataloader = DataLoader(
        mnist, batch_size=32, drop_last=True, shuffle=True
    )  # We will use the same dataset as before

    losses = {"cycle": [], "adv": [], "disc": []}
    for epoch in range(25):
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch}"):
            x = x.to(device)
            y = y.to(device)
            # get the target y by shuffling the classes
            # get the style sources by random sampling
            random_index = torch.randperm(len(y))
            x_style = x[random_index].clone()
            y_target = y[random_index].clone()

            # Set training gradients correctly
            set_requires_grad(generator, True)
            set_requires_grad(discriminator, False)
            optimizer_g.zero_grad()
            # Get the fake image
            x_fake = generator(x, x_style)
            # Try to cycle back
            x_cycled = generator(x_fake, x)
            # Discriminate
            discriminator_x_fake = discriminator(x_fake)
            # Losses to  train the generator

            # 1. make sure the image can be reconstructed
            cycle_loss = cycle_loss_fn(x, x_cycled)
            # 2. make sure the discriminator is fooled
            adv_loss = class_loss_fn(discriminator_x_fake, y_target)

            # Optimize the generator
            (cycle_loss + adv_loss).backward()
            optimizer_g.step()

            # Set training gradients correctly
            set_requires_grad(generator, False)
            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad()
            # Discriminate
            discriminator_x = discriminator(x)
            discriminator_x_fake = discriminator(x_fake.detach())
            # Losses to train the discriminator
            # 1. make sure the discriminator can tell real is real
            real_loss = class_loss_fn(discriminator_x, y)
            # 2. make sure the discriminator can't tell fake is fake
            fake_loss = -class_loss_fn(discriminator_x_fake, y_target)
            #
            disc_loss = (real_loss + fake_loss) * 0.5
            disc_loss.backward()
            # Optimize the discriminator
            optimizer_d.step()

            losses["cycle"].append(cycle_loss.item())
            losses["adv"].append(adv_loss.item())
            losses["disc"].append(disc_loss.item())

            # EMA update
            exponential_moving_average(unet, unet_ema)
            # TODO add logging, add checkpointing
        # Copy the EMA model's parameters to the generator
        copy_parameters(unet_ema, unet)
        # Store checkpoint
        torch.save(
            {
                "unet": unet.state_dict(),
                "discriminator": discriminator.state_dict(),
                "style_mapping": style_mapping.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "epoch": epoch,
            },
            save_dir / f"checkpoint_{epoch}.pth",
        )
        # Store losses
        with open(save_dir / "losses.json", "w") as f:
            json.dump(losses, f)
