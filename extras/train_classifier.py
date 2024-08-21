"""
This script was used to train the pre-trained model weights that were given as an option during the exercise.
"""

from classifier.model import DenseModel
from classifier.data import ColoredMNIST
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


def train_classifier(base_dir, epochs=10):
    checkpoint_dir = Path(base_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    data_dir = Path(base_dir) / "data"
    data_dir.mkdir(exist_ok=True)
    #
    model = DenseModel((28, 28, 3), 4)
    data = ColoredMNIST(data_dir, download=True, train=True)
    dataloader = DataLoader(data, batch_size=32, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    losses = []
    for epoch in range(epochs):
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            y_pred = model(x.to(device))
            loss = loss_fn(y_pred, y.to(device))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")
        losses.append(loss.item())
        # TODO save every epoch instead of overwriting?
        torch.save(model.state_dict(), checkpoint_dir / "model.pth")

    with open(checkpoint_dir / "losses.txt", "w") as f:
        f.write("\n".join(str(l) for l in losses))


if __name__ == "__main__":
    this_dir = Path(__file__).parent
    train_classifier(base_dir=this_dir, epochs=10)
