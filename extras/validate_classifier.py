"""
This script was used to validate the pre-trained classifier.
"""

from classifier.model import DenseModel
from classifier.data import ColoredMNIST
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def confusion_matrix(labels, predictions):
    n_classes = len(set(labels))
    matrix = np.zeros((n_classes, n_classes))
    for label, pred in zip(labels, predictions):
        matrix[label, pred] += 1
    return matrix


def validate_classifier(checkpoint_dir):
    data = ColoredMNIST("../data", download=False, train=False)
    dataloader = DataLoader(
        data, batch_size=32, shuffle=False, pin_memory=True, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseModel((28, 28, 3), 4)
    model.to(device)
    model.load_state_dict(torch.load(f"{checkpoint_dir}/model.pth", weights_only=True))

    labels = []
    predictions = []
    for x, y in tqdm(dataloader, desc=f"Validation"):
        pred = model(x.to(device))
        pred_y = torch.argmax(pred, dim=1)
        labels.extend(y.numpy())
        predictions.extend(pred_y.cpu().numpy())

    # Get confusion matrix
    matrix = confusion_matrix(labels, predictions)
    # Save matrix as text
    np.savetxt(f"{checkpoint_dir}/confusion_matrix.txt", matrix, fmt="%d")


if __name__ == "__main__":
    validate_classifier(checkpoint_dir="checkpoints")
