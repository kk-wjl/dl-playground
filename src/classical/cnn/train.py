import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from src.classical.cnn.models import TinyVGG
from src.utils.metrics import AverageMeter, accuracy
from src.utils.paths import data_dir, figure_dir
from src.utils.visualization import plot_metric_curve

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    figures = figure_dir("classical/cnn")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir("cifar10"),
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64, #mini-batch size
        shuffle=True
    )

    model = TinyVGG(num_classes=10, input_channels=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_history = []
    accuracy_history = []

    model.train()
    for epoch in range(3):  # Start with 3 epochs
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy(logits, y), batch_size)

        loss_history.append(loss_meter.avg)
        accuracy_history.append(acc_meter.avg)

        print(f"Epoch [{epoch+1}/3], Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg:.4f}")

    plot_metric_curve(loss_history, figures / "loss_curve.png", "CNN Training Loss", "Loss")
    plot_metric_curve(accuracy_history, figures / "accuracy_curve.png", "CNN Training Accuracy", "Accuracy")

if __name__ == "__main__":
    main()
