import torch
import torch.nn as nn
import torch.optim as optim
from src.transformer.config import (
    batch_size,
    device,
    dim_ff,
    dim_model,
    epochs,
    input_dim,
    lr,
    num_classes,
    num_heads,
    num_layers,
    seq_len,
)
from src.transformer.datasets import load_mnist
from src.transformer.model import Transformer
from src.utils.metrics import AverageMeter, accuracy
from src.utils.paths import figure_dir
from src.utils.visualization import plot_metric_curve

def main():
    torch_device = torch.device(device)
    figures = figure_dir("transformer")

    train_loader, _ = load_mnist(batch_size)
    model = Transformer(input_dim, seq_len, dim_model, num_heads, dim_ff, num_layers, num_classes).to(torch_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for X, y in train_loader:
            X, y = X.squeeze(1).to(torch_device), y.to(torch_device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            batch_size_now = X.size(0)
            loss_meter.update(loss.item(), batch_size_now)
            acc_meter.update(accuracy(y_hat, y), batch_size_now)

        loss_history.append(loss_meter.avg)
        accuracy_history.append(acc_meter.avg)
        print(f"Epoch {epoch+1}, Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg:.4f}")

    plot_metric_curve(loss_history, figures / "loss_curve.png", "Transformer Training Loss", "Loss")
    plot_metric_curve(accuracy_history, figures / "accuracy_curve.png", "Transformer Training Accuracy", "Accuracy")


if __name__ == "__main__":
    main()
