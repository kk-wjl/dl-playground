import torch
from src.generative.autoregressive.config import (
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
from src.generative.autoregressive.datasets import load_mnist
from src.generative.autoregressive.models import GPT

def main():
    torch_device = torch.device(device)

    train_loader, _ = load_mnist(batch_size)
    model = GPT(input_dim, seq_len, dim_model, num_heads, dim_ff, num_layers, num_classes).to(torch_device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.squeeze(1).to(torch_device), y.to(torch_device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(
                logits.reshape(-1, num_classes),
                y.reshape(-1).long(),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


if __name__ == "__main__":
    main()
