import torch
import torch.nn as nn
import torch.optim as optim

from src.classical.rnn import SimpleRNN

def generator_data(batch_size=32, seq_length=5):
    x = torch.randn(batch_size, seq_length, 1)  # input_size = 10
    y = (x[:, -1, 0] > x[:, 0, 0]).long()
    return x, y

def main():
    model = SimpleRNN(input_size=1, hidden_size=16, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    print("Initial weight_hh:")
    print(model.rnn.weight_hh_l0)


    for epoch in range(50):
        x, y = generator_data()
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch) % 10 == 0:
            print(f'Epoch [{epoch+1}/50], Loss: {loss.item():.4f}')

    print("Updated weight_hh:")
    print(model.rnn.weight_hh_l0)


if __name__ == "__main__":
    main()
