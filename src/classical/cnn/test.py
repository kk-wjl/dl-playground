import torch

from src.classical.cnn.models import TinyVGG

def test_forward():
    print("=== Testing TinyVGG forward ===")

    # Assume CIFAR-10 input
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)

    model = TinyVGG(num_classes=10, input_channels=3)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

if __name__ == "__main__":
    test_forward()
