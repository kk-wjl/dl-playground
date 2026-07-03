import torch
from torchvision import datasets, transforms

from src.transformer import config
from src.transformer.model import Transformer
from src.utils.paths import data_dir

DATA_DIR = data_dir("mnist")

def main():
    device = torch.device(config.DEVICE)
    model = Transformer(
        input_dim=config.INPUT_SIZE,
        seq_len=config.SEQ_LEN,
        dim_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        dim_ff=config.D_FF,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
    ).to(device)

    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    x, y = test_dataset[0]
    x = x.view(1, config.SEQ_LEN, config.INPUT_SIZE).to(device)
    y_hat = model(x)
    print("Predicted logits:", y_hat)
    print("Predicted class:", torch.argmax(y_hat, dim=1).item())
    print("True label:", y)

if __name__ == "__main__":
    main()
