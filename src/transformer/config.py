# config.py
# Hyperparameters

import torch

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 3

INPUT_SIZE = 1      # One feature per pixel
SEQ_LEN = 28 * 28   # Flattened MNIST length
D_MODEL = 32
NUM_HEADS = 2
NUM_LAYERS = 1
D_FF = 64
NUM_CLASSES = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# config.py

# Device
device = 'cpu'  # Use 'cuda' if a GPU is available

# Data settings
batch_size = 64
seq_len = 28       # Treat each MNIST row as one token
input_dim = 28
num_classes = 10

# Model settings
dim_model = 64
num_heads = 4
dim_ff = 128
num_layers = 1

# Training settings
lr = 1e-3
epochs = 2
