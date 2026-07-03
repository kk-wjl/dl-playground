# ---------------------------
# Device
# ---------------------------
device = 'cpu'  # Use 'cuda' if a GPU is available

# ---------------------------
# Data settings
# ---------------------------
batch_size = 64
seq_len = 28       # One token per MNIST row
input_dim = 28     # 28 pixels per row
num_classes = 28   # Predict row positions 0-27

# ---------------------------
# Model settings
# ---------------------------
dim_model = 64     # Embedding / hidden size
num_heads = 4      # Number of attention heads
dim_ff = 128       # Feedforward hidden size
num_layers = 1     # Number of decoder layers

# ---------------------------
# Training settings
# ---------------------------
lr = 1e-3
epochs = 2
