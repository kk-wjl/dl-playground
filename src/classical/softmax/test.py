import torch

from src.classical.softmax.models import SoftmaxRegression, softmax

def test_softmax():
    print("=== Test softmax ===")
    X = torch.rand((2, 5))
    X_prob = softmax(X)

    print("softmax output:\n", X_prob)
    print("row sum (should be 1):\n", X_prob.sum(dim=1))



def test_model():
    print("\n=== Test Softmax Regression forward ===")

    batch_size = 4
    num_inputs = 10
    num_outputs = 3

    model = SoftmaxRegression(num_inputs, num_outputs)

    X = torch.rand((batch_size, num_inputs))
    y_hat = model(X)

    print("output shape:", y_hat.shape)
    print("output:\n", y_hat)
    print("row sum (should be 1):\n", y_hat.sum(dim=1))



if __name__ == "__main__":
    test_softmax()
    test_model()
