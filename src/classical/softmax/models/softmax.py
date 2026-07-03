import torch

def softmax(X):
    """
    X: (batch_size, num_classes)
    return: same shape, each row sums to 1
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


class SoftmaxRegression(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        # W: (num_inputs, num_outputs)
        # b: (num_outputs,)
        self.W = torch.nn.Parameter(
            torch.randn(num_inputs, num_outputs) * 0.01
        )
        self.b = torch.nn.Parameter(
            torch.zeros(num_outputs)
        )

    def forward(self, X):
        X = X.reshape(X.shape[0], -1)

        # logits = XW + b
        logits = X @ self.W + self.b

        return softmax(logits)
    