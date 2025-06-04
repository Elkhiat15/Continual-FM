import torch
import torch.nn.functional as F

class FrozenRandomProjection:
    def __init__(self, input_dim=1024, output_dim=2048, seed=42):
        torch.manual_seed(seed)
        self.W = torch.randn(output_dim, input_dim) / input_dim**0.5 
        self.W.requires_grad = False

    def transform(self, X_np):
        X = torch.tensor(X_np, dtype=torch.float32)
        projected = F.relu(F.linear(X, self.W)) 
        # projected = F.linear(X, self.W)
        return projected.numpy()