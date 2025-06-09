import random
import torch
import numpy as np
import torch.nn as nn
from geoopt import ManifoldParameter, PoincareBall


seed = 15
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class HyperbolicClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes=7, manifold=None):
        super().__init__()
        self.manifold = manifold or PoincareBall()
        self.prototypes = ManifoldParameter(
            torch.randn(num_classes, embed_dim) * 1e-3,
            manifold=self.manifold
        )
        self.prototypes.data = self.manifold.expmap0(self.prototypes.data)

    def forward(self, embeddings):
        hyp_embeddings = self.manifold.expmap0(embeddings)  # To hyperbolic
        logits = -self._pairwise_distances(hyp_embeddings, self.prototypes)
        return logits

    def _pairwise_distances(self, x, y):
        return torch.stack([
            self.manifold.dist(x, y_i) for y_i in y
        ], dim=1)

    def train_model(self, embeddings, labels, optimizer, loss_fn, device='cpu', epochs=10, batch_size=64):
        """
        embeddings: Tensor of shape [N, D] (Euclidean embeddings)
        labels: Tensor of shape [N]
        """
        self.train()
        self.to(device)
        
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        dataset_size = embeddings.size(0)
        
        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)
            total_loss = 0.0
            correct = 0
            total = 0
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                batch_embeddings = embeddings[indices]
                batch_labels = labels[indices]

                optimizer.zero_grad()
                logits = self.forward(batch_embeddings)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_embeddings.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_embeddings.size(0)
            
            avg_loss = total_loss / total
            accuracy = correct / total * 100
            if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
            
    def test_model(self, embeddings, labels, device='cpu', batch_size=64):
        """
        Evaluate the model on test embeddings and labels.
    
        embeddings: Tensor of shape [N, D]
        labels: Tensor of shape [N]
        """
        self.eval()
        self.to(device)
    
        embeddings = embeddings.to(device)
        labels = labels.to(device)
    
        dataset_size = embeddings.size(0)
        correct = 0
        total = 0
    
        with torch.no_grad():
            for i in range(0, dataset_size, batch_size):
                batch_embeddings = embeddings[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
    
                logits = self.forward(batch_embeddings)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_embeddings.size(0)
    
        accuracy = correct / total * 100
        print(f"\033[1mTest Accuracy: {accuracy:.2f}%\033[0m")
        # print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy