
import torch
from ..utils import preprocess_image_np

class PanDermImageEmbedder:
    def __init__(self, model, transform, device="cpu"):
        self.device = device
        self.model, self.transform = model, transform 
        self.model = self.model.to(self.device).eval()
        print(f"PanDerm model loaded on {self.device}")

    @torch.no_grad()
    def extract_embedding(self, image_path):
        """
        Extract embedding from a single image using PanDerm.
        Args:
            image_path (str): Path to the image.
        Returns:
            np.ndarray: Embedding vector.
        """
        
        image = preprocess_image_np(image_path)
        # image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(input_tensor).detach().cpu().squeeze(0).numpy()
        return embedding