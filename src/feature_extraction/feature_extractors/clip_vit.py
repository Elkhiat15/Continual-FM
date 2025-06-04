import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPImageEmbedder:
    def __init__(self, model_name='openai/clip-vit-large-patch14', device='cpu'):
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print(f"Hugging Face CLIP ({model_name}) loaded on {self.device}")

    @torch.no_grad()
    def extract_embedding(self, image_path):
        """
        Extract embedding from an image using Hugging Face CLIP ViT-L/14.
        Args:
            image_path (str): Path to the image file.
        Returns:
            np.ndarray: Image embedding vector.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)  # optional: normalize
        return outputs.squeeze(0).cpu().numpy()
