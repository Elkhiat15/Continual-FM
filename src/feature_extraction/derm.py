import tensorflow as tf
from io import BytesIO
from huggingface_hub import from_pretrained_keras, login
from src.feature_extraction.data_utils import preprocess_image_np
# from utils import preprocess_image_np

class DermImageEmbedder:
    def __init__(self, model_name="google/derm-foundation", device="cpu"):
        self.device = device
        self.model = from_pretrained_keras(model_name)
        self.infer = self.model.signatures["serving_default"]
        print(f"Model loaded on {device}")

    def preprocess_image(self, image_path):
        """
        Load and preprocess the image for model inference.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            tf.train.Example: Serialized image in tf.train.Example format.
        """
        img = preprocess_image_np(image_path)
        buf = BytesIO()
        img.convert('RGB').save(buf, 'PNG')
        image_bytes = buf.getvalue()

        input_tensor = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/encoded': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_bytes])
                    )
                }
            )
        ).SerializeToString()

        return input_tensor

    def run_inference(self, input_tensor):
        """
        Run the inference on the processed image.

        Args:
            input_tensor (tf.train.Example): Preprocessed image in tf.train.Example format.

        Returns:
            dict: Inference results from the model.
        """
        return self.infer(inputs=tf.constant([input_tensor]))

    def extract_embedding(self, image_path):
        """
        Extract the embedding from an image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            numpy.ndarray: The embedding vector.
        """
        input_tensor = self.preprocess_image(image_path)
        output = self.run_inference(input_tensor)
        embedding_vector = output['embedding'].numpy()
        return embedding_vector