import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from feature_extractors.derm import DermImageEmbedder
from feature_extractors.panderm import PanDermImageEmbedder
from feature_extractors.clip_vit import CLIPImageEmbedder

def preprocess_image_np(img_path, size=(224, 224), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # Open image and convert to RGB
    img = Image.open(img_path).convert("RGB")

    # Resize image
    img = img.resize(size, Image.BILINEAR)

    img_np = np.array(img).astype(np.float32) / 255.0  # scale to [0,1]

    # Normalize each channel: (x - mean) / std
    for c in range(3):
        img_np[..., c] = (img_np[..., c] - mean[c]) / std[c]
        
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(img_np)


def get_all_image_paths(root_dir, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                full_path = os.path.join(subdir, file)
                image_paths.append(full_path)
    return image_paths


def get_image_paths(pths_lst, start, end):
    """
    Retrieve all image paths in the directory.
    """
    return pths_lst[start:end]


def save_embeddings_in_batches_as_csv(image_paths, embedder, batch_size=32, output_csv='embeddings.csv'):
    """
    Save embeddings in batches to prevent loss of progress during long processing.
    Args:
        image_paths (list): List of image file paths.
        image_lesion_ids (dict) : Dictionary of image_id with corresponding lesion_id
        embedder (DermImageEmbedder): Embedder instance.
        batch_size (int): Number of images to process per batch.
        output_csv (str): Path where embeddings CSV will be saved.
    """
    # Open the CSV file and write the header if it's the first batch
    header_written = False

    # Process images in batches
    for start_idx in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
        end_idx = start_idx + batch_size
        if end_idx > 5000:
            end_idx = 5015
        batch_paths = image_paths[start_idx:end_idx]
        batch_embeddings = []
        batch_image_ids = []
        batch_lesion_ids = []
        batch_classes = []
        start = time.time()

        # Process each image in the batch
        for image_path in batch_paths:
            embedding = embedder.extract_embedding(image_path)
            batch_embeddings.append(embedding.flatten())  # Flatten and cast to np.float32
            image_id = os.path.basename(image_path).split('.')[0] # For HAM
            # image_id = image_paths.index(image_path) + 1  # For D7P & DMF
            batch_image_ids.append(image_id)



        # Convert the batch to a DataFrame
        embeddings_df = pd.DataFrame(batch_embeddings)
        embeddings_df['image_id'] = batch_image_ids


        # Write to CSV (append if file already exists, otherwise create new file)
        mode = 'a' if header_written else 'w'
        header = not header_written
        embeddings_df.to_csv(output_csv, mode=mode, header=header, index=False)

        # Set flag to indicate that header has been written
        header_written = True
        print(f"Saved batch to {output_csv}.")
        end = time.time()
        print(f"Time to compute embeddings for batch {start_idx//batch_size +1}: ", end - start, "seconds")


def extract_features(rslt_dict, start, end, batch_size, model_name, output_csv):
    
    if model_name == 'PanDerm':
        embedder = PanDermImageEmbedder() 
    elif model_name == 'derm-foundation':
        embedder = DermImageEmbedder() 
    elif model_name == 'vit':
        embedder = CLIPImageEmbedder()    
    else:
        print("please choose a corect model name (PanDerm or derm-foundation)")

    image_paths = get_image_paths(rslt_dict, start=start, end = end)

    save_embeddings_in_batches_as_csv(
        image_paths, embedder, batch_size=batch_size, 
        output_csv=output_csv)
    
    