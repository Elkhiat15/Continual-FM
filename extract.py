import argparse
from src.feature_extraction.df_utils import extract_and_save
from dotenv import load_dotenv


if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser(description='Extract features from dataset using a given model.')
    
    parser.add_argument('--data_name', type=str, required=True, help='Dataset name (e.g. ham, d7p, dmf).')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g. PanDerm, clip, derm).')
    # parser.add_argument('--start', type=int, default=0, help='Start index for image slicing.')
    # parser.add_argument('--end', type=int, default=None, help='End index for image slicing.')
    # parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing.')

    args = parser.parse_args()


    print(f"Extracting features from {args.data_name} using {args.model_name}")
    extract_and_save(
        model_name=args.model_name,
        data_name=args.data_name
    )


# Example usage:
#     python extract.py \
#   --data_name dmf \
#   --model_name panderm \



