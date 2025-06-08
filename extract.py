import argparse
from src.feature_extraction.utils import login_hf
from src.feature_extraction.df_utils import extract_and_save

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from dataset using a given model.')
    
    parser.add_argument('--hf_token', type=str, required=True, help='Hugging Face token for authentication.')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g. PanDerm, clip, derm).')
    parser.add_argument('--data_name', type=str, required=True, help='Dataset name (e.g. ham, d7p, dmf).')
    # parser.add_argument('--start', type=int, default=0, help='Start index for image slicing.')
    # parser.add_argument('--end', type=int, default=None, help='End index for image slicing.')
    # parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing.')

    args = parser.parse_args()

    print(f"Logging into Hugging Face with token.")
    login_hf(token=args.hf_token)

    print(f"Extracting features from {args.data_name} using {args.model_name}")
    extract_and_save(
        model_name=args.model_name,
        data_name=args.data_name,
        start=0,
        end=20,
        batch_size=10
    )


#TODO: Add hf_token to .env for more security 
    # Example usage:
#     python extract.py \
#   --hf_token sk-xxx \
#   --model_name panderm \
#   --data_name dmf \


