import argparse
from src.evaluation import df_utils
from src.evaluation.df_utils import setup
from src.evaluation.experiment import run_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment on embeddings.')
    parser.add_argument('--file_path', type=str, required=None, help='Path to the CSV file with embeddings.')
    parser.add_argument('--data_name', type=str, required=True, help='Dataset name (e.g. ham, d7p, dmf)')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g. clip, panderm)')

    args = parser.parse_args()

    # Map data_name to the appropriate CSV file path
    data_paths = {
        'ham': 'outputs/clip_ham.csv',
        'd7p': 'outputs/clip_d7p.csv',
        'dmf': 'outputs/clip_dmf.csv',
    }

    if args.data_name not in data_paths:
        raise ValueError(f"Unknown data_name: {args.data_name}. Must be one of {list(data_paths.keys())}")

    # file_path = data_paths[args.data_name]

    file_path = args.file_path or data_paths[args.data_name]

    print(f"Using file: {file_path}")
    print(f"Dataset: {args.data_name}")
    print(f"Model: {args.model_name}")
    # Run setup and experiment
    X_train, y_train, X_test, y_test = setup(file_path, data_name=args.data_name)
    run_experiment(X_train, y_train, X_test, y_test, model_name=args.model_name, data_name=args.data_name)

# Example usage:
# file_path is optional, if not provided it will use the default path based on data_name
# python run_experiment.py --file_path outputs/clip_dmf.csv --data_name dmf --model_name clip