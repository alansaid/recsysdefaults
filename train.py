from recbole.quick_start import run_recbole
import torch
import time
import os

def log_memory_usage(operation):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f'After {operation}: Allocated memory: {allocated/1024**3:.4f} GB, Reserved memory: {reserved/1024**3:.4f} GB')

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    """
    Main function to run experiments for multiple models and datasets.

    This function performs the following steps:
    1. Initializes timing and device information, printing out PyTorch and CUDA details.
    2. Defines configurations for datasets and models to be evaluated.
    3. Iterates over each combination of dataset and model:
        - Runs the `run_recbole` function with the appropriate configuration files.
        - Logs memory usage after each run.
        - Handles and prints any exceptions that occur during execution.
    4. Prints the total elapsed time for all experiments.

    """
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f'Torch Version: {torch.__version__}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    print("cuDNN Enabled: ", torch.backends.cudnn.enabled)

    dataset_configs = {
        'ml-1m': 'config/datasets/ml-1m.yaml',
        'beeradvocate': 'config/datasets/beeradvocate.yaml',
        "ml-100k": 'config/datasets/ml-100k.yaml',
    }

    model_configs = {
        'EASE': 'config/recmodels/ease.yaml',
        'ItemKNN': 'config/recmodels/itemknn.yaml',
        'NeuMF': 'config/recmodels/neumf.yaml',
        'MultiVAE': 'config/recmodels/multivae.yaml',
        'RecVAE': 'config/recmodels/recvae.yaml',
        'SGL': 'config/recmodels/sgl.yaml',
    }

    trainingeval_config = 'config/default.yaml'

    for dataset_name, dataset_config in dataset_configs.items():
        for model_name, model_config in model_configs.items():
            print(f"\nRunning {model_name} on {dataset_name}...")
            config_file_list = [dataset_config, model_config, trainingeval_config]
            try:
                result = run_recbole(
                    dataset=dataset_name,
                    model=model_name,
                    config_file_list=config_file_list,
                    saved=False
                )
                log_memory_usage(f'{dataset_name}-{model_name}')

                result_dir = f'results/default'
                ensure_dir(result_dir)

                ndcg = result.get('test_result', {}).get('ndcg@10', None)
                output_path = f'results/default/{dataset_name}_{model_name}_metrics.txt'
                os.makedirs('results', exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(f'NDCG@10: {ndcg}\n')
                    f.write(f'Full test_result: {result.get("test_result")}\n')

            except Exception as e:
                print(f"Error running {model_name} on {dataset_name}: {e}")

    t_end = time.time()
    elapsed_time = t_end - t_start
    print(f"\nElapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

