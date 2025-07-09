from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function
import time
import numpy as np
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def patch_hyperopt_rng(seed=42):
    import hyperopt.pyll.stochastic
    rng = np.random.default_rng(seed)
    hyperopt.pyll.stochastic.rng = rng

def main():
    """
    Main function to perform hyperparameter optimization for multiple models and datasets.

    This function iterates over a set of hyperparameter optimization algorithms ('bayes', 'random', 'exhaustive'),
    datasets, and models. For each combination, it runs the optimization process using the HyperTuning class,
    saves the best parameters and results, and logs the elapsed time. Results and overviews are saved to
    designated directories. Errors during optimization are caught and reported, allowing the process to continue
    with other combinations.
    """
    patch_hyperopt_rng()
    np.random.seed(42)  # For reproducibility

    dataset_configs = {
        'ml-1m': 'config/datasets/ml-1m.yaml',
        'beeradvocate': 'config/datasets/beeradvocate.yaml',
        "ml-100k": 'config/datasets/ml-100k.yaml',
    }

    recmodel_configs = {
        'EASE': 'config/recmodels/ease.yaml',
        'ItemKNN': 'config/recmodels/itemknn.yaml',
        'MultiVAE': 'config/recmodels/multivae.yaml',
        'NeuMF': 'config/recmodels/neumf.yaml',
        'RecVAE': 'config/recmodels/recvae.yaml',
        'SGL': 'config/recmodels/sgl.yaml',
    }

    trainingeval_config = 'config/default.yaml'
    hyperopt_algos = ['bayes', 'random', 'exhaustive']

    for hyperopt_algo in hyperopt_algos:
        num_runs = 2 if hyperopt_algo in ['bayes', 'random'] else 1

        for run_id in range(1, num_runs + 1):
            for dataset_name, dataset_config in dataset_configs.items():
                for recmodel_name, recmodel_config in recmodel_configs.items():
                    print(f"\nRunning {hyperopt_algo} optimization (run {run_id}) for {recmodel_name} on {dataset_name}...")
                    t_start = time.time()
                    config_file_list = [dataset_config, recmodel_config, trainingeval_config]
                    result_dir = f'results/{hyperopt_algo}'
                    ensure_dir(result_dir)
                    try:
                        hp = HyperTuning(
                            objective_function=objective_function, 
                            algo=hyperopt_algo, 
                            early_stop=15,
                            max_evals=10, 
                            params_file=f'config/recmodels/{recmodel_name}_range.hyper', 
                            fixed_config_file_list=config_file_list, 
                            display_file=f'{dataset_name}_{recmodel_name}-{hyperopt_algo}-run{run_id}-displayfile'
                        )
                        hp.run()
                        hp.export_result(output_file=f'{result_dir}/hyper_{dataset_name}_{recmodel_name}_run{run_id}.result')

                        best_result = hp.params2result[hp.params2str(hp.best_params)]
                        test_result = best_result.get('test_result', {})
                        ndcg_test = test_result.get('ndcg@10', None)
                        
                        print('Best params:', hp.best_params)
                        print('NDCG@10:', ndcg_test)
                        print('Full test result:', test_result)
                    except Exception as e:
                        print(f"Error during {hyperopt_algo} optimization (run {run_id}) for {recmodel_name} on {dataset_name}: {e}")
                        continue
                    elapsed_time = time.time() - t_start
                    print(f"Elapsed time: {elapsed_time:.2f} seconds")

                    overview_path = f'{result_dir}/{dataset_name}_{recmodel_name}_run{run_id}_Overview.txt'
                    with open(overview_path, 'w') as file:
                        file.write(f'NDCG@10: {ndcg_test}\n')
                        file.write(f'Full test_result: {test_result}\n')


if __name__ == "__main__":
    main()
