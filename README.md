# Hyperparameter Evaluation for RecBole

This repository accompanies the study **"The Hidden Cost of Defaults in Recommender System Evaluation"**, which investigates how framework-level defaults in RecBole affect hyperparameter search outcomes and reproducibility.

We provide scripts to prepare datasets, run baseline evaluations, and conduct hyperparameter optimization experiments across multiple models and datasets.

---

## Repository Structure

```
.
├── prepare_data.py       # Unzips datasets if not already extracted
├── train.py              # Runs RecBole experiments with fixed configs
├── optimize.py           # Runs RecBole's HyperTuning with multiple strategies
├── config/
│   ├── datasets/         # YAMLs defining dataset configurations
│   └── recmodels/        # YAMLs for individual model configurations
├── results/              # Output metrics and optimization results
└── dataset/              # Input datasets (zipped and extracted)
```

---

## Requirements

- Python 3.8+
- PyTorch (tested with 1.13)
- RecBole (>=1.2.0)
- HyperOpt
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

> On Apple Silicon (M1/M2), avoid TensorFlow unless needed, and downgrade `ray` to `1.13.0` if using `HyperTuning`.

---

## Datasets

Extract zipped datasets in the `dataset/` directory by running:

```bash
python prepare_data.py
```

This will extract all `.zip` files into folders if those folders don't already exist.

---

## Training with Default Configurations

Run baseline RecBole experiments using default hyperparameters:

```bash
python train.py
```

Each combination of model and dataset will produce:

- `results/default/{dataset}_{model}_metrics.txt` with:
  - NDCG@10
  - Full test result dictionary

---

## Hyperparameter Optimization

Run hyperparameter tuning using the RecBole `HyperTuning` module:

```bash
python optimize.py
```

- Strategies: `bayes`, `random`, `exhaustive`
- `bayes` and `random` are repeated twice per model
- Results are saved to `results/{strategy}/` including:
  - `.result` files with trial outputs
  - `_Overview.txt` files with best NDCG@10 and full test result

---


## Reproducibility Note

This repository was designed to help surface and analyze undocumented behaviors in RecBole—such as early stopping defaults in `HyperTuning`—that can affect result interpretation and reproducibility.

See our paper for a detailed audit and discussion of reproducibility challenges and recommendations for framework developers.
