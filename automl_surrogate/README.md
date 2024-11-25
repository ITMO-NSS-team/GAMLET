This directory is considered as a separate installable module, designed to replace `meta_automl/surrogate`.
The module contains 3 different architectures for pipeline ranking tasks with same GNN backbone.
The module utilizes custom data structure instead of `pytorch_geometric.data.Data` and `pytorch_geometric.data.Batch` to store heterogenous nodes with homogenoeus links.

# Installation
```
pip install fedot==0.7.2
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.4.0
pip install pytorch-lightning==2.0.5
pip install einops==0.7.0
pip install -e .
```

# Evaluation:
Use `data/get_datasets_folds.py` to load datasets from OpenML and split them into 10 cross-validation folds.

Then, follow `evaluate_on_test_datasets/README.md`.

Final model checkpoint is in `ranknet_checkpoint`.