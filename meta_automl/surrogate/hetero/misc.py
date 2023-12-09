"""Currently, this module contains information for available graphs only! TODO: extend to all possible Fedot nodes."""

# List of all possible operations
OPERATIONS = [
    "bernb",
    "dt",
    "fast_ica",
    "isolation_forest_class",
    "knn",
    "lgbm",
    "logit",
    "mlp",
    "normalization",
    "pca",
    "poly_features",
    "qda",
    "resample",
    "rf",
    "scaling",
    "dataset",  # Artificial node type to represent dataset node. Probably, will be removed later.
]

# Number of possible features for each node type
NODES_DIMENSIONS = {
    "bernb": 0,
    "dt": 3,
    "fast_ica": 2,
    "isolation_forest_class": 3,
    "knn": 3,
    "lgbm": 7,
    "logit": 1,
    "mlp": 0,
    "normalization": 0,
    "pca": 1,
    "poly_features": 2,
    "qda": 0,
    "resample": 3,
    "rf": 5,
    "scaling": 0,
    "dataset": 0,  # Artificial node type to represent dataset node. Probably, will be removed later.
}

OPERATIONS_WITH_HYPERPARAMETERS = [op_name for op_name, n_feats in NODES_DIMENSIONS.items() if n_feats > 0]
OPERATIONS_WITHOUT_HYPERPARAMETERS = [op_name for op_name, n_feats in NODES_DIMENSIONS.items() if n_feats == 0]
