import json

import numpy as np

from meta_automl.data_preparation.pipeline_features_extractors.fedot_pipeline_features_extractor import (
    FEDOTPipelineFeaturesExtractor,
)
from meta_automl.surrogate.hetero.misc import NODES_DIMENSIONS, OPERATIONS

ABSENT_NODE = -2
NO_PARAMS = -3
DATASET_PARAMS = -4


def extract_parameters(node):
    node_parameters = {}
    node_parameters.update(node["params"])
    node_parameters.update(node["custom_params"])
    return node_parameters


op2id = {k: i for i, k in enumerate(OPERATIONS)}
pipe_extractor = FEDOTPipelineFeaturesExtractor()


def deterministic_graph_embedding(pipe_json_str: str) -> np.ndarray:
    pipe = json.loads(pipe_json_str)

    # Order nodes list
    pipe_nodes = [
        None,
    ] * len(pipe["nodes"])
    for node in pipe["nodes"]:
        i = node["operation_id"]
        pipe_nodes[i] = node

    adj = np.zeros((len(op2id), len(op2id)))

    # Mark missing operations
    for op, id in op2id.items():
        if op != "dataset" and op not in pipe["total_pipeline_operations"]:
            adj[id] = ABSENT_NODE

    # Fill adjacency matrix with presented nodes
    for node in pipe_nodes:
        node_op = node["operation_type"]
        node_op_id = op2id[node_op]
        nodes_from = node["nodes_from"]
        if len(nodes_from) == 0:
            dataset_id = op2id["dataset"]
            adj[dataset_id][node_op_id] = 1.0
        else:
            for i in nodes_from:
                src_node_op = pipe_nodes[i]["operation_type"]
                src_node_op_id = op2id[src_node_op]
                adj[src_node_op_id][node_op_id] = 1.0

    adj = adj.reshape(-1)

    parameters = [None] * len(op2id)

    # Mark missing operations
    for op, id in op2id.items():
        if op != "dataset" and op not in pipe["total_pipeline_operations"]:
            node_dim = NODES_DIMENSIONS[op]
            if node_dim == 0:
                node_dim = 1
            parameters[id] = np.full(node_dim, ABSENT_NODE)

    # Fill parameters with presented nodes
    for node in pipe_nodes:
        node_op = node["operation_type"]
        node_op_id = op2id[node_op]
        node_params = extract_parameters(node)
        if len(node_params) == 0:
            parameters_vec = np.full(1, NO_PARAMS)
        else:
            parameters_vec = pipe_extractor._operation_parameters2vec(node_op, parameters)
        parameters[node_op_id] = parameters_vec

    dataset_id = op2id["dataset"]
    parameters[dataset_id] = np.full(1, DATASET_PARAMS)

    parameters = np.hstack(parameters)

    node_vec = np.hstack([parameters, adj]).astype(np.float32)
    return node_vec
