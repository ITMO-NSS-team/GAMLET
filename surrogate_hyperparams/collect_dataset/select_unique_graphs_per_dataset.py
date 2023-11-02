import json
import os
import pickle

import fire
import networkx as nx
import pandas as pd
from tqdm import tqdm


def select_unique_graphs_per_dataset(datasets_dir: str, save_path: str) -> None:
    selected_paths = {}

    for dataset_name in tqdm(os.listdir(datasets_dir)):
        if dataset_name == ".DS_Store":  # MacOS workaround.
            continue

        models_dir = os.path.join(datasets_dir, dataset_name, "models")
        model_files = os.listdir(models_dir)
        
        graphs = []
        for model_file in model_files:
            if model_file == ".DS_Store":  # MacOS workaround.
                continue
            
            model_path = os.path.join(models_dir, model_file)
            with open(model_path) as f:
                model = json.load(f)
                
            g = nx.DiGraph()
            for n in model["nodes"]:
                g.add_node(n["operation_id"], operation_type=n["operation_type"])
                if len(n["nodes_from"]) > 0:
                    for n_from in n["nodes_from"]:
                        g.add_edge(n_from, n["operation_id"])
                        
            graphs.append(g)
            
        unique_pipelines_indexes = pd.Series([str(g.nodes().data()) for g in graphs]).drop_duplicates().index.to_list()
        model_files = [os.path.join(models_dir, model_files[i]) for i in unique_pipelines_indexes]
        selected_paths[dataset_name] = model_files

    with open(save_path, "wb") as f:
        pickle.dump(selected_paths, f)


if __name__ == "__main__":
    fire.Fire(select_unique_graphs_per_dataset)
