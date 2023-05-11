from sklearn import preprocessing
import pickle
from torch_geometric.utils.convert import from_networkx
import torch

import pandas as pd

def preprocess_raw_files(path):
    X_dataset = pd.read_csv(path + 'X_dataset.csv').drop(columns='Unnamed: 0')
    X_task_id = pd.read_csv(path + 'X_task_id.csv').drop(columns='Unnamed: 0')
    
    with open(path + 'pipelines_graphs/pipeline_graph_rename.pickle', 'rb') as file:
        pipeline_graph_rename = pickle.load(file)
    with open(path + 'pipelines_graphs/y.pickle', 'rb') as file:
        y_pipeline = list(pickle.load(file))
    with open(path + 'pipelines_graphs/labels.pickle', 'rb') as file:
        labels = list(pickle.load(file))
    with open(path + '/pipelines_graphs/pipelines.pickle', 'rb') as file:
        pipelines = list(pickle.load(file))
    
    uniq_pipelines = []
    pipeline_ids= []
    pipeline_map = dict()
    ind = 0
    for i,p in enumerate(pipelines):
        if p not in pipeline_map:
            pipeline_map[p] = ind
            uniq_pipelines.append(from_networkx(pipeline_graph_rename[i]))
            ind += 1
        pipeline_ids.append(ind)  
    
    d_codes = X_task_id.task_id.astype("category").cat.codes
    dict_tasks = dict(  zip(d_codes.values, np.arange(len(d_codes)))  ) 
    x_dataset = X_dataset.iloc[[dict_tasks[i] for i in range(len(dict_tasks))]]

    X_task_id['pipeline_id'] = pipeline_ids
    X_task_id['y'] = y_pipeline
    X_task_id['task_id'] = d_codes 
    return X_task_id, uniq_pipelines, X_dataset.values[[dict_tasks[i] for i in range(len(dict_tasks))], :]


