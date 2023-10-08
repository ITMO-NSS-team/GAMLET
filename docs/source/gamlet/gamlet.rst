Intro to GAMLET
==================

| This framework consists of several key components that automate and enhance the process of meta-learning. It provides functionalities for dataset and model management, meta-features extraction, dataset similarity assessment. 
| The components work together to facilitate the initial approximation fitting process.
| Each of the components may include different implementations while staying compatible. 
| This is achieved by specification and maintaining their external interfaces.

Datasets loader & Dataset
-------------------------

| Automate dataset management, including retrieval, caching, and loading into memory. 
| Optimize experiments by minimizing calls to the dataset source and conserve memory usage.

Models Loader & Model
---------------------

| Import and consolidate model evaluation data for datasets.
| Support experiment selection based on predefined criteria, currently compatible with FEDOT AutoML framework results.

Meta-features Extractor
-----------------------
| Automates the extraction of meta-features from datasets, improving efficiency by caching values. 
| Can load dataset data if it is necessary for meta-features extraction. 
| For example, one of implementations utilize the PyMFE library for meta-feature extraction.

Datasets Similarity Assessor
----------------------------
| Assesses dataset similarity based on meta-features. 
| For a given dataset, provides list of similar datasets and optionally calculates similarity measures. 
| For example, one of implementations uses the "NearestNeighbors" model from scikit-learn.

Models Advisor
--------------
| Combines results from Models Loader and Datasets Similarity Assessor. 
| Provides recommendations for models based on loaded data and similar datasets. 
| Possible implementations allow for heuristic-based suggestions.