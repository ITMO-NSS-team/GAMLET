# MetaFEDOT

[![licence](https://img.shields.io/github/license/itmo-nss-team/metafedot)](https://github.com/itmo-nss-team/metafedot/blob/main/LICENSE)
[![package](https://badge.fury.io/py/metafedot.svg)](https://badge.fury.io/py/metafedot)
[![Build](https://github.com/ITMO-NSS-team/MetaFEDOT/actions/workflows/build.yml/badge.svg)](https://github.com/ITMO-NSS-team/MetaFEDOT/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/itmo-nss-team/metafedot/branch/main/graph/badge.svg)](https://codecov.io/gh/ITMO-NSS-team/MetaFEDOT)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/add_badge/badges/ITMO_badge_rus.svg)](https://itmo.ru)

MetaFEDOT is an open platform for sharing meta-learning experiences in **AutoML** and more
general **Graph Optimization**.
The project has 3 major long-term goals:

1. Provide codebase and utilities for experiments in meta-learning (work in progress)
2. Accumulate metaknowledge for popular application fields, such as tabular classification, tabular regression,
   time series forecasting, etc., based on public datasets and benchmarks (work in progress)
3. Provide user API allowing outer target-independent usage of accumulated meta-knowledge (planned)

## Codebase and utilities for experiments in meta-learning

This framework consists of several key components that automate and enhance the process of meta-learning. It provides
functionalities for dataset and model management, meta-features extraction, dataset similarity assessment. The
components work together to facilitate the initial approximation fitting process.

Each of the components may include different implementations while staying compatible. This is achieved by specification
and maintaining their external interfaces.

### Datasets loader & Dataset

Automate dataset management, including retrieval, caching, and loading into memory. Optimize experiments by minimizing
calls to the dataset source and conserve memory usage.

### Models Loader & Model

Import and consolidate model evaluation data for datasets. Support experiment selection based on predefined criteria,
currently compatible with FEDOT AutoML framework results.

### Meta-features Extractor

Automates the extraction of meta-features from datasets, improving efficiency by caching values. Can load dataset data
if it is necessary for meta-features extraction. For example, one of implementations utilize the PyMFE library for
meta-feature extraction.

### Datasets Similarity Assessor

Assesses dataset similarity based on meta-features. For a given dataset, provides list of similar datasets and optionally calculates
similarity measures. For example, one of implementations uses the "NearestNeighbors" model from scikit-learn.

### Models Advisor

Combines results from Models Loader and Datasets Similarity Assessor. Provides recommendations for models based on
loaded data and similar datasets. Possible implementations allow for heuristic-based suggestions.


# Surrogate training | hyperparameter_search running:
From the repository root run:

`python scripts/main.py <--train|--tune> --config <path_to_your_config>`.

Follow `configs/train_surrogate_model.yml` and `configs/tune_surrogate_model.yml` as reference for training and hyperparameter search accordingly.