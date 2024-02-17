Order of use:

1) `select_unique_pipelines_per_dataset.py`
2) [Optional] `ascending_sort_pipelines_by_data_size.py`. This scripts helps to get samples that are faster to get first.
3) `tune_hyperparameters_for_selected_pipelines.py`
4) `generate_dataset_with_hyperparameters.py`
5) `generate_task_pipe_combs.py`

**Check every script you run to substitute your own paths.**

To check progress of data collection in `tune_hyperparameters_for_selected_pipelines.py` use `check_data_collecting_status.py`.