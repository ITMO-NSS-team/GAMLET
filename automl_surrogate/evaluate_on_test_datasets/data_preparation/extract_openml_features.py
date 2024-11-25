"""This script extracts OpenML meta-features of datasets to be evaluated."""
import argparse
import os
import warnings

import numpy as np
import openml
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

evaluation_datasets_names = [
    "numerai28.6",  #  We use "numerai28_6" in the rest code.
    # "sf-police-incidents",
    # "airlines",
    # "click_prediction_small",
    # "albert",
    # "kddcup09_appetency",
    # "higgs",
    # "christine",
    # "guillermo",
    # "amazon_employee_access",
]


def extract(df, name):
    def calc_majority_class_size(df, name):
        return df["y"].value_counts().max()

    def calc_minority_class_size(df, name):
        return df["y"].value_counts().min()

    def calc_number_of_classes(df, name):
        return len(df["y"].value_counts())

    def calc_number_of_features(df, name):
        return info.loc[name]["NumberOfFeatures"]

    def calc_number_of_instances(df, name):
        return len(df)

    def calc_number_of_instances_with_missing_values(df, name):
        return df.isna().sum(axis=1).astype(bool).astype(int).sum()

    def calc_number_of_missing_values(df, name):
        return df.isna().to_numpy().sum()

    def calc_number_of_numeric_features(df, name):
        return info.loc[name]["NumberOfNumericFeatures"]

    def calc_number_of_symbolic_features(df, name):
        return info.loc[name]["NumberOfSymbolicFeatures"]

    return {
        "MajorityClassSize": calc_majority_class_size(df, name),
        "MinorityClassSize": calc_minority_class_size(df, name),
        "NumberOfClasses": calc_number_of_classes(df, name),
        "NumberOfFeatures": calc_number_of_features(df, name),
        "NumberOfInstances": calc_number_of_instances(df, name),
        "NumberOfInstancesWithMissingValues": calc_number_of_instances_with_missing_values(df, name),
        "NumberOfMissingValues": calc_number_of_missing_values(df, name),
        "NumberOfNumericFeatures": calc_number_of_numeric_features(df, name),
        "NumberOfSymbolicFeatures": calc_number_of_symbolic_features(df, name),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    openml_list = openml.datasets.list_datasets(output_format="dict")
    datalist = pd.DataFrame.from_dict(openml_list, orient="index")
    datalist["name"] = datalist.name.apply(lambda x: x.lower())

    info = datalist[datalist.name.apply(lambda x: x in evaluation_datasets_names)]
    info["name"] = info["name"].apply(lambda x: x.lower())
    # Fix dataset name w.r.t our needs.
    replace_id = info[info["name"] == "numerai28.6"].index
    info.loc[replace_id, "name"] = "numerai28_6"

    def select_latest_version(df):
        latest = df.version.max()
        return df[df.version == latest]

    info = info.groupby("name").apply(select_latest_version)
    info = info.set_index("name")

    meta_features = {}
    for dataset_name in tqdm(evaluation_datasets_names):
        if dataset_name == "numerai28.6":
            dataset_name = "numerai28_6"
        for dataset_fold in range(10):
            x_fname = f"train_{dataset_name}_fold{dataset_fold}.npy"
            y_fname = f"trainy_{dataset_name}_fold{dataset_fold}.npy"
            x = np.load(os.path.join(args.datasets_root, x_fname), allow_pickle=True)
            y = np.load(os.path.join(args.datasets_root, y_fname), allow_pickle=True)
            df = pd.DataFrame(data=x)
            df["y"] = y
            ft = extract(df, dataset_name)
            dataset_id = f"{dataset_name}_{dataset_fold}"
            meta_features[dataset_id] = ft

    df = pd.DataFrame.from_dict(meta_features).T

    to_drop = ["NumberOfInstancesWithMissingValues", "NumberOfMissingValues"]
    df = df.drop(to_drop, axis=1)

    df.to_csv(args.output_csv, index=True)
