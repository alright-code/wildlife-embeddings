import os
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def num_annotations(data_df):
    return len(data_df)


def num_individuals(data_df):
    return len(pd.unique(data_df["name"]))


def num_encounters(data_df):
    if "encounter" in data_df:
        return len(pd.unique(data_df["encounter"]))
    else:
        return None


def num_singletons(data_df):
    name_counts = data_df["name"].value_counts()
    num_singles = len(name_counts[name_counts == 1])

    return num_singles


def num_shared_individuals(data_df1, data_df2):
    names1 = pd.unique(data_df1["name"])
    names2 = pd.unique(data_df2["name"])
    shared_names = np.intersect1d(names1, names2)

    shared_name_counts = data_df2[data_df2["name"].isin(shared_names)][
        "name"
    ].value_counts()

    internal_name_counts = data_df2["name"].value_counts()
    num_valid_internal_names = len(internal_name_counts[internal_name_counts > 1])

    num_shared_names = len(shared_name_counts)

    return num_shared_names, num_valid_internal_names


def print_info(data_df):
    num_annots = num_annotations(data_df)
    print(f"How many annotations? {num_annots}")
    num_ids = num_individuals(data_df)
    print(f"How many unique individuals? {num_ids}")
    encounter_number = num_encounters(data_df)
    print(f"How many encounters? {encounter_number if encounter_number > 1 else 'N/A'}")
    num_singles = num_singletons(data_df)
    print(f"How many singletons? {num_singles}")


def print_shared_info(data_df1, data_df2):
    num_shared_names, num_valid_internal_names = num_shared_individuals(
        data_df1, data_df2
    )
    print(f"How many shared individuals? {num_shared_names}")
    print(f"How many valid internal individuals? {num_valid_internal_names}")


def plot_sighting_hist(train_df, test_df, save_path):
    shared_names = np.intersect1d(train_df["name"], test_df["name"])
    unshared_names = np.setdiff1d(train_df["name"], test_df["name"])
    new_names = len(np.setdiff1d(test_df["name"], train_df["name"]))

    train_df_shared_names = train_df[train_df["name"].isin(shared_names)]
    train_df_unshared_names = train_df[train_df["name"].isin(unshared_names)]

    train_df_shared_names = train_df_shared_names.groupby("encounter").first()
    train_df_unshared_names = train_df_unshared_names.groupby("encounter").first()
    train_df_shared_names = train_df_shared_names.reset_index()
    train_df_unshared_names = train_df_unshared_names.reset_index()

    train_names = train_df_shared_names["name"]
    train_name_counter = Counter(train_names)
    shared_count_counter = Counter(train_name_counter.values())
    x_max = len(shared_count_counter)

    train_names = train_df_unshared_names["name"]
    train_name_counter = Counter(train_names)
    unshared_count_counter = Counter(train_name_counter.values())
    x_max = max(x_max, len(unshared_count_counter))

    x = range(0, x_max)
    y_shared = [shared_count_counter[i] for i in x]
    y_unshared = [unshared_count_counter[i] for i in x]
    y_unshared[0] = new_names

    plt.bar(x, y_shared, width=1.0, edgecolor="black", label="shared")
    plt.bar(
        x, y_unshared, width=1.0, edgecolor="black", label="unshared", bottom=y_shared
    )

    plt.ylabel("Number of Individuals")
    plt.xlabel("s")
    plt.xticks(x)
    plt.title("Number of Individuals Seen s Times")
    plt.legend()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/sighting-hist.png", dpi=300)

    plt.close()


def plot_temporal(data_df):
    data_df_encs = data_df.groupby("encounter", as_index=False).head(1)
    data_df_sorted = data_df_encs.sort_values(by="time")
    x = np.arange(len(data_df_sorted))
    y1 = []
    y2 = []
    from tqdm import tqdm

    for i in tqdm(x):
        df_1 = data_df_sorted.iloc[:i]
        df_2 = data_df_sorted.iloc[i:]

        y1.append(len(df_1["name"].unique()))
        y2.append(len(df_2["name"].unique()))

    plt.plot(x, y1, label="left")
    plt.plot(x, y2, label="right")

    plt.xlabel("Split Index")
    plt.ylabel("Number of Unique Individuals")
    plt.legend()
    plt.savefig(f"test.png", dpi=300)

    plt.close()
