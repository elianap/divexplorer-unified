import argparse
import pandas as pd
import time
import os
from divexplorer import DivergenceExplorer
import pickle
from tqdm import tqdm


import pandas as pd
from tqdm import tqdm
import numpy as np


def get_top_N_attributes_from_items(itemset_1_sorted, N):
    def get_attribute(item):
        return item.split("=")[0]

    attributes_sorted = [get_attribute(list(item)[0]) for item in itemset_1_sorted]

    top_N_attributes = []
    for attribute in attributes_sorted:
        if attribute not in top_N_attributes:
            top_N_attributes.append(attribute)
        if len(top_N_attributes) == N:
            break
    return top_N_attributes


def get_stats_divergence(df, min_support, target_col, type_outcome="quantitative"):
    from divexplorer import DivergenceExplorer

    fp_diver = DivergenceExplorer(df)
    try:
        if type_outcome == "quantitative":

            subgroups = fp_diver.get_pattern_divergence(
                min_support=min_support, quantitative_outcomes=[target_col]
            )
        else:
            subgroups = fp_diver.get_pattern_divergence(
                min_support=min_support, boolean_outcomes=[target_col]
            )
    except MemoryError:
        print("Memory limit exceeded!")
    max_value, min_value = (
        subgroups.sort_values(f"{target_col}_div", ascending=False)
        .iloc[[0, -1]][f"{target_col}_div"]
        .values
    )

    mean_value = subgroups[f"{target_col}_div"].mean()
    return min_value, mean_value, max_value


# Load


def limit_memory(max_mem_mb):
    import resource

    """Limit memory usage of the current notebook process."""
    soft, hard = max_mem_mb * 1024 * 1024, max_mem_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    help="Specify the dataset",
    default="airbnb",
    choices=[
        "airbnb",
        "bike",
        "folktables",
        "german",
        "adult",
        "heart",
        "bank",
        "compas",
        "artificial",
        "law",
    ],
)

parser.add_argument(
    "-max_memory", "--max_memory", type=int, help="Maximum GB memory", default=90
)

parser.add_argument(
    "-N", "--n_features", type=int, help="Number of features", default=None
)

parser.add_argument(
    "-min_supports",
    "--min_supports",
    type=float,
    nargs="+",
    help="Set the minimum support values",
    default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
)

parser.add_argument(
    "-alg",
    "--FPM_algorithm",
    type=str,
    help="Algorithm for frequent pattern mining",
    default="fpgrowth",
)

parser.add_argument(
    "-o", "--output_dir", type=str, help="Output directory", default="./results"
)

args = parser.parse_args()

max_memory_gb = int(args.max_memory)
# Limit the notebook to max_memory_gb GB of memory
limit_memory(max_memory_gb * 1024)

dataset_name = args.dataset


data_processed_dir = "./data/processed"


from utils_data import load_dataset

df_discretized, target_col, type_outcome = load_dataset(dataset_name, use_stored=True)

attributes = list(df_discretized.columns.drop(target_col))


# df_discretized_le = df_discretized.copy()
# label_encoders = {}
# for col in df_discretized.columns:
#     if df_discretized[col].dtype == "object":
#         le = LabelEncoder()
#         df_discretized[col] = le.fit_transform(df_discretized[col])
#         label_encoders[col] = le

if type_outcome not in ["quantitative", "boolean"]:
    raise ValueError(f"Outcome type not supported: {type_outcome}")


if args.n_features is None:
    print("We set N to percentages")
    n_features_select = [
        int(len(attributes) * p) for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ]
    n_features_select = set(n_features_select)
    n_features_select = [f for f in n_features_select if f > 1]

else:
    n_features_select = [int(args.n_features)]

print("n_features_select", n_features_select)

for N in n_features_select:
    results = {}

    for min_support in args.min_supports:

        maxs_dict = {}
        mins_dict = {}
        means_dict = {}

        min_support_count = min_support * len(df_discretized)
        print(f"\n\nmin_support_count: {min_support_count}")
        import resource

        def limit_memory(max_mem_mb):
            """Limit memory usage of the current notebook process."""
            soft, hard = max_mem_mb * 1024 * 1024, max_mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

        # Example: Limit the notebook to 2 GB of memory

        from divexplorer import DivergenceExplorer

        fp_diver = DivergenceExplorer(df_discretized)
        try:
            if type_outcome == "quantitative":

                subgroups_all = fp_diver.get_pattern_divergence(
                    min_support=min_support, quantitative_outcomes=[target_col]
                )
            else:
                subgroups_all = fp_diver.get_pattern_divergence(
                    min_support=min_support, boolean_outcomes=[target_col]
                )
        except MemoryError:
            print("Memory limit exceeded!")

        maxs_dict["all"], mins_dict["all"] = (
            subgroups_all.sort_values(f"{target_col}_div", ascending=False)
            .iloc[[0, -1]][f"{target_col}_div"]
            .values
        )

        means_dict["all"] = subgroups_all[f"{target_col}_div"].mean()

        # Select subset
        ## Get random columns

        random_results = []
        import random

        print("Random")
        for seed in range(30):
            random.seed(seed)
            attribute_selection = random.sample(attributes, N)

            min_value_random, mean_value_random, max_value_random = (
                get_stats_divergence(
                    df_discretized[attribute_selection + [target_col]],
                    min_support,
                    target_col,
                    type_outcome,
                )
            )

            random_results.append(
                (min_value_random, mean_value_random, max_value_random)
            )

        mins_dict["random"] = [min_v for min_v, _, _ in random_results]
        means_dict["random"] = [mean_v for _, mean_v, _ in random_results]
        maxs_dict["random"] = [max_v for _, _, max_v in random_results]

        ## Level 1
        # Mimic it by taking only one level as post-processing  - TODO implement in the package!

        subgroup_1 = subgroups_all.loc[subgroups_all["length"] == 1]
        # Sort by absolute divergence
        subgroup_1 = subgroup_1.sort_values(
            f"{target_col}_div", ascending=False, key=lambda x: x.abs()
        )
        itemset_1_sorted = list(subgroup_1.itemset)
        lenght_1_attribute_top = get_top_N_attributes_from_items(itemset_1_sorted, N)
        mins_dict["len_1"], means_dict["len_1"], maxs_dict["len_1"] = (
            get_stats_divergence(
                df_discretized[lenght_1_attribute_top + [target_col]],
                min_support,
                target_col,
                type_outcome,
            )
        )
        ## Global Shapley value level
        for level in [2, 3, 4, 5, 6, 7]:
            subgroup_level = subgroups_all.loc[subgroups_all["length"] <= level]

            from divexplorer import DivergencePatternProcessor

            fp_details_level = DivergencePatternProcessor(subgroup_level, target_col)

            try:
                gsv_level = fp_details_level.global_shapley_value()
            except MemoryError:
                print("Memory limit exceeded!")

            item_gsv_sorted = [
                key
                for key in sorted(
                    gsv_level, key=lambda k: abs(gsv_level[k]), reverse=True
                )
            ]

            gsv_level_attributes_top = get_top_N_attributes_from_items(
                item_gsv_sorted, N
            )

            (
                mins_dict[f"gsv_{level}"],
                means_dict[f"gsv_{level}"],
                maxs_dict[f"gsv_{level}"],
            ) = get_stats_divergence(
                df_discretized[gsv_level_attributes_top + [target_col]],
                min_support,
                target_col,
                type_outcome,
            )
        ## Decision Tree Regression
        from sklearn.tree import DecisionTreeRegressor

        classifier = DecisionTreeRegressor(random_state=42)

        df_discretized_enc = df_discretized.copy()

        df_discretized_enc = pd.get_dummies(
            df_discretized[attributes], columns=attributes, prefix_sep="="
        )
        df_discretized_enc[target_col] = df_discretized[target_col]
        df_discretized_enc.dropna(inplace=True)
        X = df_discretized_enc.drop(columns=[target_col])
        y = df_discretized_enc[target_col].values

        classifier.fit(X, y)
        tree_fi = dict(zip(df_discretized_enc.columns, classifier.feature_importances_))
        item_tree_sorted = [
            frozenset([key])
            for key in sorted(tree_fi, key=lambda k: abs(tree_fi[k]), reverse=True)
        ]
        tree_attributes_top = get_top_N_attributes_from_items(item_tree_sorted, N)
        mins_dict["tree"], means_dict["tree"], maxs_dict["tree"] = get_stats_divergence(
            df_discretized[tree_attributes_top + [target_col]],
            min_support,
            target_col,
            type_outcome,
        )
        ## Random forest
        from sklearn.ensemble import RandomForestRegressor

        rf_classifier = RandomForestRegressor(random_state=42)

        df_discretized_enc = df_discretized.copy()

        df_discretized_enc = pd.get_dummies(
            df_discretized[attributes], columns=attributes, prefix_sep="="
        )

        df_discretized_enc[target_col] = df_discretized[target_col]
        df_discretized_enc.dropna(inplace=True)

        X = df_discretized_enc.drop(columns=[target_col])
        y = df_discretized_enc[target_col].values

        rf_classifier.fit(X, y)
        rf_fi = dict(
            zip(df_discretized_enc.columns, rf_classifier.feature_importances_)
        )

        item_rf_sorted = [
            frozenset([key])
            for key in sorted(rf_fi, key=lambda k: abs(rf_fi[k]), reverse=True)
        ]

        rf_attributes_top = get_top_N_attributes_from_items(item_rf_sorted, N)
        mins_dict["rf"], means_dict["rf"], maxs_dict["rf"] = get_stats_divergence(
            df_discretized[rf_attributes_top + [target_col]],
            min_support,
            target_col,
            type_outcome,
        )

        results[min_support] = {"min": mins_dict, "mean": means_dict, "max": maxs_dict}

    output_pruning = os.path.join(args.output_dir, "column_pruning")

    os.makedirs(output_pruning, exist_ok=True)

    filename_pruning = os.path.join(output_pruning, f"pruning_{args.dataset}_{N}.pkl")

    print(f"Saving time results to {filename_pruning}")

    with open(filename_pruning, "wb") as f:
        pickle.dump(results, f)
