import argparse
import pandas as pd
import time
import os
from divexplorer import DivergenceExplorer
import pickle
from tqdm import tqdm


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
        "german",
        "adult",
        "heart",
        "bank",
        "compas",
        "folktables",
        "artificial",
        "law",
        "bike",
    ],
)

parser.add_argument(
    "-max_memory", "--max_memory", type=int, help="Maximum GB memory", default=90
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

dataset_name = args.dataset


data_processed_dir = "./data/processed"


from utils_data import load_dataset

df_discretized, target_col, type_outcome = load_dataset(dataset_name, use_stored=True)


# df_discretized_le = df_discretized.copy()
# label_encoders = {}
# for col in df_discretized.columns:
#     if df_discretized[col].dtype == "object":
#         le = LabelEncoder()
#         df_discretized[col] = le.fit_transform(df_discretized[col])
#         label_encoders[col] = le

if type_outcome not in ["quantitative", "boolean"]:
    raise ValueError(f"Outcome type not supported: {type_outcome}")

fp_diver = DivergenceExplorer(df_discretized)

# Limit the notebook to max_memory_gb GB of memory
limit_memory(max_memory_gb * 1024)

min_supports = args.min_supports

time_results = {}
number_subgroups = {}

for min_support in min_supports:
    min_support = float(min_support)
    print(min_support)
    try:
        start_time = time.time()
        if type_outcome == "quantitative":

            subgroups = fp_diver.get_pattern_divergence(
                min_support=min_support,
                quantitative_outcomes=[target_col],
                FPM_algorithm=args.FPM_algorithm,
            )
        else:
            subgroups = fp_diver.get_pattern_divergence(
                min_support=min_support,
                boolean_outcomes=[target_col],
                FPM_algorithm=args.FPM_algorithm,
            )

        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time}")
    except MemoryError:
        print(
            f"Memory limit exceeded! Minimum support {min_support} - dataset {args.dataset}"
        )
        continue

    number_subgroups[min_support] = len(subgroups)
    print(f"Number of subgroups: {len(subgroups)}")

    time_results[min_support] = execution_time

if number_subgroups == {}:
    print("No subgroups computed..")
    exit()

output_time_dir = os.path.join(args.output_dir, "time")

output_n_subgroups_dir = os.path.join(args.output_dir, "n_subgroups")

os.makedirs(output_time_dir, exist_ok=True)
os.makedirs(output_n_subgroups_dir, exist_ok=True)


filename_time = os.path.join(output_time_dir, f"time_{args.dataset}.pkl")

print(f"Saving time results to {filename_time}")

with open(filename_time, "wb") as f:
    pickle.dump(time_results, f)

filename_n_subgroups = os.path.join(
    output_n_subgroups_dir, f"n_subgroups_{args.dataset}.pkl"
)
print(f"Saving number of subgroups to {filename_n_subgroups}")

with open(filename_n_subgroups, "wb") as f:
    pickle.dump(number_subgroups, f)
