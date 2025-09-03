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
    "-th_reds",
    "--th_redundancy_perc_values",
    type=float,
    nargs="+",
    help="Set the percentage values",
    default=[
        0,
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.0075,
        0.01,
        0.02,
        0.025,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
    ],
)

parser.add_argument(
    "-alg",
    "--FPM_algorithm",
    type=str,
    help="Algorithm for frequent pattern mining",
    default="fpgrowth",
)

parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Output directory",
    default="./results/redundancy_pruning",
)

args = parser.parse_args()

max_memory_gb = int(args.max_memory)

dataset_name = args.dataset

th_redundancy_perc_values = args.th_redundancy_perc_values

data_processed_dir = "./data/processed"


from utils_data import load_dataset

df_discretized, target_col, type_outcome = load_dataset(dataset_name, use_stored=True)


if type_outcome not in ["quantitative", "boolean"]:
    raise ValueError(f"Outcome type not supported: {type_outcome}")

fp_diver = DivergenceExplorer(df_discretized)

# Limit the notebook to max_memory_gb GB of memory
limit_memory(max_memory_gb * 1024)

number_subgroups = {}


for min_support in tqdm(args.min_supports):
    print("min_support", min_support)
    min_support = float(min_support)

    for th_perc in [None] + th_redundancy_perc_values:
        print("\nth_perc", th_perc)

        if th_perc is None:
            # This is the first iteration. We also compute the subgroups
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

                max_v = subgroups[target_col].max()

                from divexplorer import DivergencePatternProcessor

                fp_details = DivergencePatternProcessor(subgroups, target_col)

            except MemoryError:
                print(
                    f"Memory limit exceeded! Minimum support {min_support} - dataset {args.dataset}"
                )

        th_redundancy = th_perc if th_perc is None else th_perc * max_v
        th_redundancy_str = "None" if th_perc is None else str(th_perc)

        if th_redundancy_str not in number_subgroups:
            number_subgroups[th_redundancy_str] = {}

        try:
            patterns = fp_details.get_patterns(th_redundancy=th_redundancy)
        except MemoryError:
            print("Memory limit exceeded!")

        print(
            "#patterns",
            len(patterns),
            "th_redundancy",
            th_redundancy,
            "th_perc",
            th_perc,
            "min_support",
            min_support,
        )
        if th_redundancy_str not in number_subgroups:
            number_subgroups[th_redundancy_str] = {}

        number_subgroups[th_redundancy_str][min_support] = len(patterns)


if number_subgroups == {}:
    print("No subgroups computed..")
    exit()


output_n_subgroups_dir = os.path.join(args.output_dir, "n_subgroups")

os.makedirs(output_n_subgroups_dir, exist_ok=True)


filename_n_subgroups = os.path.join(
    output_n_subgroups_dir, f"n_subgroups_{args.dataset}.pkl"
)
print(f"Saving number of subgroups to {filename_n_subgroups}")

with open(filename_n_subgroups, "wb") as f:
    pickle.dump(number_subgroups, f)
