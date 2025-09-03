from utils_data import load_dataset_undiscretized
import pandas as pd

pd.set_option("display.max_colwidth", None)


# Set all the seeds for reproducibility
import numpy as np
import random

np.random.seed(42)
random.seed(42)

import argparse
import pandas as pd
import time
import os
from divexplorer import DivergenceExplorer
import pickle
from tqdm import tqdm


import multiprocessing
import time


def run_slice_finder_wrapper(queue, *args, **kwargs):
    from utils_rw import run_slice_finder

    """Run SliceFinder and put result in a queue."""
    try:
        result = run_slice_finder(*args, **kwargs)
        queue.put(result)
    except Exception as e:
        queue.put(e)


# Wrapper to adapt  run_slice_line for queue
def run_slice_line_wrapper(
    queue, df_discretized_analyze_sl, test_errors, min_support_count, sl_params
):
    from utils_rw import run_slice_line

    sl, end_time = run_slice_line(
        df_discretized_analyze_sl, test_errors, min_support_count, sl_params
    )
    queue.put((sl, end_time))


def run_with_timeout(func, timeout, *args, **kwargs):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=func, args=(queue, *args), kwargs=kwargs)
    p.start()
    p.join(timeout)  # wait at most timeout seconds
    if p.is_alive():
        print(f"{func.__name__} took more than {timeout} seconds, terminating.")
        p.terminate()  # kills the process and all child processes
        p.join()
        return None
    else:
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        return result


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
    default="heart",
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
    "-min_support",
    "--min_support",
    type=float,
    help="Set the minimum support value",
    default=0.1,
)


parser.add_argument(
    "-t",
    "--timeout_limit",
    type=int,
    help="Set the timeout limit in seconds. 1h as default",
    default=3600,
)

parser.add_argument(
    "-o", "--output_dir", type=str, help="Output directory", default="./results_rw"
)

args = parser.parse_args()

max_memory_gb = int(args.max_memory)

dataset_name = args.dataset

output_dir = args.output_dir

data_processed_dir = "./data/processed"
min_support = args.min_support

output_dir = os.path.join(args.output_dir, str(args.min_support))
os.makedirs(output_dir, exist_ok=True)

timeout_limit = int(args.timeout_limit)

from copy import deepcopy


results = {"divexplorer": {}, "sliceline": {}, "slicefinder": {}}

print("dataset_name", dataset_name)
df, attributes, target_col_name, type_outcome = load_dataset_undiscretized(dataset_name)

if dataset_name == "bike":
    cat_attr_list = ["season", "holiday", "weathersit", "workingday"]
    numeric_feature_cols = [
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "hr",
        "weekday",
        "mnth",
        "yr",
    ]
elif dataset_name == "airbnb":
    # We need to handle NaNs for the model evaluation
    df[attributes] = df[attributes].astype("str")
    df_filled = df.fillna("nan")
    cat_attr_list = attributes
    numeric_feature_cols = []
else:
    cat_attr_list = [col for col in attributes if df[col].dtype == "object"]
    numeric_feature_cols = [col for col in attributes if df[col].dtype != "object"]

# FORCE CONVERSION
if type_outcome == "quantitative":
    from utils import discretize_column

    df["class"] = discretize_column(df, target_col_name, bins=2)
    df["class"] = pd.factorize(df["class"])[0]
    df.drop(columns=[target_col_name])
    type_outcome = "boolean"
    target_col_name = "class"


# if dataset_name == "bike":
#     # Divide the dataset into training and testing sets - continuous splitting
#     index_end_train = int(len(df) * 0.7)
#     df_train, df_test = (
#         df[attributes].loc[0:index_end_train],
#         df[attributes].loc[index_end_train + 1 :],
#     )
# else:
from sklearn import model_selection
import numpy as np

np.random.seed(1)
df_train, df_test = model_selection.train_test_split(df[attributes], train_size=0.70)

y_train = df.loc[df_train.index, target_col_name]
y_test = df.loc[df_test.index, target_col_name]
from utils_data import encode_data

# We discretize

from utils import discretize

df_discretized = discretize(
    df[attributes],
    attributes=numeric_feature_cols,
    bins=3,
    round_v=0,
    dataset_name=dataset_name,
)

if dataset_name == "bank":
    # Problem in converting pdays and previous
    df_discretized["pdays"] = df_discretized["pdays"].apply(
        lambda x: "-1" if x > 0 else ">=0"
    )
    import numpy as np

    conditions = [
        df_discretized["previous"] == 0,
        df_discretized["previous"].between(1, 3, inclusive="both"),
        df_discretized["previous"] > 3,
    ]

    choices = ["0", "[1-3]", ">3"]

    df_discretized["previous"] = np.select(conditions, choices)

df_discretized_train, df_discretized_test = (
    df_discretized.loc[df_train.index],
    df_discretized.loc[df_test.index],
)

discretize_first = True
if discretize_first:  # Relevant for SliceFinder
    # We encode the data
    df_train_enc, df_test_enc, cat_encoders, scalers = encode_data(
        df_discretized_train, df_discretized_test, attributes
    )

else:

    df_train_enc, df_test_enc, cat_encoders, scalers = encode_data(
        df_train, df_test, cat_attr_list
    )


if type_outcome == "boolean":
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()

    model.fit(df_train_enc, y_train)

    from sklearn.metrics import accuracy_score

    y_test_pred = model.predict(df_test_enc)

    training_score = accuracy_score(y_test, y_test_pred)
else:
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor()

    model.fit(df_train_enc, y_train)

    from sklearn.metrics import mean_squared_error

    y_test_pred = model.predict(df_test_enc)
    training_score = mean_squared_error(y_test, y_test_pred)

print(training_score)

### Data preparation
y_test_pred_series = pd.Series(y_test_pred, index=df_test.index)


# DivExplorer
from divexplorer import DivergenceExplorer

df_discretized_analyze_de = df_discretized_test.copy()

if dataset_name == "bike":
    target_col = "mae"
    type_outcome = "quantitative"

    df_discretized_analyze_de[target_col] = abs(y_test - y_test_pred)

else:
    target_col = "acc"
    from divexplorer.outcomes import get_accuracy_outcome

    df_discretized_analyze_de[target_col] = get_accuracy_outcome(y_test, y_test_pred)

import time

fp_diver = DivergenceExplorer(df_discretized_analyze_de)

try:
    start_time = time.time()
    if type_outcome == "boolean":
        subgroups = fp_diver.get_pattern_divergence(
            min_support=min_support,
            boolean_outcomes=[target_col],
            FPM_algorithm="fpgrowth",
        )
    else:
        subgroups = fp_diver.get_pattern_divergence(
            min_support=min_support,
            quantitative_outcomes=[target_col],
            FPM_algorithm="fpgrowth",
        )
    end_time = time.time() - start_time
    print("--- %s seconds ---" % end_time)
except MemoryError:
    print("Memory limit exceeded!")
if type_outcome == "boolean":
    subgroups["error"] = 1 - subgroups["acc"]
    max_row = subgroups.loc[subgroups["error"].idxmax()]

    max_de = max_row["error"]
    len_of_max_de = max_row["length"]
else:
    max_row = subgroups.loc[subgroups[target_col].idxmax()]
    max_de = max_row[target_col]
    len_of_max_de = max_row["length"]
results["divexplorer"] = {
    "time": end_time,
    "number_subgroups": len(subgroups),
    "max_s": max_de,
    "len_of_max": len_of_max_de,
    "max_len": subgroups["length"].max(),
}
subgroups.sort_values(target_col).head(3)


###################################

# Slice Line
df_discretized_analyze_sl = df_discretized_test.copy()
df_discretized_analyze_sl[attributes] = df_discretized_analyze_sl[attributes].astype(
    str
)

test_errors = y_test != y_test_pred

###################################

## SLice line
df_discretized_analyze_sl = df_discretized_test.copy()
df_discretized_analyze_sl[attributes] = df_discretized_analyze_sl[attributes].astype(
    str
)

test_errors = y_test != y_test_pred

min_support_count = min_support * len(df_discretized_analyze_sl)

sl_params = {"alpha": 0.95, "k": 100, "max_l": len(attributes)}

# I want that if the method does not finish in timeout_limit seconds, it is killed
sl, end_time = run_with_timeout(
    run_slice_line_wrapper,
    timeout_limit,
    df_discretized_analyze_sl,
    test_errors,
    min_support_count,
    sl_params,
)

if sl is not None:

    from utils_rw import encode_slices_df_with_metric_sl

    sl_subgroups_df = encode_slices_df_with_metric_sl(
        sl, attributes, df_discretized_analyze_sl, y_test, y_test_pred_series
    )

    # We only use the error metric for comparison
    metric_name = "error"

    if len(sl_subgroups_df) > 0:
        max_row = sl_subgroups_df.loc[sl_subgroups_df[metric_name].idxmax()]
        max_sl = max_row[metric_name]
        len_of_max_sl = max_row["length"]
        max_len = sl_subgroups_df["length"].max()

    else:
        max_row, max_sl, len_of_max_sl = 0, 0, 0


results["sliceline"] = {
    "time": end_time,
    "number_subgroups": len(sl_subgroups_df),
    "slf_params": sl_params,
    "max_s": max_sl,
    "len_of_max": len_of_max_sl,
    "max_len": max_len,
}


###################################
# Slice Finder
df_discretized_analyze_sf = df_discretized_test.copy()
df_discretized_analyze_sf[attributes] = df_discretized_analyze_sf[attributes].astype(
    str
)


sf_params = {"degree": 3, "k": 100}


recommendations_end_time = run_with_timeout(
    run_slice_finder_wrapper,
    timeout_limit,
    df_test_enc,
    y_test,
    model,
    degree=sf_params["degree"],
    k=sf_params["k"],
)

if recommendations_end_time is not None:
    recommendations, end_time_sf = recommendations_end_time

    # df_slices = encode_slices_df(recommendations, cat_encoders)

    from utils_rw import encode_slices_df_with_metric_sf

    sf_slices_df = encode_slices_df_with_metric_sf(
        recommendations,
        cat_encoders,
        df_discretized_analyze_sf,
        y_test,
        y_test_pred_series,
    )

    metric_name = "error"

    max_overall, len_of_max_overall_sf, max_sf, len_of_max_sf = 0, 0, 0, 0

    if len(sf_slices_df) > 0:

        max_row_overall = sf_slices_df.loc[sf_slices_df[metric_name].idxmax()]

        max_overall = max_row_overall[metric_name]
        len_of_max_overall_sf = max_row_overall["length"]

        filtered_above_support = sf_slices_df.loc[
            sf_slices_df["size"] >= min_support_count
        ]

        if len(filtered_above_support) > 0:
            max_row_above_supp = filtered_above_support.loc[
                filtered_above_support[metric_name].idxmax()
            ]

            max_sf = max_row_above_supp[metric_name]
            len_of_max_sf = max_row_above_supp["length"]

    results["slicefinder"] = {
        "time": end_time_sf,
        "number_subgroups": len(sf_slices_df),
        "sf_params": sf_params,
        "max_s": max_sf,
        "max_overall": max_overall,
        "len_of_max": len_of_max_sf,
        "max_len": max_sf,
        "len_of_max_overall_sf": len_of_max_overall_sf,
    }
else:
    results["slicefinder"] = {
        "time": f"> {timeout_limit} sec",
        "number_subgroups": 0,
        "sf_params": sf_params,
        "max_s": "timeout",
        "max_overall": "timeout",
        "len_of_max": 0,
        "max_len": 0,
        "len_of_max_overall_sf": 0,
    }
#########################

filename = os.path.join(output_dir, f"dict_rw_results_{dataset_name}.pkl")

print(f"Saving results to {filename}")

with open(filename, "wb") as f:
    pickle.dump(results, f)
