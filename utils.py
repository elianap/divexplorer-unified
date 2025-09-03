from sklearn.preprocessing import KBinsDiscretizer
import numpy as np


# COMPAS
# https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
# Quantize priors count between 0, 1-3, and >3
def quantizePrior(x):
    if x <= 0:
        return "0"
    elif 1 <= x <= 3:
        return "[1,3]"
    else:
        return ">3"


# Quantize length of stay
def quantizeLOS(x):
    if x <= 7:
        return "<week"
    if 8 < x <= 93:
        return "1w-3M"
    else:
        return ">3Months"


def discretize(
    df,
    bins=3,
    attributes=None,
    n_value_discretize=10,
    handle_nan=True,
    dataset_name=None,
    verbose=False,
    round_v=2,
):

    # Use all columns and all indexes if not specified
    attributes = df.columns if attributes is None else attributes

    continuous_attributes = [col for col in attributes if df[col].dtype != object]

    X_discretized = df[attributes].copy()

    if dataset_name == "adult":
        from utils_data import cap_gains_fn

        X_discretized["capital-gain"] = cap_gains_fn(
            X_discretized["capital-gain"].values
        )
        X_discretized["capital-gain"] = X_discretized["capital-gain"].replace(
            {0: "0", 1: "Low", 2: "High"}
        )
        X_discretized["capital-loss"] = cap_gains_fn(
            X_discretized["capital-loss"].values
        )
        X_discretized["capital-loss"] = X_discretized["capital-loss"].replace(
            {0: "0", 1: "Low", 2: "High"}
        )

        continuous_attributes.remove("capital-gain")
        continuous_attributes.remove("capital-loss")

    elif dataset_name == "bike_sharing":
        continuous_attributes = ["temp", "atemp", "hum", "windspeed"]
    elif dataset_name == "compas":
        X_discretized = df[attributes].copy()
        X_discretized["priors_count"] = X_discretized["priors_count"].apply(
            lambda x: quantizePrior(x)
        )
        X_discretized["length_of_stay"] = X_discretized["length_of_stay"].apply(
            lambda x: quantizeLOS(x)
        )
        continuous_attributes.remove("priors_count")
        continuous_attributes.remove("length_of_stay")

    if verbose:
        print(f"Discretizing: {continuous_attributes}")
    for col in continuous_attributes:
        for use_n_bins in range(bins, 1, -1):
            # Suppress warnings
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="sklearn.preprocessing"
                )

                discretized_series = discretize_column(
                    df,
                    col,
                    bins=use_n_bins,
                    n_value_discretize=n_value_discretize,
                    handle_nan=handle_nan,
                    round_v=round_v,
                )
            if discretized_series is not None:
                X_discretized[col] = discretized_series
                break
            print(
                f"Could not discretize '{col}' with {use_n_bins}. We decrease the number of bins to {use_n_bins-1}",
            )
    other_attributes = [a for a in df.columns if a not in attributes]
    X_discretized[other_attributes] = df[other_attributes].copy()
    return X_discretized


def bin_label(value, edges):
    """Helper function to assign bin labels based on bin edges."""
    if value <= edges[0]:
        return f"<={edges[0]}"
    for i in range(1, len(edges)):
        if edges[i - 1] < value <= edges[i]:
            return f"({edges[i - 1]}-{edges[i]}]"
    return f">{edges[-1]}"


def discretize_column(
    df, col, bins=3, n_value_discretize=10, handle_nan=True, round_v=2
):
    if df[col].nunique() > n_value_discretize:
        est = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
        if handle_nan:
            dt_values = df[~df[col].isnull()][[col]]
        else:
            dt_values = df[[col]]
        est.fit(dt_values)
        edges = est.bin_edges_[0][1:-1]
        if len([1 for edge in edges if abs(edge) < 1]) != len(edges):
            # If the edges are not all lower than 1, we convert the edges to integers for better readability
            if round_v == 0:
                edges = list(map(int, edges))
            else:
                edges = [round(edge, round_v) for edge in edges]

        # Remove duplicate edges if any
        edges = [
            edges[i]
            for i in range(len(edges))
            if i == len(edges) - 1 or edges[i] != edges[i + 1]
        ]
        if edges == []:
            return None

        # Assign bin labels
        discretized_series = df[col].apply(lambda x: bin_label(x, edges))
    else:
        discretized_series = df[col].astype(object)

    if handle_nan:
        discretized_series.loc[df[col].isnull()] = "NA"

    return discretized_series


def plot_comparison_ShapleyValues(
    sh_score_1,
    sh_score_2,
    toOrder=0,
    title=[],
    shared_items=True,
    shared_axis=False,
    height=0.8,
    linewidth=0.8,
    size_fig=(7, 7),
    save_fig=False,
    name_fig=None,
    labelsize=10,
    subcaption=False,
):
    import matplotlib.pyplot as plt

    sh_score_1 = {str(",".join(list(k))): v for k, v in sh_score_1.items()}
    sh_score_2 = {str(",".join(list(k))): v for k, v in sh_score_2.items()}

    if shared_items:
        if toOrder == 0:
            sh_score_1 = {
                k: v for k, v in sorted(sh_score_1.items(), key=lambda item: item[1])
            }
            sh_score_2 = {
                k: sh_score_2[k] for k in sorted(sh_score_1, key=sh_score_1.get)
            }
        elif toOrder == 1:
            sh_score_2 = {
                k: v for k, v in sorted(sh_score_2.items(), key=lambda item: item[1])
            }
            sh_score_1 = {
                k: sh_score_1[k] for k in sorted(sh_score_2, key=sh_score_2.get)
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size_fig)

    ax1.barh(
        range(len(sh_score_1)),
        sh_score_1.values(),
        align="center",
        color="#7CBACB",
        height=height,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    ax1.set_yticks(range(len(sh_score_1)), list(sh_score_1.keys()))
    ax1.tick_params(axis="y", labelsize=labelsize)
    ax1.tick_params(axis="x", labelsize=labelsize)

    if len(title) > 1:
        ax1.set_title(title[0], fontsize=labelsize)

    ax2.barh(
        range(len(sh_score_2)),
        sh_score_2.values(),
        align="center",
        color="#7CBACB",
        height=height,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    if shared_items:
        ax2.set_yticks(range(len(sh_score_2)), [])
    else:
        ax2.set_yticks(range(len(sh_score_2)), list(sh_score_2.keys()))
        ax2.tick_params(axis="y", labelsize=labelsize)
    ax2.tick_params(axis="x", labelsize=labelsize)

    if len(title) > 1:
        ax2.set_title(title[1], fontsize=labelsize)

    if shared_axis:
        sh_scores = list(sh_score_1.values()) + list(sh_score_2.values())
        min_x, max_x = min(sh_scores) + min(0.01, min(sh_scores)), max(sh_scores) + min(
            0.01, max(sh_scores)
        )

        ax1.set_xlim(min_x, max_x)
        ax2.set_xlim(min_x, max_x)

    s1 = "(a)" if subcaption else ""
    s2 = "(b)" if subcaption else ""
    ax1.set_xlabel(f"{s1}", size=labelsize)
    ax2.set_xlabel(f"{s2}", size=labelsize)

    plt.tight_layout()

    if save_fig:
        name_fig = "./shap.pdf" if name_fig is None else f"{name_fig}.pdf"
        plt.savefig(name_fig, format="pdf", bbox_inches="tight")
    plt.show()


def normalize_max(shap_values):
    maxv = abs(max(list(shap_values.values()), key=abs))
    shap_values_norm = {k: v / maxv for k, v in shap_values.items()}
    return shap_values_norm


def sort_itemset(x, abbreviations={}):
    x = list(x)
    x.sort()
    x = ", ".join(x)
    for k, v in abbreviations.items():
        x = x.replace(k, v)
    return x


def abbreviate_dict(d, abbreviations):
    # e.g. Shapley values (dict) as input
    # frozenset: value
    return {
        frozenset([sort_itemset(k, abbreviations=abbreviations)]): v
        for k, v in d.items()
    }


def printable(subgroups_print, target_col, abbreviations={}, use_k=False, round_v=2):
    from utils import sort_itemset

    subgroups_print["itemset"] = subgroups_print["itemset"].apply(
        lambda x: sort_itemset(x, abbreviations=abbreviations)
    )
    subgroups_print["support"] = subgroups_print["support"].apply(lambda x: round(x, 3))
    subgroups_print[f"{target_col}_t"] = subgroups_print[f"{target_col}_t"].apply(
        lambda x: round(x, 1)
    )

    if use_k:
        subgroups_print[f"{target_col}_div"] = subgroups_print[
            f"{target_col}_div"
        ].apply(lambda x: x / 1000)
        subgroups_print[f"{target_col}_div"] = subgroups_print[
            f"{target_col}_div"
        ].apply(lambda x: f"{round(x, 2)}k")
    else:
        subgroups_print[f"{target_col}_div"] = subgroups_print[
            f"{target_col}_div"
        ].apply(lambda x: round(x, round_v))

    remaining_cols = [
        col
        for col in subgroups_print.columns
        if col not in ["itemset", "support", f"{target_col}_t", f"{target_col}_div"]
    ]
    for col in remaining_cols:
        if use_k:
            subgroups_print[col] = subgroups_print[col].apply(lambda x: x / 1000)
            subgroups_print[col] = subgroups_print[col].apply(
                lambda x: f"{round(x, 2)}k"
            )
        else:
            subgroups_print[col] = subgroups_print[col].apply(
                lambda x: round(x, round_v)
            )
    return subgroups_print
