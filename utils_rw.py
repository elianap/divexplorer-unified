import pandas as pd
import time
import numpy as np


def getSliceFrozenset(s, cat_encoders={}):
    slice_i = []
    for k, v in list(s.filters.items()):
        values = ""
        if k in cat_encoders:
            cat_enc = cat_encoders[k]
            for v_ in v:
                values += "%s" % (cat_enc.inverse_transform(v_)[0])

        else:
            for v_ in sorted(v, key=lambda x: x[0]):
                if len(v_) > 1:
                    values += "%s ~ %s" % (v_[0], v_[1])
                else:
                    values += "%s" % (v_[0])

        slice_i.append("%s=%s" % (k, values))
    return frozenset(slice_i)


def convert_slice_features_and_values(s, cat_encoders):
    feature_names, feature_values = [], []
    for feature_name, feature_value in list(s.filters.items()):
        feature_names.append(feature_name)
        values = ""
        if feature_name in cat_encoders:
            cat_enc = cat_encoders[feature_name]
            for v_ in feature_value:
                values += "%s" % (cat_enc.inverse_transform(v_)[0])

        else:
            for v_ in sorted(feature_value, key=lambda x: x[0]):
                if len(v_) > 1:
                    values += "%s ~ %s" % (v_[0], v_[1])
                else:
                    values += "%s" % (v_[0])
        feature_values.append(values)
    return feature_names, feature_values


def run_slice_finder(df_test_enc, y_test, model, degree=3, k=100):
    import warnings

    warnings.filterwarnings(
        action="ignore", message="X does not have valid feature names"
    )

    from slice_finder import SliceFinder

    df_test_enc_sf = df_test_enc.copy()
    df_test_enc_sf = df_test_enc_sf.reset_index(drop=True)
    y_test_sd = y_test.copy()
    y_test_sd = y_test_sd.reset_index(drop=True)

    start_time = time.time()
    sf = SliceFinder(model, (df_test_enc_sf, y_test_sd))

    recommendations = sf.find_slice(degree=degree, k=k)

    end_time = time.time() - start_time

    print("--- %s seconds ---" % end_time)
    return recommendations, end_time


def encode_slices_df(recommendations, cat_encoders):

    rows = []
    for s in recommendations:
        rows.append([getSliceFrozenset(s, cat_encoders), s.metric, s.effect_size])
    df_slices = pd.DataFrame(rows, columns=["slice", "log_loss", "effect_size"])
    return df_slices


def encode_slices_df_with_metric_sf(
    recommendations, cat_encoders, df_discretized_analyze, y_test, y_test_pred_series
):
    sf_slices = []

    for s in recommendations:
        feature_names, feature_values = convert_slice_features_and_values(
            s, cat_encoders
        )

        predicate_conditions = [
            df_discretized_analyze[feature_name] == feature_value
            for feature_name, feature_value in zip(feature_names, feature_values)
            if feature_value is not None
        ]
        condition = " & ".join(
            [f"@predicate_conditions[{i}]" for i in range(len(predicate_conditions))]
        )
        slice_frozenset = frozenset(
            [
                f"{feature_name}={feature_value}"
                for feature_name, feature_value in zip(feature_names, feature_values)
            ]
        )

        indices = df_discretized_analyze.query(condition).index

        if len(indices) == 0:
            print(s)

        from sklearn.metrics import accuracy_score

        metric_sl = 1 - accuracy_score(y_test[indices], y_test_pred_series[indices])

        sf_slices.append(
            [slice_frozenset, s.size, len(indices), metric_sl, len(feature_names)]
        )

    sf_slices_df = pd.DataFrame(
        sf_slices, columns=["slice", "sizesf", "size", "error", "length"]
    )
    return sf_slices_df


#


def encode_slices_df_with_metric_sl(
    sl, attributes, df_discretized_analyze, y_test, y_test_pred_series
):
    sl_subgroups_df = pd.DataFrame(
        sl.top_slices_, columns=sl.feature_names_in_, index=sl.get_feature_names_out()
    )
    sl_subgroups_df["length"] = sl_subgroups_df[attributes].notna().sum(axis=1)

    metric_slices_sl = []
    for slice_index in range(sl_subgroups_df.shape[0]):
        current_slice = sl.top_slices_[slice_index]

        predicate_conditions = [
            df_discretized_analyze[feature_name] == feature_value
            for feature_name, feature_value in zip(sl.feature_names_in_, current_slice)
            if feature_value is not None
        ]
        condition = " & ".join(
            [f"@predicate_conditions[{i}]" for i in range(len(predicate_conditions))]
        )

        indices = df_discretized_analyze.query(condition).index
        from sklearn.metrics import accuracy_score

        metric_slices_sl.append(
            1 - accuracy_score(y_test[indices], y_test_pred_series[indices])
        )
    metric_name = "error"
    sl_subgroups_df[metric_name] = metric_slices_sl
    return sl_subgroups_df


def run_slice_line(
    df_discretized_analyze_sl, test_errors, min_support_count, sl_params
):
    from sliceline.slicefinder import Slicefinder

    start_time = time.time()

    try:
        sl = Slicefinder(
            alpha=sl_params["alpha"],
            k=sl_params["k"],
            max_l=sl_params["max_l"],
            min_sup=round(min_support_count),
            verbose=False,
        )

        sl.fit(df_discretized_analyze_sl, test_errors)
        end_time = time.time() - start_time

        print("--- %s seconds ---" % end_time)

    except MemoryError:
        print("Memory limit exceeded!")
    return sl, end_time
