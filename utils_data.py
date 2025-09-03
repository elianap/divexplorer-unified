import pandas as pd
import numpy as np
import os

DATASET_DIR = os.path.join(".", "data")


def import_process_airbnb(input_dir=DATASET_DIR, use_stored=True):
    filename = os.path.join(input_dir, "airbnb_listings_discretized.csv")
    if use_stored and os.path.exists(filename):
        df_discretized = pd.read_csv(filename)
        target_col = "review_scores_rating"
        return df_discretized, target_col
    else:
        print("TODO add here the code to import the dataset")
        raise ValueError("File not found")


def import_process_german(input_dir=DATASET_DIR):
    df = pd.read_csv(os.path.join(input_dir, "credit-g.csv"))
    # print(df['personal_status'].value_counts())
    gender_map = {
        "'male single'": "male",
        "'female div/dep/mar'": "female",
        "'male mar/wid'": "male",
        "'male div/sep'": "male",
    }
    status_map = {
        "'male single'": "single",
        "'female div/dep/mar'": "married/wid/sep",
        "'male mar/wid'": "married/wid/sep",
        "'male div/sep'": "married/wid/sep",
    }
    df["sex"] = df["personal_status"].replace(gender_map)
    df["civil_status"] = df["personal_status"].replace(status_map)
    df.drop(columns=["personal_status"], inplace=True)
    df.rename(columns={"credit": "class"}, inplace=True)
    df["class"] = df["class"].map({"good": 1, "bad": 0})
    return df


def import_process_adult(input_dir=DATASET_DIR):
    education_map = {
        "10th": "Dropout",
        "11th": "Dropout",
        "12th": "Dropout",
        "1st-4th": "Dropout",
        "5th-6th": "Dropout",
        "7th-8th": "Dropout",
        "9th": "Dropout",
        "Preschool": "Dropout",
        "HS-grad": "High School grad",
        "Some-college": "High School grad",
        "Masters": "Masters",
        "Prof-school": "Prof-School",
        "Assoc-acdm": "Associates",
        "Assoc-voc": "Associates",
    }
    occupation_map = {
        "Adm-clerical": "Admin",
        "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar",
        "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar",
        "Handlers-cleaners": "Blue-Collar",
        "Machine-op-inspct": "Blue-Collar",
        "Other-service": "Service",
        "Priv-house-serv": "Service",
        "Prof-specialty": "Professional",
        "Protective-serv": "Other",
        "Sales": "Sales",
        "Tech-support": "Other",
        "Transport-moving": "Blue-Collar",
    }
    married_map = {
        "Never-married": "Never-Married",
        "Married-AF-spouse": "Married",
        "Married-civ-spouse": "Married",
        "Married-spouse-absent": "Separated",
        "Separated": "Separated",
        "Divorced": "Separated",
        "Widowed": "Widowed",
    }

    country_map = {
        "Cambodia": "SE-Asia",
        "Canada": "British-Commonwealth",
        "China": "China",
        "Columbia": "South-America",
        "Cuba": "Other",
        "Dominican-Republic": "Latin-America",
        "Ecuador": "South-America",
        "El-Salvador": "South-America",
        "England": "British-Commonwealth",
        "France": "Euro_1",
        "Germany": "Euro_1",
        "Greece": "Euro_2",
        "Guatemala": "Latin-America",
        "Haiti": "Latin-America",
        "Holand-Netherlands": "Euro_1",
        "Honduras": "Latin-America",
        "Hong": "China",
        "Hungary": "Euro_2",
        "India": "British-Commonwealth",
        "Iran": "Other",
        "Ireland": "British-Commonwealth",
        "Italy": "Euro_1",
        "Jamaica": "Latin-America",
        "Japan": "Other",
        "Laos": "SE-Asia",
        "Mexico": "Latin-America",
        "Nicaragua": "Latin-America",
        "Outlying-US(Guam-USVI-etc)": "Latin-America",
        "Peru": "South-America",
        "Philippines": "SE-Asia",
        "Poland": "Euro_2",
        "Portugal": "Euro_2",
        "Puerto-Rico": "Latin-America",
        "Scotland": "British-Commonwealth",
        "South": "Euro_2",
        "Taiwan": "China",
        "Thailand": "SE-Asia",
        "Trinadad&Tobago": "Latin-America",
        "United-States": "United-States",
        "Vietnam": "SE-Asia",
    }
    # as given by adult.names
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income-per-year",
    ]
    train = pd.read_csv(
        os.path.join(input_dir, "adult.data"),
        header=None,
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )
    test = pd.read_csv(
        os.path.join(input_dir, "adult.test"),
        header=0,
        names=column_names,
        skipinitialspace=True,
        na_values="?",
    )
    dt = pd.concat([test, train], ignore_index=True)
    dt["education"] = dt["education"].replace(education_map)
    dt.drop(columns=["education-num", "fnlwgt"], inplace=True)
    dt["occupation"] = dt["occupation"].replace(occupation_map)
    dt["marital-status"] = dt["marital-status"].replace(married_map)
    dt["native-country"] = dt["native-country"].replace(country_map)

    dt.rename(columns={"income-per-year": "class"}, inplace=True)
    dt["class"] = (
        dt["class"].astype("str").replace({">50K.": ">50K", "<=50K.": "<=50K"})
    )
    dt.dropna(inplace=True)
    dt.reset_index(drop=True, inplace=True)

    dt.drop(columns=["native-country"], inplace=True)
    dt["class"] = dt["class"].map({">50K": 1, "<=50K": 0})
    return dt


def cap_gains_fn(x):
    x = x.astype(float)
    d = np.digitize(
        x, [0, np.median(x[x > 0]), float("inf")], right=True
    )  # .astype('|S128')
    return d.copy()


def import_process_compas(
    input_dir=DATASET_DIR, use_string_format=False, reset_index=True
):
    import pandas as pd

    df_raw = pd.read_csv(os.path.join(input_dir, "compas-scores-two-years.csv"))
    cols_propb = [
        "c_charge_degree",
        "race",
        "age_cat",
        "sex",
        "priors_count",
        "days_b_screening_arrest",
        "two_year_recid",
    ]  # , "is_recid"]

    df = df_raw[cols_propb]
    # Warning
    df["length_of_stay"] = pd.to_timedelta(
        pd.to_datetime(df_raw["c_jail_out"]).dt.date
        - pd.to_datetime(df_raw["c_jail_in"]).dt.date
    ).dt.days

    df = df.loc[abs(df["days_b_screening_arrest"]) <= 30]

    df = df.loc[df["c_charge_degree"] != "O"]  # F: felony, M: misconduct
    columns = [
        "age_cat",
        "c_charge_degree",
        "race",
        "sex",
        "two_year_recid",
        "priors_count",
        "length_of_stay",
    ]

    df = df[columns]
    df.rename(columns={"two_year_recid": "class"}, inplace=True)
    df["predicted"] = df_raw["decile_score"].apply(
        get_decile_score_class, use_string_format=use_string_format
    )
    if reset_index:
        df.reset_index(drop=True, inplace=True)
    return df


def get_decile_score_class(x, use_string_format=False):
    if x >= 8:
        return "High" if use_string_format else 1
    else:
        return "Medium-Low" if use_string_format else 0


# https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model
def import_process_heart(input_dir=DATASET_DIR):

    dt = pd.read_csv(os.path.join(input_dir, "heart.csv"))
    dt.columns = [
        "age",
        "sex",
        "chest_pain_type",
        "resting_blood_pressure",
        "cholesterol",
        "fasting_blood_sugar",
        "rest_ecg",
        "max_heart_rate_achieved",
        "exercise_induced_angina",
        "st_depression",
        "st_slope",
        "num_major_vessels",
        "thalassemia",
        "class",
    ]

    dt["sex"] = dt["sex"].map({0: "female", 1: "male"}).astype("object")

    dt["chest_pain_type"] = (
        dt["chest_pain_type"]
        .map(
            {
                0: "typical angina",
                1: "atypical angina",
                2: "non-anginal pain",
                3: "asymptomatic",
            }
        )
        .astype("object")
    )

    dt["fasting_blood_sugar"] = (
        dt["fasting_blood_sugar"].map({0: "<120mg/ml", 1: ">120mg/ml"}).astype("object")
    )

    dt["rest_ecg"] = (
        dt["rest_ecg"]
        .map(
            {0: "normal", 1: "ST-T wave abnormality", 2: "left ventricular hypertrophy"}
        )
        .astype("object")
    )

    dt["exercise_induced_angina"] = (
        dt["exercise_induced_angina"].map({0: "no", 1: "yes"}).astype("object")
    )

    dt["st_slope"] = (
        dt["st_slope"]
        .map({0: "upsloping", 1: "flat", 2: "downsloping"})
        .astype("object")
    )

    dt["thalassemia"] = (
        dt["thalassemia"]
        .map({1: "normal", 2: "fixed defect", 3: "reversable defect"})
        .astype("object")
    )

    dt = dt[dt.thalassemia != 0]

    dt = dt[dt.num_major_vessels != 4]

    dt["class"] = dt["class"].astype("int")
    dt["num_major_vessels"] = dt["num_major_vessels"].astype("object")
    return dt


def import_process_law(
    input_dir="./data", alpha=0.1, use_rank_score=True, use_rank=False, use_ZFYA=False
):
    def score_from_ranking(df, rank_column="rank", alpha=0.1):
        import numpy as np

        return 1 / np.power(df[rank_column], float(alpha))

    if sum([use_rank_score, use_rank, use_ZFYA]) != 1:
        raise ValueError(
            "Exactly one of use_rank_score, use_rank, or use_ZFYA must be True."
        )

    df = pd.read_csv(os.path.join(input_dir, "law_students_ranking.csv"))

    if use_ZFYA:
        target_col = "ZFYA"
        attributes = list(df.columns.drop(["ZFYA", "rank"]))
        df = df[attributes + ["ZFYA"]]
        return df, target_col
    attributes = list(df.columns.drop(["ZFYA", "rank"]))
    df = df[attributes + ["rank"]]

    if use_rank_score:
        df["score_rank"] = score_from_ranking(df, alpha=alpha)
        df.drop(columns=["rank"], inplace=True)
        target_col = "score_rank"
    if use_rank:
        target_col = "rank"

    return df, target_col


DATASET_DIR = os.path.join(".", "data")


def import_process_folktables(input_dir="./data", store_data=True, year="2023"):

    from folktables import ACSDataSource, ACSIncome
    import pandas as pd

    data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=["CA"], download=True)

    from folktables.acs import adult_filter
    from folktables import BasicProblem
    import numpy as np

    feature_names = [
        "AGEP",
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "RELP" if year in ["2018"] else "RELSHIPP",
        "WKHP",
        "SEX",
        "RAC1P",
    ]

    target = "PINCP"

    ACSIncome_outcome = BasicProblem(
        features=feature_names,
        target=target,
        group="RAC1P",
        preprocess=adult_filter,
    )

    features, label, group = ACSIncome_outcome.df_to_numpy(acs_data)

    filename = os.path.join(input_dir, f"PUMS_Data_Dictionary_{year}.csv")
    df_mappings = pd.read_csv(filename)

    continuous_attributes = ["AGEP", "WKHP"]
    categorical_attributes = list(set(feature_names) - set(continuous_attributes))

    df = pd.DataFrame(features, columns=feature_names)
    df["income"] = label

    remapping_cols = {}

    orig_col = "1"
    new_col = "Record Type"
    cols_i = [orig_col, new_col]

    # col_name = "OCCP"
    for col_name in categorical_attributes:
        dict_i = (
            df_mappings.loc["VAL"]
            .loc[col_name][cols_i]
            .set_index(orig_col)
            .to_dict()[new_col]
        )
        dict_i = {
            float(k) if (k not in ["b", "bb", "bbb", "bbbb"]) else -1: v
            for k, v in dict_i.items()
        }
        remapping_cols[col_name] = dict_i

    for column_name in remapping_cols:
        df[column_name] = df[column_name].replace(remapping_cols[column_name])

    for c in df:
        if df[c].isna().any():
            df[c].fillna("NaN", inplace=True)

    if store_data:
        import pickle

        with open(
            os.path.join(input_dir, f"census_column_mapping_{year}.pickle"),
            "wb",
        ) as fp:
            pickle.dump(remapping_cols, fp)

        df.to_csv(os.path.join(input_dir, f"folktables_{year}.csv"), index=False)

    target_col = "income"

    return df, target_col


def import_process_bank(input_dir=DATASET_DIR):
    dt = pd.read_csv(os.path.join(input_dir, "datasets_4471_6849_bank.csv"), sep=",")
    dt.drop(columns=["duration"], inplace=True)
    dt.rename(columns={"deposit": "class"}, inplace=True)
    dt["class"] = dt["class"].map({"yes": 1, "no": 0})
    return dt


DATASET_DIR = "./data"


def transform_bike_renting_dataset(df):
    # Weekdays
    df["weekday"] = df["weekday"].replace(
        {i: v for i, v in enumerate(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])}
    )

    # Seasons
    df["season"] = df["season"].replace(
        {i + 1: v for i, v in enumerate(["winter", "spring", "summer", "fall"])}
    )

    df["weathersit"] = df["weathersit"].replace(
        {i + 1: v for i, v in enumerate(["good", "misty", "rain/snow/storm"])}
    )

    df["yr"] = df["yr"].replace({0: 2011, 1: 2021})

    if "dteday" in df.columns:
        df["days_since_2011"] = pd.to_datetime(df["dteday"]) - pd.to_datetime(
            min(df["dteday"])
        )

    # temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
    df["temp"] = df["temp"] * (39 - (-8)) + (-8)

    # atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
    df["atemp"] = df["atemp"] * (50 - (16)) + (16)

    # windspeed: Normalized wind speed. The values are divided to 67 (max)
    df["windspeed"] = 67 * df["windspeed"]

    # hum: Normalized humidity. The values are divided to 100 (max)
    df["hum"] = 100 * df["hum"]

    return df


def import_process_bike_sharing_data(input_dir=DATASET_DIR, store_data=False):
    # https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset#

    df_input_hour = pd.read_csv(os.path.join(input_dir, "bike-sharing/hour.csv"))

    cols_not_interested = ["casual", "registered", "instant", "dteday"]

    target_col = "cnt"

    # making feature list for each modeling - experiment by adding feature to the exclusion list
    attributes = [
        attribute
        for attribute in df_input_hour.columns
        if attribute not in [target_col] + cols_not_interested
    ]

    df_input_hour = transform_bike_renting_dataset(
        df_input_hour[attributes + [target_col]].copy()
    )

    if store_data:
        df_input_hour.to_csv(os.path.join(input_dir, "bike-sharing.csv"), index=False)
    return df_input_hour, "cnt"


def train_predict(
    dfI,
    labelEncoding=True,
    validation="train",
    k_cv=10,
    args={},
    retClf=False,
    fold="stratified",
):
    # Attributes with one hot encoding (attribute=value)
    import numpy as np

    attributes = dfI.columns.drop("class")
    X = dfI[attributes].copy()
    y = dfI[["class"]].copy()
    encoders = {}
    if labelEncoding:
        from sklearn.preprocessing import LabelEncoder

        encoders = {}
        for column in attributes:
            if dfI.dtypes[column] == object:
                le = LabelEncoder()
                X[column] = le.fit_transform(dfI[column])
                encoders[column] = le

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(random_state=42, **args)

    if validation != "cv":
        # Data for training and testing
        if validation == "all":
            X_train, y_train = X, y
        else:
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )  # stratify=y
        clf.fit(X_train, y_train.values.ravel())
        X_FP, y_FP = (X_test, y_test) if validation == "test" else (X_train, y_train)

        y_predicted = clf.predict(X_FP)
    else:
        from sklearn.model_selection import cross_val_predict

        if fold == "stratified":
            from sklearn.model_selection import StratifiedKFold

            cv = StratifiedKFold(
                n_splits=k_cv,
                random_state=42,
                shuffle=True,  # Shuffle=True for sklearn 0.24.2
            )  # Added to fix the random state
        else:
            from sklearn.model_selection import KFold

            cv = KFold(n_splits=k_cv, random_state=42)  # Added to fix the random state
        y_predicted = cross_val_predict(clf, X, y["class"].values.ravel(), cv=cv)
        X_FP, y_FP = X, y

    y_predict_prob = None
    if validation != "cv":
        y_predict_prob = clf.predict_proba(X_FP)
    else:
        y_predict_prob = cross_val_predict(
            clf, X, y["class"].values.ravel(), cv=cv, method="predict_proba"
        )

    indexes_FP = X_FP.index
    # X_FP.reset_index(drop=True, inplace=True)
    # y_FP.reset_index(drop=True, inplace=True)

    return X_FP, y_FP, y_predicted, y_predict_prob, encoders, indexes_FP


def import_artificial(
    input_dir=DATASET_DIR,
    store_data=False,
    use_stored=False,
):
    if use_stored:
        filename = os.path.join(input_dir, "artificial_10.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print("Loading artificial dataset")
            return df
    features_3 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

    data = np.concatenate([np.full((25000, 10), 0), np.full((25000, 10), 1)])
    np.random.seed(7)
    for i in range(data.shape[1]):
        np.random.shuffle(data[:, i])

    df_artificial_3 = pd.DataFrame(data=data, columns=features_3)
    df_artificial_3["class"] = 0
    indexes = df_artificial_3.loc[
        (
            (df_artificial_3["a"] == df_artificial_3["b"])
            & (df_artificial_3["a"] == df_artificial_3["c"])
        )
    ].index
    df_artificial_3.loc[indexes, "class"] = 1
    df_artificial_3["class"].value_counts()

    import random

    indexes_class1 = list(indexes)
    random.Random(7).shuffle(indexes_class1)
    df_artificial_3["predicted"] = df_artificial_3["class"]
    df_artificial_3.loc[indexes_class1[: int(len(indexes_class1) / 2)], "class"] = 0

    if store_data:
        df_artificial_3.to_csv(
            os.path.join(input_dir, "artificial_10.csv"), index=False
        )

    return df_artificial_3


def load_dataset(
    dataset_name,
    data_processed_dir=f"{DATASET_DIR}/processed",
    use_stored=True,
    store_data_pred_results=True,
):
    """
    Load and process dataset
    Args:
        dataset_name (str): name of the dataset
        data_processed_dir (str): directory where to store or load the processed data
        use_stored (bool): if True, load the stored data if it exists
        store_data_pred_results (bool): if True, store the data with the predictions
    Returns:
        df_discretized (pd.DataFrame): processed dataset
        target_col (str): name of the target column
        type_outcome (str): type of the outcome variable
    """
    if dataset_name in ["german", "adult", "heart", "bank"]:
        filename = os.path.join(data_processed_dir, f"{dataset_name}.csv")
        if use_stored and os.path.exists(filename):
            print(f"Loading {filename} - {dataset_name}")
            df_discretized = pd.read_csv(filename, index_col=0)
        else:
            # Import and process data
            print(f"Processing {dataset_name}")
            if dataset_name == "german":
                from utils_data import import_process_german

                df = import_process_german()
            elif dataset_name == "adult":
                from utils_data import import_process_adult

                df = import_process_adult()
            elif dataset_name == "heart":
                from utils_data import import_process_heart

                df = import_process_heart()
            elif dataset_name == "bank":
                from utils_data import import_process_bank

                df = import_process_bank()
            else:
                raise ValueError(f"Dataset {dataset_name} not available")

            from utils_data import train_predict

            X_FP, y_FP, y_predicted, _, _, indexes_FP = train_predict(
                df, validation="cv"
            )

            attributes = df.columns.drop("class")

            from utils import discretize

            df_discretized = discretize(
                df.loc[X_FP.index],
                attributes=attributes,
                dataset_name=dataset_name,
                round_v=2,
            )
            df_discretized["class"] = y_FP
            df_discretized["predicted"] = y_predicted

            if store_data_pred_results:
                os.makedirs(data_processed_dir, exist_ok=True)
                df_discretized.to_csv(
                    os.path.join(data_processed_dir, f"{dataset_name}.csv")
                )

        from divexplorer.outcomes import get_false_positive_rate_outcome

        target_col = "fp"

        y_trues = df_discretized["class"]
        y_preds = df_discretized["predicted"]

        df_discretized[target_col] = get_false_positive_rate_outcome(y_trues, y_preds)

        df_discretized.drop(columns=["class", "predicted"], inplace=True)

        type_outcome = "boolean"
    elif dataset_name == "airbnb":
        from utils_data import import_process_airbnb

        df_discretized, target_col = import_process_airbnb()
        type_outcome = "quantitative"
    elif dataset_name == "law":
        from utils_data import import_process_law

        df, target_col = import_process_law(use_rank_score=True)
        attributes = df.columns.drop(target_col)
        from utils import discretize

        df_discretized = discretize(
            df, attributes=attributes, dataset_name=dataset_name, bins=4
        )
        type_outcome = "quantitative"
    elif dataset_name in ["compas", "artificial"]:
        if dataset_name == "compas":
            from utils_data import import_process_compas

            df = import_process_compas()
        else:  # dataset_name == "artificial":
            from utils_data import import_artificial

            df = import_artificial(use_stored=use_stored)

        from utils import discretize

        attributes = list(df.columns)
        attributes.remove("class")
        attributes.remove("predicted")
        df_discretized = discretize(
            df, dataset_name=dataset_name, attributes=attributes
        )
        target_col = "fp"
        y_trues = df_discretized["class"]
        y_preds = df_discretized["predicted"]
        from divexplorer.outcomes import get_false_positive_rate_outcome

        df_discretized[target_col] = get_false_positive_rate_outcome(y_trues, y_preds)
        df_discretized.drop(columns=["class", "predicted"], inplace=True)
        type_outcome = "boolean"
    elif dataset_name == "bike":
        from utils import discretize

        df, target_col = import_process_bike_sharing_data()

        continuous_attributes = ["temp", "atemp", "hum", "windspeed"]
        df_discretized = discretize(
            df, attributes=continuous_attributes, bins=3, round_v=0
        )
        type_outcome = "quantitative"
    elif dataset_name == "folktables":
        from utils import discretize

        year = "2023"
        filename = os.path.join("./data", f"folktables_{year}.csv")
        if os.path.exists(filename):
            df = pd.read_csv(os.path.join("./data", f"folktables_{year}.csv"))
            target_col = "income"
        else:

            df, target_col = import_process_folktables(
                input_dir="./data", store_data=True, year=year
            )
        df = df.loc[df["AGEP"] >= 18]
        attributes = list(df.columns)
        attributes.remove(target_col)
        df_discretized = discretize(df, attributes=attributes)
        type_outcome = "quantitative"
    else:
        raise ValueError("Dataset not found")
    return df_discretized, target_col, type_outcome


def load_dataset_undiscretized(
    dataset_name,
    data_processed_dir=f"{DATASET_DIR}/processed",
):
    """
    Load and process dataset without processing of DivExplorer
    Args:
        dataset_name (str): name of the dataset
        data_processed_dir (str): directory where to store or load the processed data
        use_stored (bool): if True, load the stored data if it exists
        store_data_pred_results (bool): if True, store the data with the predictions
    Returns:
        df_discretized (pd.DataFrame): processed dataset
        attributes (list): list of attributes
        target_col_name (str): name of the target column (class or value - unprocessed)
        type_outcome (str): type of the outcome variable
    """
    if dataset_name in ["german", "adult", "heart", "bank"]:
        filename = os.path.join(data_processed_dir, f"{dataset_name}.csv")
        print(f"Processing {dataset_name}")
        if dataset_name == "german":
            from utils_data import import_process_german

            df = import_process_german()
        elif dataset_name == "adult":
            from utils_data import import_process_adult

            df = import_process_adult()
        elif dataset_name == "heart":
            from utils_data import import_process_heart

            df = import_process_heart()
        elif dataset_name == "bank":
            from utils_data import import_process_bank

            df = import_process_bank()
        else:
            raise ValueError(f"Dataset {dataset_name} not available")
        attributes = list(df.columns)
        attributes.remove("class")
        target_col_name = "class"
        type_outcome = "boolean"
    elif dataset_name == "airbnb":
        print(f"Processing {dataset_name}")
        from utils_data import import_process_airbnb

        df_discretized, target_col_name = import_process_airbnb()
        import warnings

        warnings.warn("Airbnb dataset is returned discretized.")
        df = df_discretized
        type_outcome = "quantitative"
        attributes = list(df.columns)
        attributes.remove(target_col_name)

    elif dataset_name == "law":
        from utils_data import import_process_law

        df, target_col_name = import_process_law(use_rank_score=False, use_ZFYA=True)

        type_outcome = "quantitative"
        attributes = list(df.columns)
        attributes.remove(target_col_name)
    elif dataset_name in ["compas", "artificial"]:
        if dataset_name == "compas":
            from utils_data import import_process_compas

            df = import_process_compas()
        else:  # dataset_name == "artificial":
            from utils_data import import_artificial

            df = import_artificial(use_stored=True)

        attributes = list(df.columns)
        attributes.remove("class")
        attributes.remove("predicted")
        target_col_name = "class"
        type_outcome = "boolean"
    elif dataset_name == "bike":

        df, target_col_name = import_process_bike_sharing_data()

        # continuous_attributes = ["temp", "atemp", "hum", "windspeed"]

        attributes = list(df.columns)
        attributes.remove(target_col_name)
        type_outcome = "quantitative"
    elif dataset_name == "folktables":
        from utils import discretize

        year = "2023"
        filename = os.path.join("./data", f"folktables_{year}.csv")
        if os.path.exists(filename):
            df = pd.read_csv(os.path.join("./data", f"folktables_{year}.csv"))
            target_col_name = "income"
        else:

            df, target_col_name = import_process_folktables(
                input_dir="./data", store_data=True, year=year
            )
        df = df.loc[df["AGEP"] >= 18]
        attributes = list(df.columns)
        attributes.remove(target_col_name)
        type_outcome = "quantitative"
    else:
        raise ValueError("Dataset not found")
    return df, attributes, target_col_name, type_outcome


def encode_data(X_train, X_test, cat_attr_list, use_oh=False):

    oh_enc, scaler = None, None
    import pandas as pd
    from sklearn import preprocessing

    if cat_attr_list:
        other_cols = list(set(X_train.columns) - set(cat_attr_list))
        if use_oh:
            cat_encs = preprocessing.OneHotEncoder(sparse_output=False)

            encoded_data = cat_encs.fit_transform(X_train[cat_attr_list].astype(str))
            df_train_enc = pd.DataFrame(
                encoded_data, columns=cat_encs.get_feature_names_out(cat_attr_list)
            )

            encoded_data_test = cat_encs.transform(X_test[cat_attr_list].astype(str))
            df_test_enc = pd.DataFrame(
                encoded_data_test, columns=cat_encs.get_feature_names_out(cat_attr_list)
            )
        else:
            cat_encs = {}

            df_train_enc = X_train.copy()
            df_test_enc = X_test.copy()

            if "weekday" in X_train.columns and "weekday" in cat_attr_list:
                # Hard encoded
                weekday_mapping = {
                    v: i
                    for i, v in enumerate(
                        ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
                    )
                }
                df_train_enc["weekday"] = df_train_enc["weekday"].map(weekday_mapping)
                df_test_enc["weekday"] = df_test_enc["weekday"].map(weekday_mapping)
                cat_attr_list.remove("weekday")

            for col in cat_attr_list:
                le = preprocessing.LabelEncoder()
                df_train_enc[col] = le.fit_transform(df_train_enc[col].astype(str))
                # df_test_enc[col] = le.transform(df_test_enc[col].astype(str))

                # build mapping dict
                mapping = {cls: i for i, cls in enumerate(le.classes_)}

                # transform test with fallback (-1 for unseen)
                df_test_enc[col] = (
                    df_test_enc[col].astype(str).map(mapping).fillna(-1).astype(int)
                )

                cat_encs[col] = le

    else:
        other_cols = list(X_train.columns)
        df_train_enc = pd.DataFrame(index=X_train.index)
        df_test_enc = pd.DataFrame(index=X_test.index)

    # Add non-categorical columns
    df_train_enc[other_cols] = X_train[other_cols].values
    df_test_enc[other_cols] = X_test[other_cols].values

    if other_cols != []:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        df_train_enc[other_cols] = scaler.fit_transform(df_train_enc[other_cols])
        df_test_enc[other_cols] = scaler.transform(df_test_enc[other_cols])

    return df_train_enc, df_test_enc, cat_encs, scaler
