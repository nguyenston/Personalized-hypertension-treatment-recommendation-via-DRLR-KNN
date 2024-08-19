import sys

sys.path.append(
    "/home/yeping/HTNProject/Personalized-hypertension-treatment-recommendation-via-DRLR-KNN"
)
from joblib import numpy_pickle_compat
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from collections import Counter
from util import get_base_path
import pickle
import os.path


# NOTE: This function is irrelevant to the project as we only deal with hypertension

def load_diabetes_final_table_for_prescription(trial_id, test_ratio=0.2):
    """
    load preprocess diabetes data
    :param trial_id: trial id
    :param test_ratio: ratio of test data
    :return: train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u
    """
    df = pd.read_csv(
        "/home/HTNclinical/Select-Optimal-Decisions-via-DRO-KNN-master/training-data/HTN_RegistryOutput.csv",
        nrows=int(10e4),
        on_bad_lines="skip",
    )
    prescription_columns = ["prescription_oral", "prescription_injectable"]
    hist_pres_columns = ["hist_prescription_oral", "hist_prescription_injectable"]
    useful_feature = [
        item
        for item in df.columns.tolist()
        if item not in prescription_columns and item != "future_a1c"
    ]

    X = np.array(df[useful_feature].values, dtype=np.float32)
    y = np.array(df["future_a1c"].values, dtype=np.float32)
    z = np.array(df[prescription_columns].values, dtype=int)
    u = np.array(df[hist_pres_columns].values, dtype=int)

    z = np.array(z[:, 0] + 2 * z[:, 1], dtype=int)
    u = np.array(u[:, 0] + 2 * u[:, 1], dtype=int)

    train_all_x = []
    train_all_y = []
    train_all_z = []
    train_all_u = []

    test_x = []
    test_y = []
    test_z = []
    test_u = []

    for pres_id in range(4):
        valid_id = z == pres_id

        this_X = X[valid_id]
        this_y = y[valid_id]
        this_z = z[valid_id]
        this_u = u[valid_id]

        rs = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=trial_id)
        train_index, test_index = rs.split(this_X).__next__()
        X_train_all, X_test = this_X[train_index], this_X[test_index]
        y_train_all, y_test = this_y[train_index], this_y[test_index]
        z_train_all, z_test = this_z[train_index], this_z[test_index]
        u_train_all, u_test = this_u[train_index], this_u[test_index]

        train_all_x.append(X_train_all)
        train_all_y.append(y_train_all)
        train_all_z.append(z_train_all)
        train_all_u.append(u_train_all)

        test_x.append(X_test)
        test_y.append(y_test)
        test_z.append(z_test)
        test_u.append(u_test)

    return (
        train_all_x,
        train_all_y,
        train_all_z,
        train_all_u,
        test_x,
        test_y,
        test_z,
        test_u,
    )


# NOTE: Function of interests
def load_hypertension_final_table_for_prescription(trial_id, test_ratio=0.2):
    """
    load preprocess hypertension data
    :param trial_id: trial id
    :param test_ratio: ratio of test data
    :return: train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u
    """
    file_stem = "/home/HTNclinical/Select-Optimal-Decisions-via-DRO-KNN-master/training-data/FullFiles/Processed/HTN_RegistryOutput_2024-07-03"
    nrows = 2e5
    pickle_file_name = file_stem + "-nrows=" + str(nrows) + ".pkl"
    not_use_columns = [
            "PatientEpicKey",
            "PatientDurableKey",
            "LastServDate_UrineCult",
            "LastServDate_OfficeVisit",
            "PatientEndDate",
            "LookBackDate",
            "SBPFuture_Avg",
        ]
    prescription_columns = [
            "Thiazide90_Ind",
            "CalciumChannelBlock90_Ind",
            "ARB90_Ind",
            "ACEI90_Ind",
            "BetaBlock90_Ind",
            "LoopDiuretic90_Ind",
            "MineralcorticoidRecAnt90_Ind",
        ]
    hist_pres_columns = [
            "Thiazide_Ind",
            "CalciumChannelBlock_Ind",
            "ARB_Ind",
            "ACEI_Ind",
            "BetaBlock_Ind",
            "LoopDiuretic_Ind",
            "Leuprolide22_Ind",
            "Cyclopentolate1Pct_Ind",
            "Augmentin875_Ind",
            "MineralcorticoidRecAnt_Ind",
        ]
    if not os.path.isfile(pickle_file_name):
        df = pd.read_csv(
            file_stem + ".csv",
            nrows=nrows,
            on_bad_lines="skip",
            dtype=str,
        )
        

        # Apply one-hot encoding
        df = pd.get_dummies(
            df,
            columns=["LegalSex", "GeneralRace", "SmokingStatus"],
            dtype=int,
        )
        # Impute by two logics for numerical/categorical values with the same PatientEpicKey.
        # Commented out entries are deprecated and is no longer used
        numerical_columns = [
            "RedCellDist_Avg",
            "MeanCorpHemo_Avg",
            "PartialPressureCO2_Avg",
            "MeanCorpHemoConcentrate_Avg",
            "AnionGapNoPotassium_Avg",
            "NeutrophilCnt_Avg",
            "LymphocytesCnt_Avg",
            "MonocytesPct",
            "Monocytes_Avg",
            "Basophils_Avg",
            "BodySurfaceArea",
            "Eosinophils_Avg",
            "NucleatedRBC_Avg",
            "AbsNeutrophilPCT_Avg",
            "UrinePH_Avg",
            "UrineSpecificGravity_Avg",
            "Glucose_Avg",
            "Leukocyte_Avg",
            "GlucoseWBGV_Avg",
            "EosinophilsPCT_Avg",
            "MeanCorpusHemoRBC_Avg",
            "Chloride_Avg",
            "Cholesterol_Avg",
            "Sodium_Avg",
            "SerumCreatinine_Avg",
            "Albumin_Avg",
            "BasophilsPCT_Avg",
            "HDL_Avg",
            "HemoA1C_Avg",
            "AlkalinePhosphate_Avg",
            "LDLDirect_Avg",
            "MeanCorpuscVol_Avg",
            "LymphocytesPCT_Avg",
            "AspartateAminoTran_Avg",
            "AlanineAminoTran_Avg",
            "Bilirubin_Avg",
            "Triglycerides_Avg",
            "Hematocrit_Avg",
            "Hemoglobin_Avg",
            "Platelet_Avg",
            "BloodUreaNitrogen_Avg",
            "SBP90_Avg",
            "SBP90180_Avg",
            "SBP180270_Avg",
            "DBP90_Avg",
            "DBP90180_Avg",
            "DBP180270_Avg",
            "Temp90_Avg",
            # "Temp90180_Avg",
            # "Temp180270_Avg",
            "SPO2_90_Avg",
            # "SPO2_90180_Avg",
            # "SPO2_180270_Avg",
            "Respiration90_Avg",
            # "Respiration90180_Avg",
            # "Respiration180270_Avg",
            "Pulse90_Avg",
            # "Pulse90180_Avg",
            # "Pulse180270_Avg",
            "BMI90_Avg",
            # "BMI90180_Avg",
            # "BMI180270_Avg",
        ]
        boolean_columns = [
            item
            for item in df.columns.tolist()
            if item not in numerical_columns and item not in not_use_columns
        ]
        df = df.apply(pd.to_numeric, errors="coerce")

        # Function to impute missing values by median within each group
        def impute_numerical_by_median(table):
            med = table.median()
            values = {c: med[c] for c in numerical_columns}
            return table.fillna(value=values)

        # Apply the functions to each group
        df = (
            df.groupby("PatientEpicKey")
            .apply(impute_numerical_by_median, include_groups=False)
            .reset_index(level=0)
        )
        df[numerical_columns] = df[numerical_columns].fillna(
            df[numerical_columns].median().to_dict()
        )

        df[boolean_columns] = df.groupby("PatientEpicKey")[boolean_columns].ffill()
        df[boolean_columns] = df[boolean_columns].fillna(value=0)
        pickle.dump(
            df,
            open(pickle_file_name, "wb"),
        )
        print("Finish Imputation")
    else:
        df = pd.read_pickle(pickle_file_name)

    useful_feature = [
        item
        for item in df.columns.tolist()
        if item not in not_use_columns
        and item not in prescription_columns
        and item not in hist_pres_columns
    ]
    for col in useful_feature:
        if bool(df[col].isnull().any()):
            print(col)
    X = df[useful_feature].to_numpy()
    y = df["SBP90_Avg"].to_numpy()
    z = df[prescription_columns].to_numpy()
    u = df[hist_pres_columns].to_numpy()
    print(X.shape)

    z_c = (
        z[:, 0]
        + 2 * z[:, 1]
        + 4 * z[:, 2]
        + 8 * z[:, 3]
        + 16 * z[:, 4]
        + 32 * z[:, 5]
        + 64 * z[:, 6]
    )
    z_c = np.asanyarray(z_c, dtype=int)

    u_c = (
        u[:, 0]
        + 2 * u[:, 1]
        + 4 * u[:, 2]
        + 8 * u[:, 3]
        + 16 * u[:, 4]
        + 32 * u[:, 5]
        + 64 * u[:, 6]
        + 128 * u[:, 7]
        + 256 * u[:, 8]
        + 512 * u[:, 9]
    )
    u_c = np.asanyarray(u_c, dtype=int)

    commom_19 = [item[0] for item in Counter(z_c).most_common(19)]
    print(commom_19)
    new_id = {item: item_id for item_id, item in enumerate(commom_19)}
    for i in range(1024):
        if i not in new_id.keys():
            new_id[i] = 19

    z = np.array([new_id[item] for item in z_c], dtype=int)
    u = np.array([new_id[item] for item in u_c], dtype=int)
    train_all_x = []
    train_all_y = []
    train_all_z = []
    train_all_u = []

    test_x = []
    test_y = []
    test_z = []
    test_u = []
    print("Finish counters")
    for pres_id in range(20):
        valid_id = z == pres_id
        this_X = X[valid_id]
        this_y = y[valid_id]
        this_z = z[valid_id]
        this_u = u[valid_id]

        rs = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=trial_id)
        train_index, test_index = rs.split(this_X).__next__()

        X_train_all, X_test = this_X[train_index], this_X[test_index]
        y_train_all, y_test = this_y[train_index], this_y[test_index]
        z_train_all, z_test = this_z[train_index], this_z[test_index]
        u_train_all, u_test = this_u[train_index], this_u[test_index]

        train_all_x.append(X_train_all)
        train_all_y.append(y_train_all)
        train_all_z.append(z_train_all)
        train_all_u.append(u_train_all)

        test_x.append(X_test)
        test_y.append(y_test)
        test_z.append(z_test)
        test_u.append(u_test)
    print("Finish Preprocess")

    return (
        train_all_x,
        train_all_y,
        train_all_z,
        train_all_u,
        test_x,
        test_y,
        test_z,
        test_u,
    )
