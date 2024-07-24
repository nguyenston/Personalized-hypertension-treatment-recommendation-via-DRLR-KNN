import sys
sys.path.append('/home/yeping/HTNProject/Personalized-hypertension-treatment-recommendation-via-DRLR-KNN')
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from collections import Counter
from util import get_base_path


# NOTE: This function is irrelevant to the project as we only deal with hypertension
def load_diabetes_final_table_for_prescription(trial_id, test_ratio=0.2):
    """
    load preprocess diabetes data
    :param trial_id: trial id
    :param test_ratio: ratio of test data
    :return: train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u
    """
    df = pd.read_csv('/home/HTNclinical/Select-Optimal-Decisions-via-DRO-KNN-master/training-data/HTN_RegistryOutput.csv',nrows=10e4,on_bad_lines='skip')
    prescription_columns = ['prescription_oral', 'prescription_injectable']
    hist_pres_columns = ['hist_prescription_oral', 'hist_prescription_injectable']
    useful_feature = [item for item in df.columns.tolist() if
                      item not in prescription_columns and item != 'future_a1c']

    X = np.array(df[useful_feature].values, dtype=np.float32)
    y = np.array(df['future_a1c'].values, dtype=np.float32)
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

    return train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u


# NOTE: Function of interests
def load_hypertension_final_table_for_prescription(trial_id, test_ratio=0.2):
    """
    load preprocess hypertension data
    :param trial_id: trial id
    :param test_ratio: ratio of test data
    :return: train_all_x, train_all_y, train_all_z, train_all_u, test_x, test_y, test_z, test_u
    """
    df = pd.read_csv("/home/HTNclinical/Select-Optimal-Decisions-via-DRO-KNN-master/training-data/HTN_RegistryOutput.csv",nrows=1000,on_bad_lines="skip")
    not_use_columns = ["PatientEpicKey","PatientDurableKey","LastServDate_UrineCult",
                       "LastServDate_OfficeVisit","PatientEndDate","LookBackDate"]
    prescription_columns = [
        "Thiazide90_Ind", "CalciumChannelBlock90_Ind", "ARB90_Ind","ACEI90_Ind", "BetaBlock90_Ind",
        "LoopDiuretic90_Ind", "MineralcorticoidRecAnt90_Ind"
    ]
    hist_pres_columns = ["Thiazide_Ind", "CalciumChannelBlock_Ind", "ARB_Ind","ACEI_Ind", "BetaBlock_Ind",
        "LoopDiuretic_Ind", "Leuprolide22_Ind", "Cyclopentolate1Pct_Ind", "Augmentin875_Ind", "MineralcorticoidRecAnt_Ind"
    ]
    useful_feature = [item for item in df.columns.tolist()
                      if item not in not_use_columns and item not in prescription_columns and item not in hist_pres_columns]
    X = pd.get_dummies(df[useful_feature], columns = ["LegalSex","GeneralRace","SmokingStatus"]).to_numpy()
    y = df["SBP90_Avg"].to_numpy()
    z = df[prescription_columns].to_numpy()
    u = df[hist_pres_columns].to_numpy()

    # Encode data from multiple columns as a single bitstring, as the datatype is flag 0/1
    # In this case, a combination of medications is grouped up as a "prescription"
    z_c = (
        z[:, 0] + 2 * z[:, 1] + 4 * z[:, 2] + 8 * z[:, 3] + 16 * z[:, 4] + 32 * z[:, 5]
    )
    z_c = np.asanyarray(z_c, dtype=int)

    u_c = (
        u[:, 0] + 2 * u[:, 1] + 4 * u[:, 2] + 8 * u[:, 3] + 16 * u[:, 4] + 32 * u[:, 5]
    )
    u_c = np.asanyarray(u_c, dtype=int)

    # assign id to prescriptions, prescriptions not in the most 19 commonly used ones are assigned the same id
    # in other words, we are grouping up rare prescriptions into one single classification
    commom_19 = [item[0] for item in Counter(z_c).most_common(19)]
    new_id = {item: item_id for item_id, item in enumerate(commom_19)}
    for i in range(64):
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
