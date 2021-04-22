import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def normalize(x):
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled


def partial_normalize(df, no_norm_cols, to_norm_cols):
    no_norm_values = df[no_norm_cols].values
    to_norm_values = df[to_norm_cols].values
    norm_values = normalize(to_norm_values)
    values = np.concatenate([no_norm_values, norm_values], axis=1)
    cols = no_norm_cols + to_norm_cols
    return pd.DataFrame(data=values, columns=cols)


def normalize_df(df):
    column_names = df.columns
    x = df.values  # returns a numpy array
    # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # x_scaled = scaler.fit_transform(x)
    x_scaled = normalize(x)
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df


def preprocess(dir, to_dir):
    feat_file_list = ['wide_col.csv', 'debt_feat.csv', 'payment_feat.csv', 'payment_debt_cross_feat.csv',
                      'multi_acc_feat.csv', 'mal_behavior_feat.csv']
    for file_name in feat_file_list:
        print(f"=> preprocess:{file_name}")
        df = pd.read_csv(dir + file_name)
        df = df.fillna(-99)
        assert df.isnull().sum().sum() == 0

        norm_df = normalize_df(df)
        file_path = to_dir + "p_" + file_name
        norm_df.to_csv(file_path, index=False)
        print(f"save normalized data to: {file_path}")

    # 'loan_feat.csv', 'qualify_feat.csv',

    qualify_feat = [
        'grade',
        'emp_length',
        'home_ownership',
        'verification_status',
        'annual_inc_comp',
        'purpose',
        'application_type',
        'disbursement_method'
    ]
    file_name = 'qualify_feat.csv'
    df = pd.read_csv(dir + file_name)
    df = df[qualify_feat]
    assert df.isnull().sum().sum() == 0

    # no_norm_cols = ["grade", "emp_length", "home_ownership", "verification_status"]
    # to_norm_cols = ["annual_inc_comp"]
    no_norm_cols = ["home_ownership", "purpose"]
    to_norm_cols = ["grade", "emp_length", "verification_status",
                    'annual_inc_comp', 'application_type', 'disbursement_method']
    norm_df = partial_normalize(df, no_norm_cols, to_norm_cols)
    file_path = to_dir + "p_" + file_name
    norm_df.to_csv(file_path, index=False)
    print(f"save normalized data to: {file_path}")

    # loan_feat = [
    #     'term',
    #     'initial_list_status',
    #     'purpose',
    #     'application_type',
    #     'disbursement_method'
    # ]
    # file_name = 'loan_feat.csv'
    # df = pd.read_csv(dir + file_name)
    # df = df[loan_feat]
    # assert df.isnull().sum().sum() == 0
    # file_path = to_dir + "p_" + file_name
    # df.to_csv(file_path, index=False)
    # print(f"save data to: {file_path}")


if __name__ == "__main__":
    # print("=> preprocess 2018 data")
    # dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"
    # to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"
    # preprocess(dir, to_dir)
    #
    # print("=> preprocess 2015-2017 data")
    # dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_17/"
    # to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_17/"
    # preprocess(dir, to_dir)

    print("=> preprocess 2015-2016 data")
    dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_16/"
    to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_16/"
    preprocess(dir, to_dir)

    print("=> preprocess 2016-2017 data")
    dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2016_17/"
    to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2016_17/"
    preprocess(dir, to_dir)
