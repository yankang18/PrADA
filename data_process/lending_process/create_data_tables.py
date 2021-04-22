import pandas as pd
import numpy as np

from data_process.lending_process.feat_group_dicts import wide_col_feat, loan_feat, qualify_feat, debt_feat, payment_feat, \
    multi_acc_feat, mal_behavior_feat


def save_table(file_path, df_table):
    df_table.to_csv(file_path, index=False)
    print(f" ==> save table with shape:{df_table.shape} to :{file_path}")


def create_target_table(to_dir, df_loan):
    print("=> create_target_table")
    df = df_loan[['target']]
    file_path = to_dir + "target.csv"
    save_table(file_path, df)


def create_wide_col_table(to_dir, df_loan):
    print("=> create_wide_col_table")
    df = df_loan[wide_col_feat]
    file_path = to_dir + "wide_col.csv"
    save_table(file_path, df)


def create_loan_feat_table(to_dir, df_loan):
    print("=> create_loan_feat_table")
    df = df_loan[loan_feat]
    file_path = to_dir + "loan_feat.csv"
    save_table(file_path, df)


# 资质表现
def create_qualify_table(to_dir, df_loan):
    print("=> create_qualify_table")
    df = df_loan[qualify_feat]
    file_path = to_dir + "qualify_feat.csv"
    save_table(file_path, df)


# 债务表现
def create_debt_table(to_dir, df_loan):
    print("=> create_debt_table")
    df = df_loan[debt_feat]
    file_path = to_dir + "debt_feat.csv"
    save_table(file_path, df)


def create_payment_table(to_dir, df_loan):
    print("=> create_payment_table")
    df = df_loan[payment_feat]
    file_path = to_dir + "payment_feat.csv"
    save_table(file_path, df)


def create_payment_debt_cross_table(to_dir, df_loan):
    print("=> create_payment_debt_cross_table")
    cross_feat = debt_feat + payment_feat
    df = df_loan[cross_feat]
    file_path = to_dir + "payment_debt_cross_feat.csv"
    save_table(file_path, df)


# 多投表现
def create_multi_account_table(to_dir, df_loan):
    print("=> create_multi_account_table")
    df = df_loan[multi_acc_feat]
    file_path = to_dir + "multi_acc_feat.csv"
    save_table(file_path, df)


def create_mal_behavior_table(to_dir, df_loan):
    print("=> create_mal_behavior_table")
    df = df_loan[mal_behavior_feat]
    file_path = to_dir + "mal_behavior_feat.csv"
    save_table(file_path, df)


def sample_data(data, pos_num_samples, neg_num_samples):
    pos_df = data[data["target"] == 1]
    neg_df = data[data["target"] == 0]

    print(f"data shape:{data.shape} with pos:{pos_df.shape} and neg:{neg_df.shape}")
    print(f"select {pos_num_samples} pos samples from pos:{pos_df.shape}")
    print(f"select {neg_num_samples} neg samples from neg:{neg_df.shape}")

    columns = data.columns
    # data = data.values
    pos_data = pos_df.values
    neg_data = neg_df.values

    pos_idxs = [i for i in range(len(pos_data))]
    neg_idxs = [i for i in range(len(neg_data))]

    sampled_pos_idxs = np.random.choice(pos_idxs, pos_num_samples, replace=False)
    sampled_neg_idxs = np.random.choice(neg_idxs, neg_num_samples, replace=False)

    sampled_pos_data = pos_data[sampled_pos_idxs]
    sampled_neg_data = neg_data[sampled_neg_idxs]

    sampled_data = np.concatenate([sampled_pos_data, sampled_neg_data], axis=0)
    np.random.shuffle(sampled_data)

    sampled_df = pd.DataFrame(sampled_data, columns=columns)

    pos_df = sampled_df[sampled_df["target"] == 1]
    neg_df = sampled_df[sampled_df["target"] == 0]
    print(f"data shape:{data.shape} with pos:{pos_df.shape} and neg:{neg_df.shape}")

    return sampled_df


def create_tables(to_dir, df_loan):
    create_target_table(to_dir, df_loan)
    create_wide_col_table(to_dir, df_loan)
    create_loan_feat_table(to_dir, df_loan)
    create_qualify_table(to_dir, df_loan)
    create_debt_table(to_dir, df_loan)
    create_payment_table(to_dir, df_loan)
    create_multi_account_table(to_dir, df_loan)
    create_mal_behavior_table(to_dir, df_loan)
    create_payment_debt_cross_table(to_dir, df_loan)


if __name__ == "__main__":
    # dir = "../../../data/lending_club_bundle_archive/"
    # to_dir = "../../../data/lending_club_bundle_archive/loan_processed_2015_17/"

    dir = "../../../data/lending_club_bundle_archive/loan_data_v2/"
    # to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_17/"
    # file_path = dir + "loan_processed_2015_17.csv"
    # df_loan = pd.read_csv(file_path, low_memory=False)
    # create_tables(to_dir, df_loan)
    #
    # file_path = dir + "loan_processed_2018.csv"
    # to_dir = "../../../data/lending_club_bundle_archive/loan_processed_2018/"
    file_path = dir + "loan_processed_2018.csv"
    to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"
    df_loan = pd.read_csv(file_path, low_memory=False)

    sample_data(df_loan, pos_num_samples=10, neg_num_samples=1)
    create_tables(to_dir, df_loan)

    # dir = "../../../data/lending_club_bundle_archive/loan_data_v2/"
    # to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_16/"
    # file_path = dir + "loan_processed_2015_16.csv"
    # df_loan = pd.read_csv(file_path, low_memory=False)
    # create_tables(to_dir, df_loan)
    #
    # # file_path = dir + "loan_processed_2018.csv"
    # # to_dir = "../../../data/lending_club_bundle_archive/loan_processed_2018/"
    # to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2016_17/"
    # file_path = dir + "loan_processed_2016_17.csv"
    # df_loan = pd.read_csv(file_path, low_memory=False)
    # create_tables(to_dir, df_loan)
