import numpy as np
import pandas as pd

from data_process.cell_process.split_train_val_test_data import split_train_val_test


def split_201517_data(table_names):
    print("[INFO] ==> split train, val, test of 201517 data")

    num_total_samples = 1299081
    perm_idxs = np.random.permutation(num_total_samples)

    num_train = 1100000
    num_val = None

    # dir = "../../../data/lending_club_bundle_archive/loan_processed_2015_18/"
    # to_dir = "../../../data/lending_club_bundle_archive/loan_processed_2015_18/"
    dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_17/"
    to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_17/"
    split_train_val_test(dir, to_dir, table_names, num_train, num_val,
                         create_val_data=False, perm_indices=perm_idxs, up_sample_train_idxs=None)


def split_201617_data(table_names):
    print("[INFO] ==> split train, val, test of 201617 data")

    # num_total_samples = 877986
    # num_total_samples = 220000
    num_total_samples = 100000
    perm_idxs = np.random.permutation(num_total_samples)

    num_train = 80000
    num_val = None

    dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2016_17/"
    # to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2016_17/"
    to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2016_17_small/"
    split_train_val_test(dir, to_dir, table_names, num_train, num_val,
                         create_val_data=False, perm_indices=perm_idxs, up_sample_train_idxs=None)


def split_201516_data(table_names):
    print("[INFO] ==> split train, val, test of 201516 data")

    num_total_samples = 855502
    perm_idxs = np.random.permutation(num_total_samples)

    num_train = 755502
    num_val = None

    dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_16/"
    to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_16/"
    split_train_val_test(dir, to_dir, table_names, num_train, num_val,
                         create_val_data=False, perm_indices=perm_idxs, up_sample_train_idxs=None)


def split_2018_data(table_names):
    print("[INFO] ==> split train, val, test of 2018 data")

    # num_total_samples = 495242
    # perm_indices = np.random.permutation(num_total_samples)
    perm_indices = None
    # num_train = 29636
    # num_train = 120000
    # num_train = 4000
    # perm_idxs = None
    # num_val = 10000
    # up_sample_idxs = np.random.choice(num_train, 777986, replace=True)
    up_sample_idxs = None
    num_train = None

    dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"
    data_2018 = pd.read_csv(dir + 'target.csv', skipinitialspace=True)
    pos_indices = data_2018.index[data_2018["target"] == 1].values
    neg_indices = data_2018.index[data_2018["target"] == 0].values

    # prepare train data
    num_train_pos_samples = 3950
    num_train_neg_samples = 50
    print(f"number of pos indices :{len(pos_indices)}")
    print(f"number of neg indices :{len(neg_indices)}")
    sampled_train_pos_idxs = np.random.choice(pos_indices, num_train_pos_samples, replace=False)
    sampled_train_neg_idxs = np.random.choice(neg_indices, num_train_neg_samples, replace=False)
    sampled_train_idxs = np.concatenate([sampled_train_pos_idxs, sampled_train_neg_idxs], axis=0)
    np.random.shuffle(sampled_train_idxs)
    print(f"sampled_train_idxs:{sampled_train_idxs.shape}")

    # prepare non-train data
    all_idxs = np.arange(len(data_2018))
    non_train_idxs = np.setdiff1d(all_idxs, sampled_train_idxs)
    print(f"non_train_idxs:{non_train_idxs.shape}")
    num_sampled_non_train_idxs = 100000
    non_train_idxs = np.random.choice(non_train_idxs, num_sampled_non_train_idxs, replace=False)
    print(f"sampled_non_train_idxs:{non_train_idxs.shape}")

    # dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"
    to_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"
    split_train_val_test(dir, to_dir, table_names, num_train,
                         num_val=None,
                         create_val_data=True,
                         perm_indices=perm_indices,
                         train_indices=sampled_train_idxs,
                         non_train_indices=non_train_idxs,
                         up_sample_train_idxs=up_sample_idxs)


if __name__ == "__main__":
    # table_names = ["p_wide_col.csv", "p_debt_feat.csv", "p_payment_feat.csv",
    #                "p_payment_debt_cross_feat.csv", "p_multi_acc_feat.csv", "p_mal_behavior_feat.csv",
    #                "p_qualify_feat.csv", "p_loan_feat.csv", "target.csv"]
    table_names = ["p_wide_col.csv", "p_debt_feat.csv", "p_payment_feat.csv",
                   "p_payment_debt_cross_feat.csv", "p_multi_acc_feat.csv", "p_mal_behavior_feat.csv",
                   "p_qualify_feat.csv", "target.csv"]
    # split_201517_data(table_names)
    split_201617_data(table_names)
    # split_201516_data(table_names)
    # split_2018_data(table_names)
    # TODO: validate splitted data between 201517 and 2018
    print("finished splitting data!")
