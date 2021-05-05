import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def create_train_and_test(df_data, df_datetime, num_train, to_dir):
    df_2014 = df_data[df_datetime['ListingInfo_Year'] == 2014]
    df_datetime_2014 = df_datetime[df_datetime['ListingInfo_Year'] == 2014]
    df_2014, df_datetime_2014 = shuffle(df_2014, df_datetime_2014)

    df_train = df_2014[:num_train]
    df_datetime_train = df_datetime_2014[:num_train]

    df_test = df_2014[num_train:]
    df_datetime_test = df_datetime_2014[num_train:]

    print(f"df_train with shape: {df_train.shape}")
    print(f"df_test with shape: {df_test.shape}")

    title = "PPD"
    tag = '2014'
    df_train.to_csv("{}/{}_data_{}_train.csv".format(to_dir, title, tag), index=False)
    df_test.to_csv("{}/{}_data_{}_test.csv".format(to_dir, title, tag), index=False)
    df_datetime_train.to_csv("{}/{}_datetime_{}_train.csv".format(to_dir, title, tag), index=False)
    df_datetime_test.to_csv("{}/{}_datetime_{}_test.csv".format(to_dir, title, tag), index=False)


def normalize_df(df):
    column_names = df.columns
    x = df.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df


def standandize(df_data, df_columns_list, df_cat_mask_list):
    df_list = list()
    df_target = df_data[['target']]
    for columns, is_cat in zip(df_columns_list, df_cat_mask_list):
        df_subset = df_data[columns].copy()
        if is_cat is False:
            df_subset = normalize_df(df_subset)
        df_list.append(df_subset)
    df_list.append(df_target)
    return pd.concat(df_list, axis=1)


def select_positive(data, select_pos_ratio=0.5):
    target_col = data[:, -1]
    data_1 = data[target_col == 1]
    data_0 = data[target_col == 0]

    print(f"before select positive, data_1: {data_1.shape}")
    print(f"before select positive, data_0: {data_0.shape}")

    select_num_positive = int(data_1.shape[0] * select_pos_ratio)
    data_1 = data_1[:select_num_positive]

    print(f"after select positive, data_1: {data_1.shape}")
    print(f"after select positive, data_0: {data_0.shape}")
    return np.concatenate((data_1, data_0), axis=0)


def create_source_and_target(df_2014, df_datetime_2014,
                             df_column_split_list, df_cat_mask_list, to_dir, train=True):
    print("--- create_degree_source_target_data for {} data --- ".format("train" if train else "test"))

    df_2014_1to8 = df_2014[df_datetime_2014['ListingInfo_Month'] <= 8]
    df_2014_1to9 = df_2014[df_datetime_2014['ListingInfo_Month'] <= 9]
    df_2014_10to12 = df_2014[df_datetime_2014['ListingInfo_Month'] >= 10]

    df_2014_9 = df_2014[df_datetime_2014['ListingInfo_Month'] == 9]
    df_2014_10 = df_2014[df_datetime_2014['ListingInfo_Month'] == 10]
    df_2014_11 = df_2014[df_datetime_2014['ListingInfo_Month'] == 11]

    all_col_list = list(df_2014.columns)

    print("df_2014_1to8:", df_2014_1to8.shape)
    print("df_2014_1to9:", df_2014_1to9.shape)
    print("df_2014_10to12:", df_2014_10to12.shape)
    print("df_2014_9:", df_2014_9.shape)
    print("df_2014_10:", df_2014_10.shape)
    print("df_2014_11:", df_2014_11.shape)

    # define source
    data_2014_src_1to8 = df_2014_1to8.values
    data_2014_src_1to9 = df_2014_1to9.values
    print("data_2014_src_1to8", data_2014_src_1to8.shape)
    print("data_2014_src_1to9", data_2014_src_1to9.shape)

    # define target
    df_datetime_2014_10 = df_datetime_2014[df_datetime_2014['ListingInfo_Month'] == 10]
    # df_2014_10_day15 = df_2014_10[df_datetime_2014_10['ListingInfo_DayofMonth'] >= 15]
    # data_2014_tgt_10_last_day15 = df_2014_10_day15.values
    data_2014_tgt_10 = df_2014_10.values
    data_2014_tgt_11 = df_2014_11.values

    # data_2014_tgt_10to11 = np.concatenate((data_2014_tgt_10_last_day15, data_2014_tgt_11), axis=0)
    data_2014_tgt_10to11 = np.concatenate((data_2014_tgt_10, data_2014_tgt_11), axis=0)
    data_2014_tgt_9 = df_2014_9.values

    print("data_2014_tgt_10", data_2014_tgt_10.shape)
    print("data_2014_tgt_11", data_2014_tgt_11.shape)
    print("[INFO] before select pos samples, data_2014_tgt_10to11:", data_2014_tgt_10to11.shape)
    print("[INFO] before select pos samples, data_2014_tgt_9:", data_2014_tgt_9.shape)

    # select positive samples
    if train:
        print("[INFO] apply positive sample selection")
        # data_2014_src_1to8 = select_positive(data_2014_src_1to8, select_pos_ratio=0.5)
        # data_2014_src_1to9 = select_positive(data_2014_src_1to9, select_pos_ratio=0.5)

        data_2014_tgt_10to11 = select_positive(data_2014_tgt_10to11, select_pos_ratio=0.1)
        data_2014_tgt_9 = select_positive(data_2014_tgt_9, select_pos_ratio=0.1)

    print("[INFO] after select pos samples, data_2014_tgt_10to11:", data_2014_tgt_10to11.shape)
    print("[INFO] after select pos samples, data_2014_tgt_9:", data_2014_tgt_9.shape)

    # shuffle
    data_2014_src_1to8 = shuffle(data_2014_src_1to8)
    data_2014_src_1to9 = shuffle(data_2014_src_1to9)
    if train:
        data_2014_tgt_9 = shuffle(data_2014_tgt_9)[:3000]
        data_2014_tgt_10to11 = shuffle(data_2014_tgt_10to11)[:3000]

    df_2014_src_1to8 = pd.DataFrame(data=data_2014_src_1to8, columns=all_col_list)
    df_2014_src_1to9 = pd.DataFrame(data=data_2014_src_1to9, columns=all_col_list)

    df_2014_tgt_9 = pd.DataFrame(data=data_2014_tgt_9, columns=all_col_list)
    df_2014_tgt_10to11 = pd.DataFrame(data=data_2014_tgt_10to11, columns=all_col_list)

    # standardize
    df_2014_src_1to8 = standandize(df_2014_src_1to8, df_column_split_list, df_cat_mask_list)
    df_2014_src_1to9 = standandize(df_2014_src_1to9, df_column_split_list, df_cat_mask_list)
    df_2014_tgt_9 = standandize(df_2014_tgt_9, df_column_split_list, df_cat_mask_list)
    df_2014_tgt_10to11 = standandize(df_2014_tgt_10to11, df_column_split_list, df_cat_mask_list)

    print("[INFO] (final) df_2014_src_1to8: ", df_2014_src_1to8.shape)
    print("[INFO] (final) df_2014_src_1to9: ", df_2014_src_1to9.shape)
    print("[INFO] (final) df_2014_tgt_9: ", df_2014_tgt_9.shape)
    print("[INFO] (final) df_2014_tgt_10to11: ", df_2014_tgt_10to11.shape)

    # save
    mode = "train" if train else "test"
    df_2014_src_1to8.to_csv("{}/PPD_2014_src_1to8_{}.csv".format(to_dir, mode), index=False)
    df_2014_src_1to9.to_csv("{}/PPD_2014_src_1to9_{}.csv".format(to_dir, mode), index=False)

    df_2014_tgt_9.to_csv("{}/PPD_2014_tgt_9_{}.csv".format(to_dir, mode), index=False)
    df_2014_tgt_10to11.to_csv("{}/PPD_2014_tgt_10to11_{}.csv".format(to_dir, mode), index=False)


if __name__ == "__main__":
    # df_cat_mask_list = [False, True, False, False, True, False, True, False, True]

    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output/"

    timestamp = '1620085151'
    data_all = data_dir + 'PPD_data_all_{}.csv'.format(timestamp)
    data_datetime = data_dir + 'PPD_data_datetime_{}.csv'.format(timestamp)
    meta_data = data_dir + 'PPD_meta_data_{}.json'.format(timestamp)

    df_data_all = pd.read_csv(data_all, skipinitialspace=True)
    df_data_datetime = pd.read_csv(data_datetime, skipinitialspace=True)

    print(f"df_data_all: {df_data_all.shape}")
    print(f"df_data_datetime: {df_data_datetime.shape}")

    # print("df_data_all:")
    # print(df_data_all.head(5))

    with open(meta_data) as json_file:
        print(f"[INFO] load task meta file from {meta_data}")
        meta_data_dict = json.load(json_file)

    print("meta_data_dict", meta_data_dict)

    df_cat_mask_list = meta_data_dict['df_cat_mask_list']
    df_column_split_list = meta_data_dict['df_column_split_list']
    # df_all_column_list = meta_data_dict['df_all_column_list'] + ['target']

    to_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{timestamp}/"

    # num_train = 60000
    # create_train_and_test(df_data_all, df_data_datetime, num_train, to_dir)

    df_2014 = pd.read_csv(to_dir + "PPD_data_2014_train.csv", skipinitialspace=True)
    df_datetime_2014 = pd.read_csv(to_dir + "PPD_datetime_2014_train.csv", skipinitialspace=True)
    # print("df_2014:")
    # print(df_2014.head())
    create_source_and_target(df_2014, df_datetime_2014,
                             df_column_split_list, df_cat_mask_list, to_dir, train=True)
    create_source_and_target(df_2014, df_datetime_2014,
                             df_column_split_list, df_cat_mask_list, to_dir, train=False)
