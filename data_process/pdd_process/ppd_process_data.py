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
    df_train.to_csv("{}/{}_data_{}_{}_train.csv".format(to_dir, title, tag, str(num_train)), index=False)
    df_test.to_csv("{}/{}_data_{}_{}_test.csv".format(to_dir, title, tag, str(num_train)), index=False)
    df_datetime_train.to_csv("{}/{}_datetime_{}_{}_train.csv".format(to_dir, title, tag, str(num_train)), index=False)
    df_datetime_test.to_csv("{}/{}_datetime_{}_{}_test.csv".format(to_dir, title, tag, str(num_train)), index=False)


def normalize_df(df):
    column_names = df.columns
    x = df.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df


def standardize(df_data, df_columns_list, df_cat_mask_list):
    df_list = list()
    df_target = df_data[['target']]
    for columns, is_cat in zip(df_columns_list, df_cat_mask_list):
        df_subset = df_data[columns].copy()
        if is_cat is False:
            df_subset = normalize_df(df_subset)
        df_list.append(df_subset)
    df_list.append(df_target)
    return pd.concat(df_list, axis=1)


def select_positive(tgt_month, data, num_target, select_pos_ratio=0.5):
    target_col = data[:, -1]
    data_1 = data[target_col == 1]
    data_0 = data[target_col == 0]

    print(f"=> select pos samples for target of {tgt_month}.")
    print(f"| before select positive, data_1: {data_1.shape}")
    print(f"| before select positive, data_0: {data_0.shape}")

    num_positive = int(num_target * select_pos_ratio)
    select_data_1 = data_1[:num_positive]
    # test_data_1 = data_1[num_positive:]
    test_data_1 = data_1[-400:]
    num_negative = num_target - num_positive
    select_data_0 = data_0[:num_negative]
    test_data_0 = data_0[-2000:]

    print(f"| after select positive, select_data_1: {select_data_1.shape}")
    print(f"| after select positive, select_data_0: {select_data_0.shape}")
    print(f"| after select positive, test_data_1: {test_data_1.shape}")
    print(f"| after select positive, test_data_0: {test_data_0.shape}")

    select_data = np.concatenate((select_data_1, select_data_0), axis=0)
    rest_data = np.concatenate((test_data_1, test_data_0), axis=0)
    return select_data, rest_data


def create_source_and_target(df_dict,
                             df_column_split_list,
                             df_cat_mask_list,
                             to_dir,
                             data_config,
                             all_col_list,
                             train=True,
                             data_2014_tgt_9_test=None,
                             data_2014_tgt_10to12_test=None):
    print(
        "============================= create_degree_source_target_data for  {} data ======================== ".format(
            "train" if train else "test"))

    num_tgt = data_config['num_tgt']
    select_tgt_pos_ratio = data_config['select_tgt_pos_ratio']
    data_tag = data_config['data_tag']

    df_2014_1to8 = df_dict['df_2014_1to8']
    df_2014_1to9 = df_dict['df_2014_1to9']
    df_2014_9 = df_dict['df_2014_9']
    df_2014_10to12 = df_dict['df_2014_10to12']

    df_2014_9_for_da = df_dict.get('df_2014_9_test_for_da')
    df_2014_10to12_test_for_da = df_dict.get('df_2014_10to12_test_for_da')

    print("[INFO] (orig) src: df_2014_1to8:", df_2014_1to8.shape)
    print("[INFO] (orig) src: df_2014_1to9:", df_2014_1to9.shape)
    print("[INFO] (orig) tgt: df_2014_10to12:", df_2014_10to12.shape)
    print("[INFO] (orig) tgt: df_2014_9:", df_2014_9.shape)

    data_2014_src_1to8 = df_2014_1to8.values
    data_2014_src_1to9 = df_2014_1to9.values
    data_2014_tgt_10to12 = df_2014_10to12.values
    data_2014_tgt_9 = df_2014_9.values

    print("[INFO] apply positive sample selection")
    print("[INFO] before select pos samples, data_2014_src_1to8:", data_2014_src_1to8.shape)
    print("[INFO] before select pos samples, data_2014_src_1to9:", data_2014_src_1to9.shape)
    print("[INFO] before select pos samples, data_2014_tgt_10to12:", data_2014_tgt_10to12.shape)
    print("[INFO] before select pos samples, data_2014_tgt_9:", data_2014_tgt_9.shape)

    print("[INFO] select pos samples for target data.")
    data_2014_tgt_9_for_test = None
    data_2014_tgt_10to12_for_test = None
    if train:
        data_2014_tgt_9_ft, data_2014_tgt_9_for_test = select_positive("9",
                                                                       data_2014_tgt_9,
                                                                       num_target=num_tgt,
                                                                       select_pos_ratio=select_tgt_pos_ratio)
        data_2014_tgt_10to12_ft, data_2014_tgt_10to12_for_test = select_positive("10to12",
                                                                                 data_2014_tgt_10to12,
                                                                                 num_target=num_tgt,
                                                                                 select_pos_ratio=select_tgt_pos_ratio)
    else:
        # test
        data_2014_tgt_9_ft = np.concatenate((data_2014_tgt_9, data_2014_tgt_9_test), axis=0)
        data_2014_tgt_10to12_ft = np.concatenate((data_2014_tgt_10to12, data_2014_tgt_10to12_test), axis=0)

    # shuffle
    data_2014_src_1to8 = shuffle(data_2014_src_1to8)
    data_2014_src_1to9 = shuffle(data_2014_src_1to9)

    data_2014_tgt_9 = shuffle(data_2014_tgt_9)
    # print(data_2014_tgt_10to12.shape)
    # print(data_2014_tgt_10to11_da.shape)
    if df_2014_10to12_test_for_da is not None:
        data_2014_tgt_10to12_da = shuffle(np.concatenate((data_2014_tgt_10to12, df_2014_10to12_test_for_da.values), axis=0))
    else:
        data_2014_tgt_10to12_da = data_2014_tgt_10to12
    target_col = data_2014_tgt_10to12_da[:, -1]
    data_1 = data_2014_tgt_10to12_da[target_col == 1]
    data_0 = data_2014_tgt_10to12_da[target_col == 0]
    data_2014_tgt_10to12_da = shuffle(np.concatenate((data_1, data_0[:8000]), axis=0))

    data_2014_tgt_9_ft = shuffle(data_2014_tgt_9_ft)
    data_2014_tgt_10to12_ft = shuffle(data_2014_tgt_10to12_ft)

    print("[INFO] data_2014_src_1to8:", data_2014_src_1to8.shape)
    print("[INFO] data_2014_src_1to9:", data_2014_src_1to9.shape)
    print("[INFO] data_2014_tgt_9:", data_2014_tgt_9.shape)
    print("[INFO] data_2014_tgt_9_ft:", data_2014_tgt_9_ft.shape)
    print("[INFO] data_2014_tgt_10to12_da:", data_2014_tgt_10to12_da.shape)
    print("[INFO] data_2014_tgt_10to12_ft:", data_2014_tgt_10to12_ft.shape)

    target_col = data_2014_src_1to8[:, -1]
    data_1 = data_2014_src_1to8[target_col == 1]
    data_0 = data_2014_src_1to8[target_col == 0]
    print("data_2014_src_1to8 pos:", data_1.shape)
    print("data_2014_src_1to8 neg:", data_0.shape)

    target_col = data_2014_src_1to9[:, -1]
    data_1 = data_2014_src_1to9[target_col == 1]
    data_0 = data_2014_src_1to9[target_col == 0]
    print("data_2014_src_1to9 pos:", data_1.shape)
    print("data_2014_src_1to9 neg:", data_0.shape)

    target_col = data_2014_tgt_10to12_da[:, -1]
    data_1 = data_2014_tgt_10to12_da[target_col == 1]
    data_0 = data_2014_tgt_10to12_da[target_col == 0]
    print("data_2014_tgt_10to12_da pos:", data_1.shape)
    print("data_2014_tgt_10to12_da neg:", data_0.shape)

    target_col = data_2014_tgt_10to12_ft[:, -1]
    data_1 = data_2014_tgt_10to12_ft[target_col == 1]
    data_0 = data_2014_tgt_10to12_ft[target_col == 0]
    print("data_2014_tgt_10to12_ft pos:", data_1.shape)
    print("data_2014_tgt_10to12_ft neg:", data_0.shape)

    # for source domain adaptation and source classification
    df_2014_src_1to8_da = pd.DataFrame(data=data_2014_src_1to8, columns=all_col_list)
    df_2014_src_1to9_da = pd.DataFrame(data=data_2014_src_1to9, columns=all_col_list)

    # target domain adaptation
    df_2014_tgt_9_da = pd.DataFrame(data=data_2014_tgt_9, columns=all_col_list)
    data_2014_tgt_10to12_da = pd.DataFrame(data=data_2014_tgt_10to12_da, columns=all_col_list)

    # target classification (fine-tune)
    df_2014_tgt_9_ft = pd.DataFrame(data=data_2014_tgt_9_ft, columns=all_col_list)
    df_2014_tgt_10to12_ft = pd.DataFrame(data=data_2014_tgt_10to12_ft, columns=all_col_list)

    # standardize
    df_2014_src_1to8_da = standardize(df_2014_src_1to8_da, df_column_split_list, df_cat_mask_list)
    df_2014_src_1to9_da = standardize(df_2014_src_1to9_da, df_column_split_list, df_cat_mask_list)
    df_2014_tgt_9_da = standardize(df_2014_tgt_9_da, df_column_split_list, df_cat_mask_list)
    data_2014_tgt_10to12_da = standardize(data_2014_tgt_10to12_da, df_column_split_list, df_cat_mask_list)
    df_2014_tgt_9_ft = standardize(df_2014_tgt_9_ft, df_column_split_list, df_cat_mask_list)
    df_2014_tgt_10to12_ft = standardize(df_2014_tgt_10to12_ft, df_column_split_list, df_cat_mask_list)

    print("[INFO] (final) df_2014_src_1to8_da: ", df_2014_src_1to8_da.shape)
    print("[INFO] (final) df_2014_src_1to9_da: ", df_2014_src_1to9_da.shape)
    print("[INFO] (final) df_2014_tgt_9_da: ", df_2014_tgt_9_da.shape)
    print("[INFO] (final) data_2014_tgt_10to12_da: ", data_2014_tgt_10to12_da.shape)
    print("[INFO] (final) df_2014_tgt_9_ft: ", df_2014_tgt_9_ft.shape)
    print("[INFO] (final) df_2014_tgt_10to12_ft: ", df_2014_tgt_10to12_ft.shape)

    # save
    mode = "train" if train else "test"
    file_full_name = "{}/PPD_2014_src_1to8_da_{}_{}.csv".format(to_dir, data_tag, mode)
    df_2014_src_1to8_da.to_csv(file_full_name, index=False)
    print(f"[INFO] save df_2014_src_1to8_da to {file_full_name}")

    file_full_name = "{}/PPD_2014_src_1to9_da_{}_{}.csv".format(to_dir, data_tag, mode)
    df_2014_src_1to9_da.to_csv(file_full_name, index=False)
    print(f"[INFO] save df_2014_src_1to9_da to {file_full_name}")

    file_full_name = "{}/PPD_2014_tgt_9_da_{}_{}.csv".format(to_dir, data_tag, mode)
    df_2014_tgt_9_da.to_csv(file_full_name, index=False)
    print(f"[INFO] save df_2014_tgt_9_da to {file_full_name}")

    file_full_name = "{}/PPD_2014_tgt_10to12_da_{}_{}.csv".format(to_dir, data_tag, mode)
    data_2014_tgt_10to12_da.to_csv(file_full_name, index=False)
    print(f"[INFO] save data_2014_tgt_10to12_da to {file_full_name}")

    file_full_name = "{}/PPD_2014_tgt_9_ft_{}_{}.csv".format(to_dir, data_tag, mode)
    df_2014_tgt_9_ft.to_csv(file_full_name, index=False)
    print(f"[INFO] save df_2014_tgt_9_ft to {file_full_name}")

    file_full_name = "{}/PPD_2014_tgt_10to12_ft_{}_{}.csv".format(to_dir, data_tag, mode)
    df_2014_tgt_10to12_ft.to_csv(file_full_name, index=False)
    print(f"[INFO] save df_2014_tgt_10to12_ft to {file_full_name}")

    return data_2014_tgt_9_for_test, data_2014_tgt_10to12_for_test


if __name__ == "__main__":
    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output/"
    # timestamp = '1620085151'
    timestamp = '20210522'
    meta_data = data_dir + 'PPD_meta_data_1620085151.json'

    with open(meta_data) as json_file:
        print(f"[INFO] load task meta file from {meta_data}")
        meta_data_dict = json.load(json_file)

    print("meta_data_dict", meta_data_dict)

    df_cat_mask_list = meta_data_dict['df_cat_mask_list']
    df_column_split_list = meta_data_dict['df_column_split_list']
    # df_all_column_list = meta_data_dict['df_all_column_list'] + ['target']

    from_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_1620085151/"
    to_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{timestamp}/"

    df_2014_train = pd.read_csv(from_dir + "PPD_data_2014_55000_train.csv", skipinitialspace=True)
    df_2014_test = pd.read_csv(from_dir + "PPD_data_2014_55000_test.csv", skipinitialspace=True)
    df_datetime_2014_train = pd.read_csv(from_dir + "PPD_datetime_2014_55000_train.csv", skipinitialspace=True)
    df_datetime_2014_test = pd.read_csv(from_dir + "PPD_datetime_2014_55000_test.csv", skipinitialspace=True)

    print(f"[INFO] df_2014_train.shape:{df_2014_train.shape}")
    print(f"[INFO] df_2014_test.shape:{df_2014_test.shape}")
    print(f"[INFO] df_datetime_2014_train.shape:{df_datetime_2014_train.shape}")
    print(f"[INFO] df_datetime_2014_test.shape:{df_datetime_2014_test.shape}")

    df_2014_1to8_train = df_2014_train[df_datetime_2014_train['ListingInfo_Month'] <= 8]
    df_2014_1to9_train = df_2014_train[df_datetime_2014_train['ListingInfo_Month'] <= 9]
    df_2014_9_train = df_2014_train[df_datetime_2014_train['ListingInfo_Month'] == 9]
    df_2014_10to12_train = df_2014_train[df_datetime_2014_train['ListingInfo_Month'] >= 10]

    df_2014_1to8_test = df_2014_test[df_datetime_2014_test['ListingInfo_Month'] <= 8]
    df_2014_1to9_test = df_2014_test[df_datetime_2014_test['ListingInfo_Month'] <= 9]
    df_2014_9_test = df_2014_test[df_datetime_2014_test['ListingInfo_Month'] == 9]
    df_2014_10to12_test = df_2014_test[df_datetime_2014_test['ListingInfo_Month'] >= 10]

    df_train_dict = {
        "df_2014_1to8": df_2014_1to8_train,
        "df_2014_1to9": df_2014_1to9_train,
        "df_2014_9": df_2014_9_train,
        "df_2014_10to12": df_2014_10to12_train,
        "df_2014_9_test_for_da": df_2014_9_test,
        "df_2014_10to12_test_for_da": df_2014_10to12_test
    }

    df_test_dict = {
        "df_2014_1to8": df_2014_1to8_test,
        "df_2014_1to9": df_2014_1to9_test,
        "df_2014_9": df_2014_9_test,
        "df_2014_10to12": df_2014_10to12_test
    }

    num_tgt = 4000
    data_tag = 'lbl001tgt4000'
    select_tgt_pos_ratio = 0.01

    data_config = {"select_tgt_pos_ratio": select_tgt_pos_ratio,
                   "num_tgt": num_tgt,
                   "data_tag": data_tag}

    data_2014_tgt_9_rest, data_2014_tgt_10to11_rest = create_source_and_target(df_train_dict,
                                                                               df_column_split_list,
                                                                               df_cat_mask_list,
                                                                               to_dir,
                                                                               data_config,
                                                                               all_col_list=df_2014_train.columns,
                                                                               train=True)
    create_source_and_target(df_test_dict,
                             df_column_split_list,
                             df_cat_mask_list,
                             to_dir,
                             data_config,
                             all_col_list=df_2014_train.columns,
                             train=False,
                             data_2014_tgt_9_test=data_2014_tgt_9_rest,
                             data_2014_tgt_10to12_test=data_2014_tgt_10to11_rest)
