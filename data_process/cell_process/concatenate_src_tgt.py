from collections import OrderedDict

import numpy as np
import pandas as pd


def create_df(values, cols):
    return pd.DataFrame(data=values, columns=cols)


def save_combined_df(data_dict, to_dir):
    df_list_dict = {"train": list(), "train_us": list(), "val": list(), "test": list()}
    for table_name, df_dict in data_dict.items():
        for key, df in df_dict.items():
            df_list_dict[key].append(df)

    for key, df_list in df_list_dict.items():
        if len(df_list) == 0:
            continue
        comb_df = pd.concat(df_list, axis=1)
        file_name = key + "_all_feat.csv"
        save_df(comb_df, to_dir, file_name)


def save_all_df(data_dict, to_dir):
    for table_name, df_dict in data_dict.items():
        for key, df in df_dict.items():
            file_name = key + "_" + table_name
            save_df(df, to_dir, file_name)


def save_df(df, to_dir, file_name):
    file_path = to_dir + file_name
    df.to_csv(file_path, index=False)
    print(f"save {file_name} with shape {df.shape} to {file_path}")


def up_sampling(values, up_samples_size):
    size = values.shape[0]
    sampled_idxs = np.random.choice(size, up_samples_size, replace=True)
    return values[sampled_idxs]


def split_train_val_test(dir,
                         table_names,
                         num_train,
                         num_val=None,
                         create_val_data=False,
                         perm_indices=None,
                         train_indices=None,
                         non_train_indices=None,
                         up_sample_train_idxs=None):
    result_dict = OrderedDict()
    for table_name in table_names:
        result_dict[table_name] = dict()

        df = pd.read_csv(dir + table_name)
        cols = df.columns
        data_values = df.values
        print(f"[INFO] loaded {table_name} table, which has shape:{data_values.shape}")

        if perm_indices is not None:
            data_values = data_values[perm_indices]
            print(f"[INFO] data have been shuffled.")

        # train_indices = np.random.permutation(np.arange(len(data_values)))[:num_train]
        # if num_train is None:
        #     train_values = data_values[train_indices]
        # else:
        #     train_values = data_values[:num_train]
        train_values = data_values[:num_train] if num_train is not None else data_values[train_indices]
        if up_sample_train_idxs is not None:
            print(f"[INFO] non-up-sampled train_values has shape:{train_values.shape}")
            result_dict[table_name]["train"] = create_df(values=train_values, cols=cols)

            train_values = train_values[up_sample_train_idxs]
            print(f"[INFO] up-sampling train_values has shape:{train_values.shape}")
            result_dict[table_name]["train_us"] = create_df(values=train_values, cols=cols)
        else:
            print(f"[INFO] train_values has shape:{train_values.shape}")
            result_dict[table_name]["train"] = create_df(values=train_values, cols=cols)

        # rest of the data treated as validation/test data
        # if num_train < data_values.shape[0]:
        if num_train is None:
            test_values = data_values[non_train_indices]
        else:
            test_values = data_values[num_train:]
        if create_val_data:
            print("[INFO] creating val_values and test_values ...")
            num_val = int(test_values.shape[0] / 2) if num_val is None else num_val
            val_values = test_values[:num_val]
            test_values = test_values[num_val:]
            print(f"[INFO] val_values has shape:{val_values.shape}.")
            result_dict[table_name]["val"] = create_df(values=val_values, cols=cols)
        print(f"[INFO] test_values has shape:{test_values.shape}.")
        result_dict[table_name]["test"] = create_df(values=test_values, cols=cols)
    return result_dict


def split_A_data(table_names):
    print("==> split train, val, test of A data")

    # num_total_samples = 177986
    num_total_samples = 217847
    perm_idxs = np.random.permutation(num_total_samples)

    num_train = 180000
    num_val = None

    dir = "../../../data/cell_manager/A_train_data_3/"
    to_dir = "../../../data/cell_manager/A_train_data_3/"
    result_dict = split_train_val_test(dir, table_names, num_train, num_val,
                                       create_val_data=False, perm_indices=perm_idxs,
                                       up_sample_train_idxs=None)
    save_all_df(result_dict, to_dir)
    save_combined_df(result_dict, to_dir)


def split_B_data(table_names, original_data_dir, prepared_data_dir, create_val_data=False):
    print("==> split train, val, test of B data")

    num_total_samples = 61192
    # num_total_samples = 29636
    perm_idxs = np.random.permutation(num_total_samples)

    # num_train = 29636
    # num_train = 25000
    num_train = 10000
    # perm_idxs = None
    num_val = 10000
    up_sample_idxs = np.random.choice(num_train, 180000, replace=True)

    result_dict = split_train_val_test(original_data_dir, table_names, num_train, num_val,
                                       create_val_data=create_val_data, perm_indices=perm_idxs,
                                       up_sample_train_idxs=up_sample_idxs)
    save_all_df(result_dict, prepared_data_dir)
    save_combined_df(result_dict, prepared_data_dir)


def split_C_data(table_names):
    print("==> split train, val, test of C data")

    num_total_samples = 9891
    perm_idxs = np.random.permutation(num_total_samples)

    num_train = 5000
    # num_train = 7891
    # perm_idxs = None
    num_val = None
    up_sample_idxs = np.random.choice(num_train, 180000, replace=True)

    dir = "../../../data/cell_manager/C_train_data_3/"
    to_dir = "../../../data/cell_manager/C_train_data_3/"
    result_dict = split_train_val_test(dir, table_names, num_train, num_val,
                                       create_val_data=False, perm_indices=perm_idxs,
                                       up_sample_train_idxs=up_sample_idxs)
    save_all_df(result_dict, to_dir)


if __name__ == "__main__":

    # table_names = ["demographics.csv", "equipment.csv", "risk.csv",
    #                "cell_number_appears.csv", "app_finance_install.csv", "app_finance_usage.csv",
    #                "app_life_install.csv", "app_life_usage.csv", "fraud.csv", "tgi.csv", "target.csv"]
    table_names = ["demo.csv", "asset.csv", "equipment.csv", "risk.csv",
                   "cell_number_appears.csv", "app_finance_install.csv", "app_finance_usage.csv",
                   "app_life_install.csv", "app_life_usage.csv", "fraud.csv", "tgi.csv", "target.csv"]

    source_dir = "/Users/yankang/Documents/Data/cell_manager/A_train_data_3/"
    target_dir = "../../../data/cell_manager/B_train_data_4/"
    to_dir = "../../../data/cell_manager/A_B_train_data/"
    for table_name in table_names:
        table_name = "train_" + table_name
        print("concatenate {0}".format(table_name))
        df_src = pd.read_csv(source_dir + table_name)
        df_tgt = pd.read_csv(target_dir + table_name)
        df_all = pd.concat([df_src, df_tgt], axis=0)
        print("df_src shape:{0}".format(df_src.shape))
        print("df_tgt shape:{0}".format(df_tgt.shape))
        print("df_all shape:{0}".format(df_all.shape))
        save_df(df_all, to_dir, table_name)

    print("finished concatenate data!")
