import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from datasets.ppd_dataloader import get_selected_columns
from data_process.cell_process.bechmark import run_benchmark


def train_benchmark(samples_train, samples_test):
    train_data, train_label = samples_train[:, :-1], samples_train[:, -1]
    test_data, test_label = samples_test[:, :-1], samples_test[:, -1]

    print(f"train_data shape : {train_data.shape}")
    print(f"train_label shape : {train_label.shape}")
    print(f"test_data shape : {test_data.shape}")
    print(f"test_label shape : {test_label.shape}")
    run_benchmark(train_data, train_label, test_data, test_label)


def train_on_dann(file_dict):
    source_train_file = file_dict['source_train_file']
    target_train_file = file_dict['target_train_file']
    target_test_file = file_dict['target_test_file']
    data_dir = file_dict['data_dir']

    src_train_data = pd.read_csv(data_dir + source_train_file, skipinitialspace=True)
    tgt_train_data = pd.read_csv(data_dir + target_train_file, skipinitialspace=True)
    tgt_test_data = pd.read_csv(data_dir + target_test_file, skipinitialspace=True)

    train_columns = get_selected_columns(src_train_data, use_all=True)
    print("[INFO] select_columns:", len(train_columns), train_columns)
    src_train_data = src_train_data[train_columns]
    tgt_train_data = tgt_train_data[train_columns]
    tgt_test_data = tgt_test_data[train_columns]

    print("[INFO] columns:", src_train_data.columns, len(src_train_data.columns))
    print("[INFO] columns:", tgt_train_data.columns, len(tgt_train_data.columns))
    print("[INFO] columns:", tgt_test_data.columns, len(tgt_test_data.columns))

    tgt_train_label = tgt_train_data.values[:, -1].reshape(-1, 1)
    tgt_test_label = tgt_test_data.values[:, -1].reshape(-1, 1)
    local_target_train_data = np.concatenate([tgt_train_data.values[:, :17], tgt_train_label], axis=1)
    local_target_test_data = np.concatenate([tgt_test_data.values[:, :17], tgt_test_label], axis=1)

    print(f"source_train shape:{src_train_data.shape}")
    print(f"target_train shape:{tgt_train_data.shape}")
    print(f"target_test shape:{tgt_test_data.shape}")
    print(f"local_target_train_data shape:{local_target_train_data.shape}")
    print(f"local_target_test_data shape:{local_target_test_data.shape}")

    print("====== train model only on local target =======")
    local_train_data = shuffle(local_target_train_data)
    local_test_data = shuffle(local_target_test_data)
    train_benchmark(local_train_data, local_test_data)
    print('\n')
    print("====== train model only on target =======")
    tgt_train_data = tgt_train_data.values
    tgt_test_data = tgt_test_data.values
    tgt_train_data = shuffle(tgt_train_data)
    tgt_test_data = shuffle(tgt_test_data)
    train_benchmark(tgt_train_data, tgt_test_data)
    print('\n')
    print("====== train model on src+tgt =======")
    src_train_data = src_train_data.values
    adult_all_train = np.concatenate([src_train_data, tgt_train_data], axis=0)
    adult_all_train = shuffle(adult_all_train)
    train_benchmark(adult_all_train, tgt_test_data)


if __name__ == "__main__":
    source_train_file = "PPD_2014_1to9_train.csv"
    target_train_file = 'PPD_2014_10to12_train.csv'
    # source_test_file = 'PPD_2014_1to9_test.csv'
    target_test_file = 'PPD_2014_10to12_test.csv'
    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    columns_list = None

    file_dict = dict()
    file_dict['source_train_file'] = source_train_file
    file_dict['target_train_file'] = target_train_file
    file_dict['target_test_file'] = target_test_file
    file_dict['data_dir'] = data_dir
    train_on_dann(file_dict)
