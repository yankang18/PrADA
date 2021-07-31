import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from benchmark.benchmark_utils import run_benchmark, save_benchmark_result, find_args_for_best_metric
from datasets.ppd_dataloader import get_selected_columns


def train_benchmark(samples_train, samples_test, tag):
    train_data, train_label = samples_train[:, :-1], samples_train[:, -1]
    test_data, test_label = samples_test[:, :-1], samples_test[:, -1]

    print(f"[INFO] train_data shape : {train_data.shape}")
    print(f"[INFO] train_label shape : {train_label.shape}")
    print(f"[INFO] test_data shape : {test_data.shape}")
    print(f"[INFO] test_label shape : {test_label.shape}")

    n_tree_list = [200, 300, 400]
    max_depth_list = [2, 4, 6, 8]
    result_list = list()
    for n_tree in n_tree_list:
        for max_depth in max_depth_list:
            kwargs = {'n_tree_estimators': n_tree, 'max_depth': max_depth}
            result_dict = run_benchmark(train_data=train_data,
                                        train_label=train_label,
                                        test_data=test_data,
                                        test_label=test_label,
                                        **kwargs)
            result_list.append((kwargs, result_dict))
    for result in result_list:
        print("-" * 100)
        print("args:", result[0])
        print("result:", result[1])

    best_auc, best_arg = find_args_for_best_metric(result_list, model_name='xgb', metric_name='auc')

    file_name = f"ppd_benchmark_{tag}_result"
    result_dict = dict()
    result_dict["ppd_benchmark_result"] = result_list
    result_dict['best_result'] = {'xgb': {'best_auc': best_auc, "best_arg": best_arg}}
    save_benchmark_result(result=result_dict, to_dir="../../data_process/ppd_process/", file_name=file_name)


def train_on_dann(file_dict, data_tag):
    source_train_file = file_dict['source_train_file']
    target_train_file = file_dict['target_train_file']
    target_test_file = file_dict['target_test_file']
    data_dir = file_dict['data_dir']

    src_train_data = pd.read_csv(data_dir + source_train_file, skipinitialspace=True, dtype={'target': int})
    tgt_train_data = pd.read_csv(data_dir + target_train_file, skipinitialspace=True, dtype={'target': int})
    tgt_test_data = pd.read_csv(data_dir + target_test_file, skipinitialspace=True, dtype={'target': int})

    train_columns = get_selected_columns(src_train_data, use_all_features=False)
    print("[INFO] select_columns:", len(train_columns), train_columns)
    src_train_data = src_train_data[train_columns]
    tgt_train_data = tgt_train_data[train_columns]
    tgt_test_data = tgt_test_data[train_columns]

    print("[INFO] columns:", src_train_data.columns, len(src_train_data.columns))
    print("[INFO] columns:", tgt_train_data.columns, len(tgt_train_data.columns))
    print("[INFO] columns:", tgt_test_data.columns, len(tgt_test_data.columns))

    print(f"source_train shape:{src_train_data.shape}")
    print(f"target_train shape:{tgt_train_data.shape}")
    print(f"target_test shape:{tgt_test_data.shape}")

    print("====== train model only on local target =======")
    tgt_train_label = tgt_train_data.values[:, -1].reshape(-1, 1)
    tgt_test_label = tgt_test_data.values[:, -1].reshape(-1, 1)
    local_target_train_data = np.concatenate([tgt_train_data.values[:, :6], tgt_train_label], axis=1)
    local_target_test_data = np.concatenate([tgt_test_data.values[:, :6], tgt_test_label], axis=1)
    print(f"local_target_train_data shape:{local_target_train_data.shape}")
    print(f"local_target_test_data shape:{local_target_test_data.shape}")
    tag = data_tag + "_local"
    local_train_data = shuffle(local_target_train_data)
    local_test_data = shuffle(local_target_test_data)
    train_benchmark(local_train_data, local_test_data, tag)
    print('\n')

    # tgt_train_data = tgt_train_data.values
    # tgt_test_data = tgt_test_data.values
    # tgt_train_data = shuffle(tgt_train_data)
    # tgt_test_data = shuffle(tgt_test_data)

    # print("====== train model only on target =======")
    # tag = data_tag + "_target"
    # train_benchmark(tgt_train_data, tgt_test_data, tag)
    # print('\n')

    # print("====== train model on src+tgt =======")
    # tag = data_tag + "_all"
    # src_train_data = src_train_data.values
    # all_train_data = np.concatenate([src_train_data, tgt_train_data], axis=0)
    # all_train_data = shuffle(all_train_data)
    # train_benchmark(all_train_data, tgt_test_data, tag)


if __name__ == "__main__":

    timestamp = '20210522'
    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{timestamp}/"
    tgt_data_tag = ''
    # data_tag = 'lbl004tgt4000v4'
    # data_tag = 'lbl002tgt4000'
    data_tag = 'lbl001tgt4000'

    source_train_file = f"PPD_2014_src_1to9_da_{data_tag}_train.csv"
    # source_train_file = f"PPD_2014_src_1to8_da_{data_tag}_train.csv"
    target_train_file = f'PPD_2014_tgt_10to12_ft_{data_tag}_train.csv'
    target_test_file = f'PPD_2014_tgt_10to12_ft_{data_tag}_test.csv'

    # tgt_data_tag = "test"
    # tgt_tag = "test"
    file_dict = dict()
    file_dict['source_train_file'] = source_train_file
    file_dict['target_train_file'] = target_train_file
    file_dict['target_test_file'] = target_test_file
    file_dict['data_dir'] = data_dir
    train_on_dann(file_dict, data_tag)
