import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from benchmark.benchmark_utils import run_benchmark, save_benchmark_result, find_args_for_best_metric
from data_process.census_process.mapping_resource import categorical_cols, continuous_cols, target_col_name


def train_benchmark(samples_train, samples_test, tag):
    train_data, train_label = samples_train[:, :-1], samples_train[:, -1]
    test_data, test_label = samples_test[:, :-1], samples_test[:, -1]

    print(f"[INFO] train_data shape:{train_data.shape}")
    print(f"[INFO] train_label shape:{train_label.shape}， {np.sum(train_label)}")
    print(f"[INFO] test_data shape:{test_data.shape}")
    print(f"[INFO] test_label shape:{test_label.shape}， {np.sum(test_label)}")

    n_tree_list = [200, 300, 400]
    max_depth_list = [2, 4, 6, 8]
    # n_tree_list = [10, 20, 30]
    # max_depth_list = [2, 4, 6]
    result_list = list()
    for n_tree in n_tree_list:
        for max_depth in max_depth_list:
            kwargs = {'n_tree_estimators': n_tree, 'max_depth': max_depth}
            result_dict = run_benchmark(train_data, train_label, test_data, test_label, **kwargs)
            result_list.append((kwargs, result_dict))
    for result in result_list:
        print("-" * 100)
        print("args:", result[0])
        print("result:", result[1])

    best_auc, best_arg = find_args_for_best_metric(result_list, model_name='xgb', metric_name='auc')

    file_name = "income_benchmark_result_" + tag
    result_dict = dict()
    result_dict["income_benchmark_result"] = result_list
    result_dict['best_result'] = {'xgb': {'best_auc': best_auc, "best_arg": best_arg}}
    save_benchmark_result(result=result_dict, to_dir="../../data_process/census_process/", file_name=file_name)


def train_on_dann(file_dict, continuous_cols, columns=None, data_tag=None, label_name='income_label'):
    source_train_file = file_dict['source_train_file']
    target_train_file = file_dict['target_train_file']
    target_test_file = file_dict['target_test_file']
    data_dir = file_dict['data_dir']

    adult_source_train = pd.read_csv(data_dir + source_train_file, skipinitialspace=True)
    adult_target_train = pd.read_csv(data_dir + target_train_file, skipinitialspace=True)
    adult_target_test = pd.read_csv(data_dir + target_test_file, skipinitialspace=True)

    if columns:
        adult_source_train = adult_source_train[columns]
        adult_target_train = adult_target_train[columns]
        adult_target_test = adult_target_test[columns]

    src_train_num = adult_source_train.shape[0]
    tgt_train_num = adult_target_train.shape[0]
    tgt_test_num = adult_target_test.shape[0]

    src_train_pos_num = adult_source_train[adult_source_train[label_name] == 1].shape[0]
    tgt_train_pos_num = adult_target_train[adult_target_train[label_name] == 1].shape[0]
    tgt_test_pos_num = adult_target_test[adult_target_test[label_name] == 1].shape[0]

    src_train_pos_ratio = src_train_pos_num / src_train_num
    tgt_train_pos_ratio = tgt_train_pos_num / tgt_train_num
    tgt_test_pos_ratio = tgt_test_pos_num / tgt_test_num

    src_train_neg_num = adult_source_train[adult_source_train[label_name] == 0].shape[0]
    tgt_train_neg_num = adult_target_train[adult_target_train[label_name] == 0].shape[0]
    tgt_test_neg_num = adult_target_test[adult_target_test[label_name] == 0].shape[0]

    print(
        f"source_train shape:{src_train_num}, pos:{src_train_pos_num}({src_train_pos_ratio}), neg:{src_train_neg_num}")
    print(
        f"target_train shape:{tgt_train_num}, pos:{tgt_train_pos_num}({tgt_train_pos_ratio}), neg:{tgt_train_neg_num}")
    print(f"target_test shape:{tgt_test_num}, pos:{tgt_test_pos_num}({tgt_test_pos_ratio}), neg:{tgt_test_neg_num}")

    print("====== train model only on local target =======")
    tgt_train_label = adult_target_train.values[:, -1].reshape(-1, 1)
    tgt_test_label = adult_target_test.values[:, -1].reshape(-1, 1)
    local_target_train_data = np.concatenate([adult_target_train[continuous_cols].values, tgt_train_label], axis=1)
    local_target_test_data = np.concatenate([adult_target_test[continuous_cols].values, tgt_test_label], axis=1)
    print(f"local_target_train_data shape:{local_target_train_data.shape}")
    print(f"local_target_test_data shape:{local_target_test_data.shape}")
    tag = "local" + '_' + data_tag
    local_train_data = shuffle(local_target_train_data)
    local_test_data = shuffle(local_target_test_data)
    train_benchmark(local_train_data, local_test_data, tag)
    print('\n')

    # adult_target_train = adult_target_train.values
    # adult_target_test = adult_target_test.values
    # adult_target_train = shuffle(adult_target_train)
    # adult_target_test = shuffle(adult_target_test)

    # print("=========================================")
    # print("====== train model only on target =======")
    # print("=========================================")
    # tag = "tgt" + '_' + data_tag
    # train_benchmark(adult_target_train, adult_target_test, tag)
    # print('\n')

    # print("=========================================")
    # print("====== train model on src + tgt =========")
    # print("=========================================")
    # if data_tag:
    #     tag = "src_tgt" + '_' + data_tag
    # else:
    #     tag = "src_tgt"
    # adult_source_train = adult_source_train.values
    # adult_all_train = np.concatenate([adult_source_train, adult_target_train], axis=0)
    # adult_all_train = shuffle(adult_all_train)
    # train_benchmark(adult_all_train, adult_target_test, tag)


if __name__ == "__main__":
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    tag = "all4000pos004v4"
    # tag = "all4000pos002"
    # tag = "all4000pos001"
    source_train_file = f'undergrad_census9495_da_{tag}_train.csv'
    target_train_file = f'grad_census9495_ft_{tag}_train.csv'
    target_test_file = f'grad_census9495_ft_{tag}_test.csv'

    columns_list = continuous_cols + categorical_cols + [target_col_name]
    print("[INFO] columns_list:", len(columns_list), columns_list)

    file_dict = dict()
    file_dict['source_train_file'] = source_train_file
    file_dict['target_train_file'] = target_train_file
    file_dict['target_test_file'] = target_test_file
    file_dict['data_dir'] = data_dir
    train_on_dann(file_dict,
                  columns=columns_list,
                  data_tag=tag,
                  label_name=target_col_name,
                  continuous_cols=continuous_cols)
