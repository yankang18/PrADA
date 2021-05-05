import numpy as np

from data_process.benchmark_utils import run_benchmark
from datasets.cell_manage_dataloader import get_data


def get_datasets(dir, file_name_list, data_mode, is_shuffle=True, suffix=None):
    data_list, target = get_data(dir, file_name_list, data_mode, is_shuffle=is_shuffle, suffix=suffix)
    if len(data_list) > 1:
        combined_data = np.concatenate(data_list, axis=1)
    else:
        combined_data = data_list[0]
    return combined_data, target


# table_names = ["p_named_demographics.csv", "p_named_asset.csv", "p_named_equipment.csv", "p_named_risk.csv",
#                "p_named_cell_number_appears.csv", "p_app_finance_install.csv", "p_app_finance_usage.csv",
#                "p_app_life_install.csv", "p_app_life_usage.csv", "target.csv"]
# table_names = ["p_named_demographics.csv", "p_named_equipment.csv", "p_named_risk.csv",
#                "p_named_cell_number_appears.csv", "p_tgi.csv", "p_fraud.csv", "p_app_finance_install.csv",
#                "p_app_finance_usage.csv", "p_app_life_install.csv", "p_app_life_usage.csv", "target.csv"]
table_names = ["demographics.csv", "equipment.csv", "risk.csv",
               "cell_number_appears.csv", "tgi.csv", "fraud.csv", "app_finance_install.csv",
               "app_finance_usage.csv", "app_life_install.csv", "app_life_usage.csv", "target.csv"]


def prepare_data(data_dir, table_names):
    train_data, train_label = get_datasets(data_dir, table_names, "train")
    print(f"train_data shape:{train_data.shape}")
    print(f"train_label shape:{train_label.shape}， {np.sum(train_label)}")

    test_data, test_label = get_datasets(data_dir, table_names, "test")
    print(f"test_data shape:{test_data.shape}")
    print(f"test_label shape:{test_label.shape}， {np.sum(test_label)}")

    return train_data, train_label, test_data, test_label


def run_local_benchmark():
    local_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_4/"
    local_data = prepare_data(local_dir, ["demo.csv", "target.csv"])
    train_data, train_lbl, test_data, test_lbl = local_data
    run_benchmark(train_data=train_data, train_label=train_lbl, test_data=test_data, test_label=test_lbl)


if __name__ == "__main__":
    run_local_benchmark()
    # # src_dir = "../../../data/cell_manager/A_train_data_2/"
    # # tgt_dir = "../../../data/cell_manager/B_train_data_2/"
    # # tgt_dir = "../../../data/cell_manager/C_train_data_2/"
    #
    # src_dir = "/Users/yankang/Documents/Data/cell_manager/A_train_data_3/"
    # # tgt_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_3/"
    # tgt_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_4/"
    #
    # src_train_data, src_train_label = get_datasets(src_dir, table_names, "train")
    # print(f"src_train_data shape:{src_train_data.shape}")
    # print(f"src_train_label shape:{src_train_label.shape}， {np.sum(src_train_label)}")
    #
    # tgt_train_data, tgt_train_label = get_datasets(tgt_dir, table_names, "train")
    # print(f"tgt_train_data shape:{tgt_train_data.shape}")
    # print(f"tgt_train_label shape:{tgt_train_label.shape}， {np.sum(tgt_train_label)}")
    #
    # src_test_data, src_test_label = get_datasets(src_dir, table_names, "test")
    # print(f"src_test_data shape:{src_test_data.shape}")
    # print(f"src_test_label shape:{src_test_label.shape}， {np.sum(src_test_label)}")
    #
    # tgt_test_data, tgt_test_label = get_datasets(tgt_dir, table_names, "test")
    # print(f"tgt_test_data shape:{tgt_test_data.shape}")
    # print(f"tgt_test_label shape:{tgt_test_label.shape}， {np.sum(tgt_test_label)}")
    #
    # all_train_data = np.concatenate([src_train_data, tgt_train_data], axis=0)
    # all_train_label = np.concatenate([src_train_label, tgt_train_label], axis=0)
    #
    # all_test_data = np.concatenate([src_test_data, tgt_test_data], axis=0)
    # all_test_label = np.concatenate([src_test_label, tgt_test_label], axis=0)
    #
    # # train_data = tgt_train_data
    # # train_label = tgt_train_label
    # all_train_data, all_train_label = shuffle(all_train_data, all_train_label)
    # all_train_label = all_train_label.ravel()
    #
    # # run_benchmark(train_data, train_label, tgt_test_data, tgt_test_label)
    # # run_benchmark(all_train_data, all_train_label, all_test_data, all_test_label)
    # run_benchmark(train_data=all_train_data, train_label=all_train_label,
    #               test_data=all_train_data, test_label=all_train_label)
    # # run_benchmark(all_train_data, all_train_label, src_test_data, src_test_label)
    #
    # # only using data of C
    # # C_train_data, C_train_label = shuffle(C_train_data, C_train_label)
    # # C_train_label = C_train_label.ravel()
    # # C_test_data, C_test_label = get_datasets(C_dir, table_names, "test")
    # # print(f"C_train_data shape:{C_train_data.shape}")
    # # print(f"C_train_label shape:{C_train_label.shape}， {np.sum(C_train_label)}")
    # # run_benchmark(C_train_data, C_train_label, C_test_data, C_test_label)
    # # # run_benchmark(train_data, train_label, C_test_data, C_test_label)
