import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import get_latest_timestamp
# from scipy.interpolate import spline
from scipy.interpolate import make_interp_spline


def load_exp_result_auc_list(root, task_folder, timestamp=None):
    # task_folder = "task_" + task_id
    task_folder_path = os.path.join(root, task_folder)
    if not os.path.exists(task_folder_path):
        raise FileNotFoundError(f"{task_folder_path} is not found.")
    print(f"[INFO] load model from:{task_folder_path}")

    if timestamp is None:
        timestamp = get_latest_timestamp("models_checkpoint", task_folder_path)
        print(f"[INFO] get latest timestamp {timestamp}")

    dann_exp_result_file_name = "dann_exp_result_" + str(timestamp) + ".json"
    dann_exp_result_file_name = os.path.join(task_folder_path, dann_exp_result_file_name)
    if not os.path.exists(dann_exp_result_file_name):
        raise FileNotFoundError(f"{dann_exp_result_file_name} is not found.")

    with open(dann_exp_result_file_name) as json_file:
        print(f"[INFO] load dann exp result file meta file from {dann_exp_result_file_name}")
        dann_exp_result_dict = json.load(json_file)

    return dann_exp_result_dict["metrics"]


if __name__ == "__main__":
    batch_64_list = [
        "task_20200127_cell_pw1.0_bs64_lr0.001_v0_t1611631270",
        # "task_20200127_cell_pw1.0_bs64_lr0.001_v1_t1611631270",
        # "task_20200127_cell_pw1.0_bs64_lr0.001_v0_t1611665448",
        "task_20200127_cell_pw1.0_bs64_lr0.001_v1_t1611665448",
        "task_20200128_cell_pw1.0_bs64_lr0.001_v0_t1611764455"
    ]
    batch_128_list = [
        "task_20200128_cell_pw1.0_bs128_lr0.001_v0_t1611722192",
        "task_20200128_cell_pw1.0_bs128_lr0.001_v1_t1611722192",
        # "task_20200128_cell_pw1.0_bs128_lr0.001_v2_t1611722192",
        # "task_20200127_cell_pw1.0_bs128_lr0.001_v0_t1611645886",
        "task_20200127_cell_pw1.0_bs128_lr0.001_v1_t1611645886",
        # "task_20200127_cell_pw1.0_bs128_lr0.001_v2_t1611645886",
        # "task_20200127_cell_pw1.0_bs128_lr0.001_v3_t1611645886",
        "task_20200128_cell_pw1.0_bs128_lr0.001_v0_t1611686000",
        # "task_20200128_cell_pw1.0_bs128_lr0.001_v0_t1611689760"
    ]
    batch_256_list = [
        "task_20200115_cell_256_0.001_v0_1610697233",
        "task_20200115_cell_256_0.001_v0_1610772062",
        "task_20200115_cell_256_0.001_v0_1610931263",
        # "task_20200115_cell_256_0.001_v1_1610697233",
        "task_20200128_cell_pw1.0_bs256_lr0.001_v0_t1611693542"
    ]
    batch_512_list = [
        # "task_20200128_cell_pw1.0_bs512_lr0.001_v0_t1611697843",
        "task_20200128_cell_pw1.0_bs512_lr0.001_v1_t1611697843",
        "task_20200128_cell_pw1.0_bs512_lr0.001_v2_t1611697843",
        # "task_20200127_cell_pw1.0_bs512_lr0.001_v0_t1611612627",
        "task_20200127_cell_pw1.0_bs512_lr0.001_v1_t1611612627",
        # "task_20200127_cell_pw1.0_bs512_lr0.001_v2_t1611612627",
        # "task_20200127_cell_pw1.0_bs512_lr0.001_v3_t1611612627",
        # "task_20200128_cell_pw1.0_bs512_lr0.001_v0_t1611697843",
        "task_20200128_cell_pw1.0_bs512_lr0.001_v1_t1611697843",
        "task_20200128_cell_pw1.0_bs512_lr0.001_v2_t1611697843"
    ]
    batch_1024_list = [
        "task_20200127_cell_pw1.0_bs1024_lr0.001_v0_t1611612627",
        "task_20200127_cell_pw1.0_bs1024_lr0.001_v1_t1611612627",
        # "task_20200127_cell_pw1.0_bs1024_lr0.001_v0_t1611674951",
        # "task_20200127_cell_pw1.0_bs1024_lr0.001_v3_t1611612627",
        # "task_20200128_cell_pw1.0_bs1024_lr0.001_v0_t1611708419",
        "task_20200128_cell_pw1.0_bs1024_lr0.001_v0_t1611710735",
        "task_20200128_cell_pw1.0_bs1024_lr0.001_v1_t1611710735"
    ]

    exp_batch_dic = {'64': batch_64_list,
                     '128': batch_128_list,
                     '256': batch_256_list,
                     '512': batch_512_list,
                     '1024': batch_1024_list}

    dann_root_folder = "cell_dann/"
    # task_id = "20200127_cell_pw1.0_bs1024_lr0.001_v0_t1611612627"
    # auc_list = load_exp_result_auc_list(dann_root_folder, task_id)
    # print(auc_list)

    auc_threshold = 0.69

    valid_batch_interval = 10
    exp_results = dict()
    batch_spend_results = dict()
    for batch_num, batch_list in exp_batch_dic.items():
        batch_auc_list = list()
        batch_spend_list = list()
        auc_threshold_batch_list = list()
        for task_folder in batch_list:
            metrcis_dict = load_exp_result_auc_list(dann_root_folder, task_folder)
            auc_list = metrcis_dict["target_batch_auc_list"]
            batch_auc_list.append(auc_list)

            auc_threshold_batch = 0
            count = 0
            for i, val in enumerate(auc_list):
                if val >= auc_threshold:
                    auc_threshold_batch = i
                    count += 1
                    if count == 3:
                        break
                else:
                    count = 0
            auc_threshold_batch_list.append(auc_threshold_batch)

            optim_epoch = metrcis_dict.get("current_epoch")
            if optim_epoch:
                optim_batch = metrcis_dict["current_batch_idx"]
                num_batch = metrcis_dict["num_batches_per_epoch"]
                batch_spend = optim_epoch * num_batch + optim_batch
                batch_spend_list.append(batch_spend)
        exp_results[batch_num] = batch_auc_list
        # batch_spend_results[batch_num] = np.mean(batch_spend_list)
        batch_spend_results[batch_num] = np.mean(auc_threshold_batch_list)

    print("exp_results:\n")
    print(exp_results)
    # exp_results['128'] = [auc_128_0, auc_128_1, auc_128_2, auc_128_3, auc_128_4]
    # exp_results['256'] = [auc_256_0, auc_256_1, auc_256_2, auc_256_3, auc_256_4]
    # exp_results['512'] = [auc_512_0, auc_512_1, auc_512_2, auc_512_3, auc_512_4]
    # exp_results['1024'] = [auc_1024_0, auc_1024_1, auc_1024_2, auc_1024_3, auc_1024_4]

    plt.rcParams['pdf.fonttype'] = 42
    # batch_time = [2.91849, 3.4858, 4.22333, 5.2235, 8.08644, 13.04606]
    # batch_time = [2.94009, 3.5107, 4.26143, 5.27, 8.15334, 13.17356]
    # batch_time = [3.03168, 3.6202, 4.39316, 5.433, 8.40628, 13.57932]
    batch_time = [3.38535, 4.09782, 5.00277, 6.4416, 9.47094, 15.53208]
    batch_size = ["64", "128", "256", "512", "1024"]
    batch_time_dict = {}
    for bs, bt in zip(batch_size, batch_time):
        batch_time_dict[bs] = bt
        bn = batch_spend_results[bs] * 10
        all_time = bn * bt
        print(bs, all_time, bn, bt)
    print("batch_time_dict:", batch_time_dict)
    # batch_time_dict = {'64': 0.8439, '128': 0.992, '256': 1.1603, '512': 1.475, '1024': 2.3064}
    color_dict = {'64': 'red', '128': 'black', '256': 'green', '512': 'blue', '1024': 'olive'}
    for key, value in exp_results.items():
        print("---- {} ----".format(key))
        len_list = [len(x) for x in value]
        print("len_list", len_list)
        min_len = np.min(len_list)
        # max_idx = np.argmax(len_list)
        # print(max_idx)
        # auc_list = value[max_idx]
        # min_len = len(auc_list)
        # min_len = 200

        batch_time = batch_time_dict[key]
        x = np.array(range(min_len))
        x = x * batch_time * valid_batch_interval

        batch_auc_list = [np.array(x)[:min_len] for x in value]
        batch_auc_mean = np.mean(batch_auc_list, axis=0)

        # xnew = np.linspace(x.min(), x.max(), 300)
        # # define spline
        # spl = make_interp_spline(x, auc_list)
        # y_smooth = spl(xnew)
        
        # print(len_list)
        plt.plot(x, batch_auc_mean, color=color_dict[key], linewidth=1.5, label="B=" + key)

    # multiple line plot
    # plt.plot(auc_64, color='blue', linewidth=1, label="auc_64")
    # plt.plot(auc_128, color='green', linewidth=1, label="auc_128")
    # plt.plot(auc_256, color='olive', linewidth=1, label="auc_256")
    # plt.plot(auc_512, color='red', linewidth=1, label="auc_512")
    # plt.plot(auc_1024, color='black', linewidth=1, label="auc_1024")
    # xs = [i for i in range(400)]
    # horiz_line_data = np.array([0.846 for i in range(len(xs))])
    # plt.plot(xs, horiz_line_data, 'r--')
    # plt.axhline(y=0.845, color='r', linestyle='-')

    plt.xlabel('time (s)')
    plt.ylabel('AUC')
    plt.legend()
    # plt.title('Test AUC on various batch sizes')
    plt.show()
