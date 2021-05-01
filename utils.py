import json
import os
from datetime import date
from datetime import datetime

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, accuracy_score


def get_latest_timestamp(timestamped_file_name, folder):
    timestamp_list = []
    for filename in os.listdir(folder):
        if filename.startswith(timestamped_file_name):
            maybe_timestamp = filename.split("_")[-1]
            if maybe_timestamp.endswith(".json"):
                timestamp = int(maybe_timestamp.split(".")[0])
            else:
                timestamp = int(maybe_timestamp)
            timestamp_list.append(timestamp)
    timestamp_list.sort()
    latest_timestamp = timestamp_list[-1]
    return latest_timestamp


def get_timestamp():
    return int(datetime.utcnow().timestamp())


def get_current_date():
    return date.today().strftime("%Y%m%d")


def save_dann_experiment_result(root, task_id, param_dict, metric_dict, timestamp):
    task_folder = "task_" + task_id
    task_root_folder = os.path.join(root, task_folder)
    if not os.path.exists(task_root_folder):
        os.makedirs(task_root_folder)

    result_dict = dict()
    result_dict["lr_param"] = param_dict
    result_dict["metrics"] = metric_dict

    file_name = "dann_exp_result_" + str(timestamp) + '.json'
    file_full_name = os.path.join(task_root_folder, file_name)
    with open(file_full_name, 'w') as outfile:
        json.dump(result_dict, outfile)


def load_dann_experiment_result(root, task_id, timestamp=None):
    task_folder = "task_" + task_id
    task_folder_path = os.path.join(root, task_folder)
    if not os.path.exists(task_folder_path):
        raise FileNotFoundError(f"{task_folder_path} is not found.")

    experiment_result = "dann_exp_result"
    if timestamp is None:
        timestamp = get_latest_timestamp(experiment_result, task_folder_path)
        print(f"[INFO] get latest timestamp {timestamp}")

    experiment_result_file_name = str(experiment_result) + "_" + str(timestamp) + '.json'
    experiment_result_file_path = os.path.join(task_folder_path, experiment_result_file_name)
    if not os.path.exists(experiment_result_file_path):
        raise FileNotFoundError(f"{experiment_result_file_path} is not found.")

    with open(experiment_result_file_path) as json_file:
        print(f"[INFO] load experiment result file from {experiment_result_file_path}")
        dann_exp_result_dict = json.load(json_file)
    return dann_exp_result_dict


def test_classification(wrapper, data_loader, tag):
    print(f"---------- {tag} classification ----------")
    correct = 0
    n_total = 0
    y_pred_list = []
    y_real_list = []
    y_pos_pred_prob_list = []
    wrapper.change_to_eval_mode()
    for batch_idx, (data, label) in enumerate(data_loader):
        label = label.flatten()
        n_total += len(label)
        batch_corr, y_pred, pos_y_prob = wrapper.calculate_classifier_correctness(data, label)
        correct += batch_corr
        y_real_list += label.tolist()
        y_pred_list += y_pred.tolist()
        y_pos_pred_prob_list += pos_y_prob.tolist()

        # print("y_pred_list:", y_real_list)
        # print("y_pred_list:", y_pred_list)
        # print("y_pos_pred_prob_list:", y_pos_pred_prob_list)

    acc = correct / n_total
    auc_0 = roc_auc_score(y_real_list, y_pred_list)
    auc_1 = roc_auc_score(y_real_list, y_pos_pred_prob_list)

    get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
    ks = get_ks(np.array(y_pos_pred_prob_list), np.array(y_real_list))
    print("[DEBUG]: {}/{}".format(correct, n_total))
    # prf_bi = precision_recall_fscore_support(y_real_list, y_pred_list, average='binary')
    # prf_mi = precision_recall_fscore_support(y_real_list, y_pred_list, average='micro')
    # prf_ma = precision_recall_fscore_support(y_real_list, y_pred_list, average='macro')
    # prf_we = precision_recall_fscore_support(y_real_list, y_pred_list, average='weighted')
    # print(f"precision_recall_fscore:\n (binary: {prf_bi})\n (micro: {prf_mi})"
    #       f"\n (macro: {prf_ma})\n (weighted: {prf_we}))")
    print("roc_auc_score_0 : ", auc_0)
    print("roc_auc_score_1 : ", auc_1)
    print("ks test : ", ks)
    print("accuracy : ", accuracy_score(y_real_list, y_pred_list))
    print("acc : ", acc)
    return acc, auc_1, ks


def test_discriminator(wrapper, num_regions, source_loader, target_loader):
    print("---------- test_discriminator ----------")
    source_correct = np.zeros(num_regions)
    target_correct = np.zeros(num_regions)
    n_source_total = 0
    n_target_total = 0

    # for batch_idx, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
    #     source_data, source_label = source_batch
    #     target_data, target_label = target_batch
    #     n_source_total += len(source_label)
    #     n_target_total += len(target_label)
    #
    #     src_corr_lst = wrapper.calculate_domain_discriminator_correctness(source_data, is_source=True)
    #     tgt_corr_lst = wrapper.calculate_domain_discriminator_correctness(target_data, is_source=False)
    #     source_correct += np.array(src_corr_lst)
    #     target_correct += np.array(tgt_corr_lst)

    for source_batch in source_loader:
        source_data, source_label = source_batch
        n_source_total += len(source_label)
        src_corr_lst = wrapper.calculate_domain_discriminator_correctness(source_data, is_source=True)
        source_correct += np.array(src_corr_lst)

    for target_batch in target_loader:
        target_data, target_label = target_batch
        n_target_total += len(target_label)
        tgt_corr_lst = wrapper.calculate_domain_discriminator_correctness(target_data, is_source=False)
        target_correct += np.array(tgt_corr_lst)

    total_acc = (source_correct + target_correct) / (n_source_total + n_target_total)
    source_acc = source_correct / n_source_total
    target_acc = target_correct / n_target_total
    cat_acc = np.concatenate((source_acc.reshape(1, -1), target_acc.reshape(1, -1)), axis=0)
    acc_sum = np.sum(cat_acc, axis=0)
    print(f"normalized domain acc:\n {cat_acc / acc_sum}")

    # overall_acc = (source_correct + target_correct)
    ave_total_acc = np.mean(total_acc)
    ave_source_acc = np.mean(source_acc)
    ave_target_acc = np.mean(target_acc)
    print(f"[DEBUG] {n_source_total} source domain acc: {source_acc}, mean: {ave_source_acc}")
    print(f"[DEBUG] {n_target_total} target domain acc: {target_acc}, mean: {ave_target_acc}")
    print(f"[DEBUG] total domain acc: {total_acc}, mean: {ave_total_acc}")
    # entropy_domain_acc = entropy(cat_acc / acc_sum)
    # print(f"[DEBUG] domain acc entropy: {entropy_domain_acc}")
    return (ave_total_acc, ave_source_acc, ave_target_acc), (list(total_acc), list(source_acc), list(target_acc))


def entropy(predictions):
    epsilon = 1e-6
    H = -predictions * np.log2(predictions + epsilon)
    return H.sum(axis=0)
