from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census.train_census_dann import create_global_model_model
from scipy.stats import ks_2samp
import numpy as np


def test_classification(wrapper, data_loader):
    print("test_classification")
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
    acc = correct / n_total
    auc_0 = roc_auc_score(y_real_list, y_pred_list)
    auc_1 = roc_auc_score(y_real_list, y_pos_pred_prob_list)

    get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
    ks_test = get_ks(np.array(y_pos_pred_prob_list), np.array(y_real_list))
    print("[DEBUG]: {}/{}".format(correct, n_total))
    prf_bi = precision_recall_fscore_support(y_real_list, y_pred_list, average='binary')
    prf_mi = precision_recall_fscore_support(y_real_list, y_pred_list, average='micro')
    prf_ma = precision_recall_fscore_support(y_real_list, y_pred_list, average='macro')
    prf_we = precision_recall_fscore_support(y_real_list, y_pred_list, average='weighted')
    print(f"precision_recall_fscore:\n (binary: {prf_bi})\n (micro: {prf_mi})"
          f"\n (macro: {prf_ma})\n (weighted: {prf_we}))")
    print("roc_auc_score_0 : ", auc_0)
    print("roc_auc_score_1 : ", auc_1)
    print(f"ks_test : {ks_test}")
    print("accuracy : ", accuracy_score(y_real_list, y_pred_list))
    print("acc : ", acc)
    # TODO: save valid metrics
    return acc, auc_1


if __name__ == "__main__":
    wrapper = create_global_model_model()
    wrapper.load_model(root="census_target",
                       task_id="2020530_002",
                       task_meta_file_name="task_meta")

    wrapper.print_parameters()

    # census_adult_file_name = './datasets/census_processed/standardized_adult.csv'
    # census_95_file_name = './datasets/census_processed/sampled_standardized_census95.csv'

    # census_adult_test_file_name = './datasets/census_processed/standardized_adult_test.csv'
    census_95_test_file_name = '../../datasets/census_processed/sampled_standardized_census95_test.csv'

    print("[INFO] Load test data")
    _, census95_test_loader = get_income_census_dataloaders(
        ds_file_name=census_95_test_file_name, batch_size=128, split_ratio=0.1)

    acc, auc = test_classification(wrapper, census95_test_loader)
    print(f"acc: {acc}, auc: {auc}")
