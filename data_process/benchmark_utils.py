import numpy as np
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import json
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, roc_curve


def show_metrics(test_label, pred_label, pred_prob_label):
    print("test_label:", test_label.shape, type(test_label), sum(test_label))
    print("pred_label:", pred_label.shape, type(pred_label), sum(test_label))
    print("pred_prob_label:", pred_prob_label.shape, type(pred_prob_label), sum(pred_prob_label))
    get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
    ks = get_ks(np.array(pred_prob_label[:, 1]), np.array(test_label))

    print("test_label shape:", test_label.shape)
    print("pred_label shape:", pred_label.shape)
    acc = accuracy_score(test_label, pred_label)
    # auc1 = roc_auc_score(test_label, pred_label)
    auc = roc_auc_score(test_label, pred_prob_label[:, 1])
    print(f"accuracy : {acc}")
    print(f"auc : {auc}")
    print(f"ks: {ks}")
    return acc, auc, ks


def find_args_for_best_metric(result_list, model_name='xgb', metric_name='auc'):
    best_metric = 0.0
    best_arg = None
    for result in result_list:
        tmp_auc = result[1][model_name][metric_name]
        tmp_ks = result[1][model_name]["ks"]
        temp_score = (tmp_auc + tmp_ks) / 2
        if temp_score > best_metric:
            best_metric = temp_score
            best_arg = result[0]
    return best_metric, best_arg


def prepare_models(kwargs):
    n_tree_estimators = kwargs.get('n_tree_estimators')
    max_depth = kwargs.get('max_depth')
    n_tree_estimators = 200 if n_tree_estimators is None else n_tree_estimators
    max_depth = 4 if max_depth is None else max_depth
    num_leaves = 10
    lgb_params = {'objective': 'binary', 'metric': 'ks_metric', 'verbosity': -1, 'boosting_type': 'gbdt',
                  'n_estimators': n_tree_estimators, 'lambda_l1': 10, 'lambda_l2': 0.001, 'num_leaves': num_leaves,
                  'feature_fraction': 0.7, 'bagging_fraction': 0.72, 'bagging_freq': 3, 'max_depth': max_depth,
                  'min_child_samples': 10}

    # xgb_params = {'booster': 'gbtree', 'lambda': 0.01, 'alpha': 0.01, 'n_estimators': n_tree_estimators,
    #               'eta': 0.05, 'gamma': 0.34, 'grow_policy': 'lossguide', 'max_depth': max_depth,
    #               'feature_fraction': 0.7, 'bagging_fraction': 0.72, 'bagging_freq': 3, 'min_child_samples': 10}

    xgb_params = {'booster': 'gbtree', 'lambda': 0.01, 'alpha': 0.01, 'n_estimators': n_tree_estimators,
                  'max_depth': max_depth, 'eta': 0.05, 'gamma': 0.34, 'grow_policy': 'lossguide',
                  'feature_fraction': 0.7, 'subsample': 0.72}

    model_dict = {
        "lr": LogisticRegression(),
        # "lgbm": LGBMClassifier(**lgb_params),
        "xgb": xgb.XGBClassifier(**xgb_params)}

    return model_dict


def run_benchmark(train_data, train_label, test_data, test_label, **kwargs):
    result_dict = dict()
    for model_name, model in prepare_models(kwargs).items():
        print("-" * 100)
        print(f"==> model:{model}")
        # trained_model = model.fit(B_train_data, B_train_label)
        trained_model = model.fit(train_data, train_label)
        pred_label = trained_model.predict(test_data)
        pred_prob_label = trained_model.predict_proba(test_data)
        test_label = test_label.flatten()
        acc, auc, ks = show_metrics(test_label, pred_label, pred_prob_label)
        result_dict[model_name] = {"acc": acc, "auc": auc, "ks": ks}
    return result_dict


def save_benchmark_result(result, to_dir, file_name):
    file_full_name = os.path.join(to_dir, file_name + ".json")
    with open(file_full_name, 'w') as outfile:
        json.dump(result, outfile)
    print(f"[INFO] save benchmark result to {file_full_name}")
