import numpy as np
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def show_metrics(test_label, pred_label, pred_prob_label):
    print("test_label:", test_label.shape, type(test_label), sum(test_label))
    print("pred_label:", pred_label.shape, type(pred_label), sum(test_label))
    print("pred_prob_label:", pred_prob_label.shape, type(pred_prob_label), sum(pred_prob_label))
    get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
    ks_test = get_ks(np.array(pred_prob_label[:, 1]), np.array(test_label))

    print("test_label shape:", test_label.shape)
    print("pred_label shape:", pred_label.shape)
    acc = accuracy_score(test_label, pred_label)
    # auc1 = roc_auc_score(test_label, pred_label)
    auc = roc_auc_score(test_label, pred_prob_label[:, 1])
    print(f"accuracy : {acc}")
    print(f"ks_test : {ks_test}")
    # print(f"auc1 : {auc1}")
    print(f"auc : {auc}")


def prepare_models(kwargs):
    model_list = list()

    n_tree_estimators = kwargs.get('n_tree_estimators')
    max_depth = kwargs.get('max_depth')
    n_tree_estimators = 200 if n_tree_estimators is None else n_tree_estimators
    max_depth = 4 if max_depth is None else max_depth
    num_leaves = 10
    lgb_params = {'objective': 'binary', 'metric': 'ks_metric', 'verbosity': -1, 'boosting_type': 'gbdt',
                  'n_estimators': n_tree_estimators, 'lambda_l1': 10, 'lambda_l2': 0.001, 'num_leaves': num_leaves,
                  'feature_fraction': 0.7, 'bagging_fraction': 0.72, 'bagging_freq': 3, 'max_depth': max_depth,
                  'min_child_samples': 10}

    xgb_params = {'booster': 'gbtree', 'lambda': 10.0, 'alpha': 0.001, 'n_estimators': n_tree_estimators,
                  'eta': 0.03, 'gamma': 0.34, 'grow_policy': 'lossguide', 'max_depth': max_depth,
                  'feature_fraction': 0.7, 'bagging_fraction': 0.72, 'bagging_freq': 3, 'min_child_samples': 20}

    model_list.append(LogisticRegression(max_iter=500, solver="sag"))
    model_list.append(LGBMClassifier(**lgb_params))
    model_list.append(xgb.XGBClassifier(**xgb_params))

    return model_list


def run_benchmark(train_data, train_label, test_data, test_label, **kwargs):
    print(f"[INFO] train_data shape:{train_data.shape}")
    print(f"[INFO] train_label shape:{train_label.shape}， {np.sum(train_label)}")
    print(f"[INFO] test_data shape:{test_data.shape}")
    print(f"[INFO] test_label shape:{test_label.shape}， {np.sum(test_label)}")

    for model in prepare_models(kwargs):
        print("-" * 100)
        print(f"==> model:{model}")
        # trained_model = model.fit(B_train_data, B_train_label)
        trained_model = model.fit(train_data, train_label)
        B_pred_label = trained_model.predict(test_data)
        B_pred_prob_label = trained_model.predict_proba(test_data)
        test_label = test_label.flatten()
        show_metrics(test_label, B_pred_label, B_pred_prob_label)
