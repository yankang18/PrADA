import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    dir = "../../../data/cell_manager/"
    B_train = pd.read_csv(dir + "B_train.csv")
    B_test = pd.read_csv(dir + "B_test.csv")

    B_train_Values = shuffle(B_train.values)
    B_test_values = shuffle(B_test.values)
    train_data, train_label = B_train_Values[:, 3:], B_train_Values[:, 1]
    print(f"train_data shape:{train_data.shape}")
    print(f"train_label shape:{train_label.shape}, {train_label}, {np.sum(train_label)}")
    test_data, test_label = B_test_values[:, 3:], B_test_values[:, 1]
    print(f"test_data shape:{test_data.shape}")
    print(f"test_label shape:{test_label.shape}, {test_label}, {np.sum(test_label)}")

    # lr = xgb.XGBClassifier()
    # param_dist = {"max_depth": [10, 30, 50],
    #               "min_child_weight": [1, 3, 6],
    #               "n_estimators": [200],
    #               "learning_rate": [0.05, 0.1, 0.16], }
    # grid_search = GridSearchCV(lr, param_grid=param_dist, cv=3,
    #                            verbose=10, n_jobs=-1)
    # grid_search.fit(train_data, train_label)
    # pred_label = grid_search.predict(test_data)

    xgb_params = {'booster': 'gbtree', 'lambda': 10.0, 'alpha': 0.001, 'num_leaves': 3, 'n_estimators': 137,
                  'eta': 0.03, 'gamma': 0.34, 'grow_policy': 'lossguide',
                  'feature_fraction': 0.7, 'bagging_fraction': 0.72, 'bagging_freq': 3, 'min_child_samples': 20}
    lr = xgb.XGBClassifier(**xgb_params)
    lr = lr.fit(train_data, train_label)
    pred_label = lr.predict(test_data)
    # predprob_label = lr.predict_proba(test_data)
    # print("predprob_label:", predprob_label[:, 1])

    test_label = test_label.astype(float).ravel()
    # print("pred:", pred, pred.shape, type(pred), sum(pred))
    print("test_label:", test_label, test_label.shape, type(test_label), sum(test_label))
    print("pred_label:", pred_label, pred_label.shape, type(pred_label), sum(pred_label))
    # pred_prob = cls.predict_proba(test_data)
    print("test_label shape:", test_label.shape)
    print("pred_label shape:", pred_label.shape)
    acc = accuracy_score(test_label, pred_label)
    # auc1 = roc_auc_score(predprob_label[:, 1], test_label)
    auc2 = roc_auc_score(test_label, pred_label)
    res = precision_recall_fscore_support(pred_label, test_label, average='weighted')
    print(f"accuracy : {acc}")
    # print(f"auc1 : {auc1}")
    print(f"auc2 : {auc2}")
    print(f"prf : {res}")
    # print(f"coef: {cls.coef_}")
