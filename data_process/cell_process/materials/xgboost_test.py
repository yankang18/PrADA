from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from scipy.stats import ks_2samp
# from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
import gc
import pandas as pd
import numpy as np

params = {'class_weight': 'balanced', 'max_iter': 1000, 'random_state': 1, 'solver': 'lbfgs'}

lgb_params = {'objective': 'binary', 'metric': 'ks_metric', 'verbosity': -1, 'boosting_type': 'gbdt',
              'n_estimators': 500,
              'lambda_l1': 10, 'lambda_l2': 0.001, 'num_leaves': 3,
              'feature_fraction': 0.7, 'bagging_fraction': 0.72, 'bagging_freq': 3, 'min_child_samples': 20}
# lgb_params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt', 'n_estimators': 1000,
#               'lambda_l1': 5.0231336288138335, 'lambda_l2': 0.3791821254074195, 'num_leaves': 3,
#               'feature_fraction': 0.52, 'bagging_fraction': 0.8135390754746483, 'bagging_freq': 1, 'min_child_samples': 50}
# lgb_params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt', 'n_estimators': 1000,
#               'lambda_l1': 9.417801675755515, 'lambda_l2': 5.262426768247271, 'num_leaves': 11,
#               'feature_fraction': 0.4, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 5}
xgb_params = {'booster': 'gbtree', 'lambda': 1.5349117684824456e-07, 'alpha': 1.3364573136609547e-06, 'max_depth': 3,
              'n_estimators': 2000,
              'eta': 0.035553606749303385, 'gamma': 0.3379899942189756, 'grow_policy': 'lossguide'}


# xgb_params = {'booster': 'dart', 'lambda': 0.00023571610444678062, 'alpha': 0.002097145076109315, 'max_depth': 2,
#               'eta': 0.47909317842376276, 'gamma': 0.06072682282464417, 'grow_policy': 'depthwise', 'n_estimators': 1000,
#               'sample_type': 'uniform', 'normalize_type': 'tree', 'rate_drop': 4.7243424518296774e-07, 'skip_drop': 0.7613259880585317}

def ks_metric(labels, preds):
    get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
    score = get_ks(preds, labels)
    return 'ks_metric', score, True


# help_df = pd.DataFrame(pd.read_csv("A_train.csv"))
dir = "../../../data/cell_manager/"
train_df = pd.read_csv(dir + "B_train.csv")
test_df = pd.read_csv(dir + "B_test.csv")

print("B train:", train_df.shape)
print("B test:", test_df.shape)

gc.collect()

X_train = train_df.iloc[:, 3:]
Y_train = train_df.loc[:, 'target']
X_test = test_df.iloc[:, 3:]
Y_test = test_df.loc[:, 'target']

#     scaler = StandardScaler().fit(shouguan_fixed_df)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
#     X_train = np.nan_to_num(X_train)
#     X_test = np.nan_to_num(X_test)

print(X_train.shape)
# print(Y_train)

# clf = LogisticRegression(**params)
# clf = SVC(C=1, kernel='rbf', probability=True)
# clf = DecisionTreeClassifier(criterion='entropy',max_features='log2')
# clf = RandomForestClassifier()
# clf = xgb.XGBClassifier(**xgb_params)
# clf = AdaBoostClassifier()
# clf = BaggingClassifier(LogisticRegression())
# clf = GradientBoostingClassifier(**params)

clf = LGBMClassifier(**lgb_params)

# clf = GradientBoostingClassifier()
# clf = MLPClassifier(activation='relu', max_iter=50, hidden_layer_sizes=(64, 64), verbose=True, early_stopping=True)
# clf = GaussianNB()
clf.fit(X_train, Y_train, eval_metric='auc', early_stopping_rounds=100, eval_set=[(X_test, Y_test), (X_train, Y_train)])
# clf.fit(X_train,Y_train)
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
y_test_pred_proba = clf.predict_proba(X_test)
y_train_pred_proba = clf.predict_proba(X_train)
print("y_test_pred:", y_test_pred)
print("y_test_pred_proba:", y_test_pred_proba)
# f1 = f1_score(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

roc_auc_test_0 = roc_auc_score(Y_test, y_test_pred)
acc_test = accuracy_score(Y_test, y_test_pred)
roc_auc_test_1 = roc_auc_score(Y_test, y_test_pred_proba[:, 1])
roc_auc_train = roc_auc_score(Y_train, y_train_pred_proba[:, 1])

get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
ks_train = get_ks(y_train_pred_proba[:, 1], Y_train)
ks_test = get_ks(y_test_pred_proba[:, 1], Y_test)
print('roc_auc_train =', roc_auc_train)
print('ks_train =', ks_train)
print("acc_test = ", acc_test)
print('roc_auc_test_0 =', roc_auc_test_0)
print('roc_auc_test_1 =', roc_auc_test_1)
print('ks_test =', ks_test)

# feature_importance_df = pd.DataFrame({
#     'column': X_train.columns,
#     'importance': clf.booster_.feature_importance(importance_type='gain'),
# }).sort_values(by='importance', ascending=False)
#
# feature_importance_df.to_csv('feature_importance.csv', index=False)
