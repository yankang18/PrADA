import numpy as np
from sklearn.utils import shuffle
from data_process.cell_process.bechmark import get_datasets, run_benchmark

# table_names = ["p_wide_col.csv", "p_debt_feat.csv", "p_payment_feat.csv",
#                "p_payment_debt_cross_feat.csv", "p_multi_acc_feat.csv", "p_mal_behavior_feat.csv",
#                "p_qualify_feat.csv", "p_loan_feat.csv", "target.csv"]

table_names = ["p_wide_col.csv", "p_debt_feat.csv", "p_payment_feat.csv",
               "p_payment_debt_cross_feat.csv", "p_multi_acc_feat.csv", "p_mal_behavior_feat.csv",
               "p_qualify_feat.csv", "target.csv"]


def prepare_train_data(train_data, train_label):
    train_label_array = train_label.ravel()
    pos_train_data = train_data[train_label_array == 1]
    pos_train_label = train_label[train_label_array == 1]
    neg_train_data = train_data[train_label_array == 0]
    neg_train_label = train_label[train_label_array == 0]

    print(f"pos_train_data shape:{pos_train_data.shape}")
    print(f"neg_train_data shape:{neg_train_data.shape}")

    pos_train_data = pos_train_data[:2970]
    pos_train_label = pos_train_label[:2970]
    neg_train_data = neg_train_data[:30]
    neg_train_label = neg_train_label[:30]
    _train_data = np.concatenate([pos_train_data, neg_train_data], axis=0)
    _train_label = np.concatenate([pos_train_label, neg_train_label], axis=0)
    return _train_data, _train_label


if __name__ == "__main__":
    # A_dir = "../../../data/lending_club_bundle_archive/loan_processed_2015_18/"
    # B_dir = "../../../data/lending_club_bundle_archive/loan_processed_2018/"

    # A_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_17/"
    # A_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_16/"
    A_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2016_17/"
    B_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"

    A_train_data, A_train_label = get_datasets(A_dir, table_names, "train")
    B_train_data, B_train_label = get_datasets(B_dir, table_names, data_mode="train", is_shuffle=True)
    # B_train_data, B_train_label = get_datasets(B_dir, table_names, "train_nus")

    train_data = np.concatenate([A_train_data, B_train_data], axis=0)
    train_label = np.concatenate([A_train_label, B_train_label], axis=0)

    print(f"B_train_data shape:{B_train_data.shape}")
    print(f"B_train_label shape:{B_train_label.shape}， {np.sum(B_train_label)}")

    # train_data = A_train_data
    # train_label = A_train_label
    # train_data = B_train_data
    # train_label = B_train_label

    print(f"train_data shape:{train_data.shape}")
    print(f"train_label shape:{train_label.shape}， {np.sum(train_label)}")

    # train_data, train_label = prepare_train_data(train_data, train_label)
    train_data, train_label = shuffle(train_data, train_label)
    # train_data, train_label = train_data[:10000], train_label[:10000]
    train_label = train_label.ravel()
    # print(f"A_train_data shape:{A_train_data.shape}")
    # print(f"A_train_label shape:{A_train_label.shape}， {np.sum(A_train_label)}")

    B_test_data, B_test_label = get_datasets(B_dir, table_names, "test")

    # pred = tr.tradaboost(X_train, trans_T, y_train, label_T, X_test, 10)
    # fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=pred, pos_label=1)
    # print('auc:', metrics.auc(fpr, tpr))

    run_benchmark(train_data, train_label, B_test_data, B_test_label)
