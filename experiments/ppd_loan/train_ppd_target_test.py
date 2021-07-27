import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from datasets.ppd_dataloader import get_pdd_dataloaders_ob
from experiments.ppd_loan.train_ppd_fg_dann import create_no_fg_pdd_global_model
from utils import test_classifier


def normalize(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled


def partial_normalize(df, no_norm_cols, to_norm_cols):
    no_norm_values = df[no_norm_cols].values
    to_norm_values = df[to_norm_cols].values
    norm_values = normalize(to_norm_values)
    values = np.concatenate([no_norm_values, norm_values], axis=1)
    cols = no_norm_cols + to_norm_cols
    return pd.DataFrame(data=values, columns=cols)


# if __name__ == "__main__":
#
#     dann_root_folder = "ppd_dann"
#
#     dann_task_id = "20210430_PDD_pw1.0_bs256_lr0.0012_v0_t1619747346"
#
#     # Load initial models
#     model = create_pdd_global_model()
#
#     # load trained model
#     model.load_model(root=dann_root_folder,
#                      task_id=dann_task_id,
#                      load_global_classifier=True,
#                      timestamp=None)
#
#     print("[DEBUG] Global classifier Model Parameter Before train:")
#     model.print_parameters()
#
#     # Load data
#     print("[INFO] Load test data")
#     data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
#     output_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/output/"
#     # target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
#     target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'
#     batch_size = 1024
#     target_test_loader, _ = get_pdd_dataloaders_ob(ds_file_name=target_test_file_name, batch_size=batch_size,
#                                                    split_ratio=1.0)
#
#     acc, auc, ks = test_classification(model, target_test_loader, "test")
#     print(f"[INFO] acc:{acc}, auc:{auc}, ks:{ks}")


def produce_data_for_lr_shap(model, data_loader, column_name_list, output_dir):
    """
    produce data for LR SHAP
    """

    sample_list = []
    for data, label in data_loader:
        feature = model.calculate_global_classifier_input_vector(data).detach().numpy()
        label = label.numpy().reshape((-1, 1))
        # print(feature.shape, label.shape)
        sample = np.concatenate((feature, label), axis=1)
        sample_list.append(sample)
    classifier_data = np.concatenate(sample_list, axis=0)

    # wide_col_names = ['user_info_1', 'user_info_2', 'user_info_3', 'user_info_4', 'user_info_5']
    # fg_col_names = ['location', 'third_party', 'education', 'social_network', 'weblog_info']
    # no_norm_cols = ['user_info_1', 'user_info_2', 'user_info_3', 'user_info_4'] + fg_col_names
    # col_names = wide_col_names + fg_col_names + ['label']
    print("[INFO] global classifier input data with shape:{}".format(classifier_data.shape))
    df_lr_input = pd.DataFrame(data=classifier_data, columns=column_name_list)
    # df_lr_input = partial_normalize(df_lr_input, to_norm_cols=['user_info_5'], no_norm_cols=no_norm_cols)
    df_lr_input.to_csv(output_dir + "lr_input.csv")
