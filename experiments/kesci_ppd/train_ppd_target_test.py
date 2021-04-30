from datasets.ppd_dataloader import get_pdd_dataloaders_ob
import numpy as np
from experiments.kesci_ppd.train_ppd_dann import create_pdd_global_model

from utils import test_classification
import pandas as pd

if __name__ == "__main__":

    dann_root_folder = "ppd_dann"

    dann_task_id = "20200729_BCE_03_lr_00001_w_10"

    # Load initial models
    model = create_pdd_global_model()

    # load trained model
    model.load_model(root=dann_root_folder,
                     task_id=dann_task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    print("[INFO] Load test data")
    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    # target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'
    batch_size = 1024
    target_test_loader, _ = get_pdd_dataloaders_ob(ds_file_name=target_test_file_name, batch_size=batch_size,
                                                   split_ratio=1.0)

    acc, auc, ks = test_classification(model, target_test_loader)
    print(f"[INFO] acc:{acc}, auc:{auc}, ks:{ks}")

    sample_list = []
    for target_data, target_label in target_test_loader:
        feature = model.calculate_global_classifier_input_vector(target_data).numpy()
        sample = np.concatenate((feature, target_label), axis=1)
        sample_list.append(sample)
    classifier_data = np.concatenate(sample_list, axis=0)

    wide_col_names = ['user_info_1', 'user_info_2', 'user_info_3', 'user_info_4', 'user_info_5']
    fg_col_names = ['location', 'third_party', 'education', 'social_network', 'weblog_info']
    col_names = wide_col_names + fg_col_names + ['label']
    print("classifier_data:", classifier_data.shape)
    pd.DataFrame(data=classifier_data, columns=col_names).to_csv("path/to/file.csv")
