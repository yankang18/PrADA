from datasets.ppd_dataloader import get_pdd_dataloaders_ob
from experiments.kesci_ppd.train_ppd_fg_dann import create_pdd_global_model
from utils import produce_data_for_lr_shap
from utils import test_classifier

if __name__ == "__main__":
    dann_root_folder = "ppd_dann"

    task_id = "20210506_PDD_pw1.0_bs256_lr0.0012_v0_t1620275086"

    # Load initial models
    model = create_pdd_global_model()

    # load trained model
    model.load_model(root=dann_root_folder,
                     task_id=task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    print("[INFO] Load test data")
    # data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    # output_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/output/"
    # target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    timestamp = '1620085151'
    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{timestamp}/"
    target_test_file_name = data_dir + 'PPD_2014_tgt_10to11_train.csv'
    batch_size = 1024
    target_test_loader, _ = get_pdd_dataloaders_ob(ds_file_name=target_test_file_name, batch_size=batch_size,
                                                   split_ratio=1.0)

    acc, auc, ks = test_classifier(model, target_test_loader, "test")
    print(f"[INFO] acc:{acc}, auc:{auc}, ks:{ks}")

    print("[INFO] Produce LR data for SHAP")
    wide_col_names = ['user_info_1', 'user_info_2', 'user_info_3', 'user_info_4', 'user_info_5', 'user_info_6']
    fg_col_names = ['location', 'third_party', 'education', 'social_network', 'weblog_info']
    columns_list = wide_col_names + fg_col_names + ['label']
    lr_data_dir = "/Users/yankang/Documents/Data/census/lr_shap_data/"
    output_file_full_name = f"{lr_data_dir}income_lr_data_{task_id}.csv"
    produce_data_for_lr_shap(model, target_test_loader, columns_list, output_file_full_name)
