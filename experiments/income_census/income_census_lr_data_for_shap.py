from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census.train_census_fg_dann import create_global_model
from utils import test_classification, produce_data_for_lr_shap

if __name__ == "__main__":
    dann_root_folder = "census_dann"
    batch_size = 1024
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    target_test_file_name = data_dir + 'grad_census9495_da_test.csv'

    feature_group_cols = ['employment', 'demographics', 'residence', 'household', 'origin']
    continuous_cols = ["age", "gender", "capital_gain", "capital_loss", "stock_dividends"]
    columns_list = continuous_cols + feature_group_cols + ['label']
    print("[INFO] columns_list:", len(columns_list), columns_list)
    lr_data_dir = "/Users/yankang/Documents/Data/census/lr_shap_data/"

    # dann_task_id = '20210506_DEGREE_0.0005_64_1620240228'
    # dann_task_id_list = ["20210506_DEGREE_0.0005_64_1620275075"]
    dann_task_id_list = ["20210507_DEGREE_0.0005_64_1620288753"]

    # Initialize models
    model = create_global_model(pos_class_weight=1.0)

    for task_id in dann_task_id_list:
        print(f'------ [INFO] create lr-data-shap for {dann_task_id_list} ------')

        print("[INFO] Load pre_trained data")
        model.load_model(root=dann_root_folder,
                         task_id=task_id,
                         load_global_classifier=True,
                         timestamp=None)
        model.print_parameters()

        print("[INFO] Load test data")
        target_test_loader, _ = get_income_census_dataloaders(ds_file_name=target_test_file_name,
                                                              batch_size=batch_size,
                                                              split_ratio=1.0)

        print("[INFO] Run test")
        acc, auc, ks = test_classification(model, target_test_loader, "test")
        print(f"[INFO] acc:{acc}, auc:{auc}, ks:{ks}")

        print("[INFO] Produce LR data for SHAP")
        output_file_full_name = f"{lr_data_dir}income_lr_data_{task_id}.csv"
        produce_data_for_lr_shap(model, target_test_loader, columns_list, output_file_full_name)
