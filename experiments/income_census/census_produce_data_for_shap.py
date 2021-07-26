from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census.train_census_fg_dann import create_fg_census_global_model
from utils import test_classifier, produce_data_for_lr_shap

if __name__ == "__main__":
    dann_root_folder = "census_dann"
    batch_size = 1024
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    # data_file_name = data_dir + 'grad_census9495_da_test.csv'
    # data_file_name = data_dir + 'grad_census9495_da_train.csv'
    # target_train_file_name = data_dir + 'undergrad_census9495_da_train.csv'
    # target_adult_train_file_name = 'grad_census9495_da_300_train.csv'
    # target_adult_test_file_name = 'grad_census9495_da_300_test.csv'

    using_interaction = True
    # data_tag = "all4000pos001"
    # data_tag = "all4000pos001v2"
    # data_tag = "all4000pos002"
    data_tag = "all4000pos004v4"
    target_adult_train_file_name = f'grad_census9495_da_{data_tag}_train.csv'
    source_adult_test_file_name = f'undergrad_census9495_da_{data_tag}_test.csv'
    target_adult_test_file_name = f'grad_census9495_ft_{data_tag}_test.csv'

    # continuous_cols = ["age", "gender", "capital_gain", "capital_loss", "stock_dividends"]
    continuous_cols = ["age", "gender", "education_year", "capital_gain", "capital_loss"]
    feature_group_cols = ['employment', 'demographics', 'migration', 'household']
    intr_feat_group_cols = ['emp-demo', 'emp-mig', 'emp-house', 'demo-mig', 'demo-house', 'mig-house']
    if using_interaction:
        columns_list = continuous_cols + feature_group_cols + intr_feat_group_cols + ['label']
    else:
        columns_list = continuous_cols + feature_group_cols + ['label']
    print("[INFO] columns_list:", len(columns_list), columns_list)
    data_output_dir = "/Users/yankang/Documents/Data/census/lr_shap_data/"
    # data_output_dir = "/Users/yankang/Documents/Data/census/target_finetune_data/"

    original_data_file_list = [target_adult_train_file_name, target_adult_test_file_name]
    # dann_task_id_list = ['20210706_src_lr0.0006_bs128_pw1.0_me600_ts1625473672']
    # dann_task_id_list = ['20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625385790']
    # dann_task_id_list = ['20210712_src_lr0.0006_bs128_pw1.0_me600_ts1626060178']
    # dann_task_id_list = ['20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625426396']
    dann_task_id_list = ['20210705_src_lr0.0006_bs128_pw1.0_me600_ts1625462494']
    # Initialize models
    model = create_fg_census_global_model(pos_class_weight=1.0, using_interaction=using_interaction)
    for task_id in dann_task_id_list:
        print(f'------ [INFO] create data for {task_id} ------')

        print("[INFO] Load pre_trained data")
        model.load_model(root=dann_root_folder,
                         task_id=task_id,
                         load_global_classifier=True,
                         timestamp=None)
        model.print_parameters()

        for data_file_full_name in original_data_file_list:
            print("[INFO] Load test data")
            data_loader, _ = get_income_census_dataloaders(ds_file_name=data_dir + data_file_full_name,
                                                           batch_size=batch_size,
                                                           split_ratio=1.0)

            data_file_name = data_file_full_name.split(".")[0]
            output_file_full_name = f"{data_output_dir}income_data-{task_id}-{data_file_name}.csv"
            produce_data_for_lr_shap(model, data_loader, columns_list, output_file_full_name)
