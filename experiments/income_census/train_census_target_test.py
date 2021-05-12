from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census.train_census_fg_dann import create_global_model
from utils import test_classification

if __name__ == "__main__":
    dann_root_folder = "census_dann"

    # dann_task_id = '20210505_DEGREE_0.0005_64_1620156015'
    # dann_task_id = '20210506_DEGREE_0.0005_64_1620240228'
    dann_task_id = '20210506_DEGREE_0.0005_64_1620253871'

    # Load models
    model = create_global_model(pos_class_weight=1.0)

    # load pre-trained model
    print("[INFO] Load pre_trained data")
    load_global_classifier = False
    model.load_model(root=dann_root_folder,
                     task_id=dann_task_id,
                     load_global_classifier=True,
                     timestamp=None)

    model.print_parameters()

    print("[INFO] Load test data")
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    target_test_file_name = data_dir + 'grad_census9495_da_test.csv'

    batch_size = 1024
    target_test_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Run test")
    acc, auc, ks = test_classification(model, target_test_loader, "test")
    print(f"[INFO] test acc:{acc}, auc:{auc}, ks:{ks}")
