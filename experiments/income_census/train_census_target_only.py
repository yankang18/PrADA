from datasets.census_dataloader import get_income_census_dataloaders
from models.experiment_target_learner import FederatedTargetLearner
from experiments.income_census.train_census_fg_dann import create_global_model
from experiments.income_census.train_census_target_test import test_classification

if __name__ == "__main__":
    # dann_root_folder = "census_dann"
    # dann_task_id = "2020530_001"

    # Load transferred models
    wrapper = create_global_model()

    print("[DEBUG] Global classifier Model Parameter Before train:")
    wrapper.print_parameters()

    # Load data
    census_95_file_name = '../../datasets/census_processed/standardized_census95_benchmark_train_9768.csv'
    census_95_test_file_name = '../../datasets/census_processed/sampled_standardized_census95_test.csv'

    batch_size = 64
    print("[INFO] Load train data")
    census95_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=census_95_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    census95_valid_loader, census95_test_loader = get_income_census_dataloaders(
        ds_file_name=census_95_test_file_name, batch_size=batch_size, split_ratio=0.5)

    # perform target training
    target_task_id = "2020530_002"
    plat_target = FederatedTargetLearner(model=wrapper,
                                         target_train_loader=census95_train_loader,
                                         target_val_loader=census95_valid_loader,
                                         patience=150)

    # plat_target.train_target_with_alternating(global_epochs=100,
    #                                           top_epochs=1,
    #                                           bottom_epochs=1,
    #                                           lr=1e-2,
    #                                           task_id=target_task_id)
    plat_target.train_target_as_whole(epochs=100,
                                      lr=1e-2,
                                      task_id=target_task_id)

    print("[DEBUG] Global classifier Model Parameter After train:")
    wrapper.print_parameters()

    acc, auc = test_classification(wrapper, census95_test_loader)
    print(f"acc:{acc}, auc:{auc}")
