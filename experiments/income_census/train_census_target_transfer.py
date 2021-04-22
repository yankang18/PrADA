from datasets.census_dataloader import get_income_census_dataloaders
from models.experiment_target_learner import FederatedTargetLearner
from experiments.income_census.train_census_dann import create_global_model_model
from experiments.income_census.train_census_target_test import test_classification

if __name__ == "__main__":
    dann_root_folder = "census_dann"
    # dann_task_id = "20200910_BCE_07_lr_003_w_7"
    # dann_task_id = '20201215_DEGREE_0.008_64_5'
    # dann_task_id = '20201215_DEGREE_0.008_64_6'
    # dann_task_id = '20201215_DEGREE_0.008_64_4'
    # dann_task_id = '20201215_DEGREE_0.008_64_7'
    # dann_task_id = "20201217_DEGREE_0.005_64_5"
    # dann_task_id = "20201218_DEGREE_0.008_64_2"
    # dann_task_id = "20201218_DEGREE_0.008_64_2"
    # dann_task_id = "20201218_DEGREE_0.008_64_1"
    dann_task_id = "20210418_DEGREE_0.0008_64_5"

    # Load models
    wrapper = create_global_model_model()

    # load pre-trained model
    load_global_classifier = True
    wrapper.load_model(root=dann_root_folder, task_id=dann_task_id, load_global_classifier=load_global_classifier,
                       timestamp=None)
    # dann_exp_result = load_dann_experiment_result(root=dann_root_folder, task_id=dann_task_id)
    dann_exp_result = None

    print("[DEBUG] Global classifier Model Parameter Before train:")
    wrapper.print_parameters()

    # Load data

    # # census_95_file_name = './datasets/census_processed/standardized_census95_train_1000.csv'
    # # census_95_file_name = './datasets/census_processed/sampled_standardized_census95.csv'
    # census_95_file_name = './datasets/census_processed/standardized_census95_benchmark_train_9768.csv'
    # census_95_test_file_name = 'datasets/census_processed/sampled_standardized_census95_test.csv'

    # target_train_file_name = '../../datasets/census_processed/degree_target_train.csv'
    # target_test_file_name = '../../datasets/census_processed/degree_target_test.csv'

    # target_adult_train_file_name = '../../datasets/census_processed/adult_target_train.csv'
    # target_adult_test_file_name = '../../datasets/census_processed/adult_target_test.csv'

    target_train_file_name = '../../datasets/census_processed/grad_census9495_da_train.csv'
    target_test_file_name = '../../datasets/census_processed/grad_census9495_da_test.csv'

    batch_size = 64; lr = 3e-4;  version = 1
    # batch_size = 128; lr = 8e-4;  version = 1
    print("[INFO] Load train data")
    target_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    # perform target training

    plat_target = FederatedTargetLearner(wrapper=wrapper,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=400)
    plat_target.set_model_save_info("census_target")

    appendix = "_" + str(batch_size) + "_" + str(lr) + "_v" + str(version)
    target_task_id = dann_task_id + "_target_finetune" + appendix
    plat_target.train_target_with_alternating(global_epochs=400, top_epochs=1, bottom_epochs=1, lr=lr,
                                              task_id=target_task_id, dann_exp_result=dann_exp_result)
    # plat_target.train_target_as_whole(global_epochs=100, lr=4e-4, task_id=target_task_id,
    #                                   dann_exp_result=dann_exp_result)

    print("[DEBUG] Global classifier Model Parameter After train:")
    wrapper.print_parameters()

    acc, auc = test_classification(wrapper, target_valid_loader)
    print(f"acc:{acc}, auc:{auc}")
