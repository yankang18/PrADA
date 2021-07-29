from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census import train_census_fg_target_finetune as fg_finetune
from experiments.income_census import train_census_no_adaptation as no_ad_finetune
from experiments.income_census import train_census_no_fg_target_finetune as no_fg_finetune
from experiments.income_census.global_config import data_hyperparameters
from experiments.income_census.test_config import census_target_test_config
from utils import test_classifier


def test_model(task_id, init_model, trained_model_root_folder, target_test_file_name):
    # load trained model
    print("[INFO] load trained model")
    init_model.load_model(root=trained_model_root_folder,
                          task_id=task_id,
                          load_global_classifier=True,
                          timestamp=None)

    init_model.print_parameters()

    print("[INFO] load test data")
    batch_size = 1024
    target_test_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Run test")
    acc, auc, ks = test_classifier(init_model, target_test_loader, "test")
    print(f"[INFO] test acc:{acc}, auc:{auc}, ks:{ks}")


if __name__ == "__main__":
    task_id = census_target_test_config['task_id']
    test_tag = census_target_test_config['test_task_tag']
    print(f"[INFO] perform test task : [{test_tag}] with id: {task_id}")
    test_models_dir = {"fg_target": fg_finetune.get_finetune_model_meta,
                       "no_fg_target": no_fg_finetune.get_finetune_model_meta,
                       "no_ad_target": no_ad_finetune.get_model_meta}
    init_model, model_root_dir = test_models_dir[test_tag]()
    target_test_file_name = data_hyperparameters['target_ft_test_file_name']
    print(f"[INFO] target_test_file_name: {target_test_file_name}.")
    test_model(task_id, init_model, model_root_dir, target_test_file_name)
