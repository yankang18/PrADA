from experiments.income_census.global_config import fine_tune_fg_dann_hyperparameters, data_hyperparameters
from experiments.income_census.train_census_fg_dann import create_fg_census_global_model
from experiments.income_census.train_census_utils import finetune_census_dann

if __name__ == "__main__":
    dann_task_id = ''
    census_pretain_model_root_dir = data_hyperparameters['census_no-fg_dann_model_dir']
    census_finetune_target_root_dir = data_hyperparameters['census_no-fg_ft_target_model_dir']
    finetune_census_dann(dann_task_id,
                         census_pretain_model_root_dir,
                         census_finetune_target_root_dir,
                         fine_tune_fg_dann_hyperparameters,
                         data_hyperparameters,
                         create_census_global_model_func=create_fg_census_global_model)
