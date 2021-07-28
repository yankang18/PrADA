from experiments.income_census.global_config import fine_tune_fg_dann_hyperparameters, data_hyperparameters
from experiments.income_census.train_census_fg_dann import create_fg_census_global_model
from experiments.income_census.train_census_utils import finetune_census_dann


def get_finetune_model_meta():
    finetune_target_root_dir = data_hyperparameters['census_fg_ft_target_model_dir']
    using_interaction = fine_tune_fg_dann_hyperparameters['using_interaction']
    model = create_fg_census_global_model(using_interaction=using_interaction)
    return model, finetune_target_root_dir


if __name__ == "__main__":
    dann_task_id = '20210728_census_fg_dann_all4000pos004v7_intrFalse_lr0.0006_bs128_me600_ts1627443028'
    # dann_task_id = '20210728_census_fg_dann_all4000pos004v6False_lr0.0006_bs128_me600_ts1627429903'
    census_pretain_model_root_dir = data_hyperparameters['census_fg_dann_model_dir']
    init_model, census_finetune_model_root_dir = get_finetune_model_meta()
    task_id = finetune_census_dann(dann_task_id,
                                   census_pretain_model_root_dir,
                                   census_finetune_model_root_dir,
                                   fine_tune_fg_dann_hyperparameters,
                                   data_hyperparameters,
                                   init_model)
    print(f"[INFO] task id:{task_id}")
