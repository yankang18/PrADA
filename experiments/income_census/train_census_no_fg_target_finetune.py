import argparse

from experiments.income_census.global_config import fine_tune_hyperparameters, data_hyperparameters
from experiments.income_census.train_census_no_fg_adaptation import create_no_fg_census_global_model
from experiments.income_census.train_census_utils import finetune_census


def get_finetune_model_meta():
    finetune_target_root_dir = data_hyperparameters['census_no-fg_ft_target_model_dir']
    model = create_no_fg_census_global_model()
    return model, finetune_target_root_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser("census_no-fg_target_fine_tune")
    parser.add_argument('--adapt_task_id', type=str, help='task id of train_census_fg_dann')
    args = parser.parse_args()
    adaptation_task_id = args.task_id
    print(f"[INFO] fine-tune pre-trained model with adaptation task id : {adaptation_task_id}")

    census_pretain_model_root_dir = data_hyperparameters['census_no-fg_dann_model_dir']
    init_model, census_finetune_model_root_dir = get_finetune_model_meta()
    task_id = finetune_census(adaptation_task_id,
                              census_pretain_model_root_dir,
                              census_finetune_model_root_dir,
                              fine_tune_hyperparameters,
                              data_hyperparameters,
                              init_model)
    print(f"[INFO] task id:{task_id}")
