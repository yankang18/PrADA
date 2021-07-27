from experiments.kesci_ppd.global_config import data_hyperparameters, fine_tune_fg_dann_hyperparameters
from experiments.kesci_ppd.train_ppd_fg_dann import create_no_fg_pdd_global_model
from experiments.kesci_ppd.train_ppd_utils import finetune_ppd_dann

if __name__ == "__main__":
    dann_task_id = ''
    ppd_pretain_model_root_dir = data_hyperparameters['ppd_no-fg_dann_model_dir']
    ppd_finetune_target_root_dir = data_hyperparameters['ppd_no-fg_ft_target_model_dir']
    finetune_ppd_dann(dann_task_id,
                      ppd_pretain_model_root_dir,
                      ppd_finetune_target_root_dir,
                      fine_tune_fg_dann_hyperparameters,
                      data_hyperparameters,
                      create_ppd_global_model_func=create_no_fg_pdd_global_model)
