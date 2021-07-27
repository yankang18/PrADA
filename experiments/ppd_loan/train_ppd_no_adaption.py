from experiments.ppd_loan.global_config import data_tag, data_hyperparameters, no_adaptation_hyperparameters
from experiments.ppd_loan.train_ppd_utils import train_no_adaptation

if __name__ == "__main__":
    ppd_no_da_root_dir = data_hyperparameters["ppd_no-da_model_dir"]
    train_no_adaptation(data_tag,
                        ppd_no_da_root_dir,
                        no_adaptation_hyperparameters,
                        data_hyperparameters)

