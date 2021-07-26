from experiments.income_census.global_config import data_tag, no_adaptation_fg_hyperparameters, data_hyperparameters
from experiments.income_census.train_census_utils import train_no_adaptation

if __name__ == "__main__":
    census_no_da_root_dir = data_hyperparameters["census_no-da_model_dir"]
    train_no_adaptation(data_tag,
                        census_no_da_root_dir,
                        no_adaptation_fg_hyperparameters,
                        data_hyperparameters)
