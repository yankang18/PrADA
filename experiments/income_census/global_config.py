fg_feature_extractor_architecture_list = [[28, 56, 28, 14],
                                          [25, 50, 25, 12],
                                          [56, 86, 56, 18],
                                          [27, 54, 27, 13]]

intr_fg_feature_extractor_for_architecture_list = [[53, 78, 53, 15],
                                                   [84, 120, 84, 20],
                                                   [55, 81, 55, 15],
                                                   [81, 120, 81, 20],
                                                   [52, 78, 52, 15],
                                                   [83, 120, 83, 20]]

no_fg_feature_extractor_architecture = [136, 150, 60, 30]

pre_train_fg_dann_hyperparameters = {
    "using_interaction": False,
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "lr": 6e-4,
    "batch_size": 128,
    "max_epochs": 600,
    "epoch_patience": 3,
    "valid_metric": ('ks', 'auc')
}

fine_tune_fg_dann_hyperparameters = {
    "using_interaction": False,
    "load_global_classifier": False,
    "momentum": 0.99,
    "weight_decay": 0.0,
    "lr": 8e-4,
    "batch_size": 128,
    "valid_metric": ('ks', 'auc')
}

no_adaptation_hyperparameters = {
    "apply_feature_group": False,
    "train_data_tag": 'all',  # can be either 'all' or 'tgt'
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "lr": 5e-4,
    "batch_size": 128,
    "epochs": 800,
    "epoch_patience": 8,
    "valid_metric": ('ks', 'auc')
}

data_dir = "/Users/yankang/Documents/Data/census/output_v2/"
# data_tag = "all4000pos001"
# data_tag = 'all4000pos002'
# data_tag = 'all4000pos004v4'
# data_tag = 'all4000pos004v5'
# data_tag = 'all4000pos004v6'
data_tag = 'all4000pos004v7'

data_hyperparameters = {
    "source_ad_train_file_name": data_dir + f'undergrad_census9495_ad_{data_tag}_train.csv',
    "source_ad_valid_file_name": data_dir + f'undergrad_census9495_ad_{data_tag}_valid.csv',
    "src_tgt_train_file_name": data_dir + f'degree_src_tgt_census9495_ad_{data_tag}_train.csv',

    "target_ad_train_file_name": data_dir + f'grad_census9495_ad_{data_tag}_train.csv',
    "target_ft_train_file_name": data_dir + f'grad_census9495_ft_{data_tag}_train.csv',
    "target_ft_valid_file_name": data_dir + f'grad_census9495_ft_{data_tag}_valid.csv',
    "target_ft_test_file_name": data_dir + f'grad_census9495_ft_{data_tag}_test.csv',

    "census_fg_dann_model_dir": "census_fg_dann_model",
    "census_fg_ft_target_model_dir": "census_fg_target_model",
    "census_no-fg_dann_model_dir": "census_no-fg_dann_model",
    "census_no-fg_ft_target_model_dir": "census_no-fg_target_model",
    "census_no-ad_model_dir": "census_no-ad_model"
}
