feature_extractor_architecture_list = [
    [15, 20, 15, 6],
    [85, 100, 60, 8],
    [30, 50, 30, 6],
    [18, 30, 18, 6],
    [55, 70, 30, 8]]

no_fg_feature_extractor_architecture = [203, 210, 70, 20]

pre_train_fg_dann_hyperparameters = {
    "using_interaction": True,
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "lr": 5e-4,
    "batch_size": 64,
    "max_epoch": 600,
    "epoch_patience": 5,
    "pos_class_weight": 1.0,
    "valid_metric": ('ks', 'auc')
}

fine_tune_fg_dann_hyperparameters = {
    "using_interaction": True,
    "load_global_classifier": False,
    "momentum": 0.99,
    "weight_decay": 0.0,
    "lr": 6e-4,
    "batch_size": 64,
    "pos_class_weight": 1.0,
    "valid_metric": ('ks', 'auc')
}

no_adaptation_hyperparameters = {
    "momentum": 0.99,
    "weight_decay": 0.00001,
    "batch_size": 64,
    "lr": 5e-4,
    "pos_class_weight": 1.0,
    "epoch_patience": 8,
    "max_epochs": 800,
    "metrics": ('ks', 'auc')
}

data_tag = 'lbl004tgt4000v4'
# data_tag = 'lbl002tgt4000'
# data_tag = 'lbl001tgt4000'
ts = '20210522'
data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"

data_hyperparameters = {
    "source_train_file_name": data_dir + f"PPD_2014_src_1to9_da_{data_tag}_train.csv",
    "source_test_file_name": data_dir + f'PPD_2014_src_1to9_da_{data_tag}_test.csv',
    "target_train_file_name": data_dir + f'PPD_2014_tgt_10to12_da_{data_tag}_train.csv',
    "target_test_file_name": data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_test.csv',
    "src_tgt_train_file_name": data_dir + f"PPD_2014_src_tgt_{data_tag}_train.csv",

    "ppd_fg_dann_model_dir": "ppd_fg_dann",
    "ppd_fg_ft_target_model_dir": "ppd_fg_target",
    "ppd_no-fg_dann_model_dir": "ppd_no-fg_dann",
    "ppd_no-fg_ft_target_model_dir": "ppd_no-fg_target",
    "ppd_no-ad_model_dir": "ppd_no-ad_model"

}
