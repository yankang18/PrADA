from datasets.ppd_dataloader import get_datasets, get_dataloader
# from experiments.kesci_ppd.global_config import tgt_data_tag, src_data_tag, tgt_tag
from experiments.kesci_ppd.train_ppd_fg_dann import create_pdd_global_model
from experiments.kesci_ppd.train_ppd_no_fg_dann import create_global_model
from models.experiment_dann_learner import FederatedDAANLearner
from utils import get_timestamp, get_current_date

if __name__ == "__main__":

    # TODO: lbl001tgt4000
    # DA: acc:0.8908530651962374, auc:0.7566014657272395, ks:0.4208019635040358

    exp_dir = "ppd_no_dann"
    # data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_1620085151/"
    # # source_train_file_name = data_dir + f"PPD_2014_src_1to9_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_train.csv"
    # # source_test_file_name = data_dir + f'PPD_2014_src_1to9_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_test.csv'
    # target_train_file_name = data_dir + f'PPD_2014_tgt_10to11_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_train.csv'
    # target_test_file_name = data_dir + f'PPD_2014_tgt_10to11_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_test.csv'
    # src_tgt_train_file_name = data_dir + f'PPD_2014_src_tgt_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_train.csv'

    ts = '20210522'
    # data_tag = 'lbl004tgt4000v4'
    # data_tag = 'lbl002tgt4000'
    data_tag = 'lbl001tgt4000'
    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"
    source_train_file_name = data_dir + f"PPD_2014_src_1to9_da_{data_tag}_train.csv"
    source_test_file_name = data_dir + f'PPD_2014_src_1to9_da_{data_tag}_test.csv'
    target_train_file_name = data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_train.csv'
    target_test_file_name = data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_test.csv'
    src_tgt_train_file_name = data_dir + f'PPD_2014_src_tgt_{data_tag}_train.csv'

    # data_file_name_dict = {"src": source_train_file_name,
    #                        "tgt": target_train_file_name,
    #                        "all": src_tgt_train_file_name}
    is_train_source = True
    is_valid_source = False

    split_ratio = 1.0
    # src_tgt_train_dataset = None
    src_tgt_train_dataset, _ = get_datasets(ds_file_name=src_tgt_train_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_train_dataset, _ = get_datasets(ds_file_name=target_train_file_name, shuffle=True, split_ratio=split_ratio)
    src_test_dataset, _ = get_datasets(ds_file_name=source_test_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_test_dataset, _ = get_datasets(ds_file_name=target_test_file_name, shuffle=True, split_ratio=split_ratio)

    momentum = 0.99
    weight_decay = 0.00001
    # weight_decay = 0.0
    # batch_size_list = [512]
    # learning_rate_list = [2e-4, 3e-4, 4e-4]
    batch_size_list = [64]
    # learning_rate_list = [5e-4]
    # learning_rate_list = [8e-4]
    learning_rate_list = [5e-4]
    is_all_list = [False]
    tries = 1
    pos_class_weight = 1.0
    epoch_patience = 8
    max_epochs = 800
    metrics = ('ks', 'auc')

    param_comb_list = list()
    for is_all in is_all_list:
        for lr in learning_rate_list:
            for bs in batch_size_list:
                param_comb_list.append((is_all, lr, bs))

    timestamp = get_timestamp()
    date = get_current_date() + "_PPD_no_da_fg"
    for param_comb in param_comb_list:
        is_all, lr, bs = param_comb
        for version in range(tries):
            train_data_tag = "all" if is_all else "tgt"
            task_id = date + "_" + train_data_tag + "-" + data_tag + "_pw" + str(pos_class_weight) + "_bs" + str(
                bs) + "_lr" + str(lr) + "_v" + str(version) + "_t" + str(timestamp)
            print("[INFO] perform task:{0}".format(task_id))

            # model = create_pdd_global_model(pos_class_weight=pos_class_weight)
            model = create_global_model(aggregation_dim=5,
                                        num_wide_feature=6,
                                        pos_class_weight=pos_class_weight)
            print("[INFO] model created.")

            if is_all:
                src_train_loader = get_dataloader(src_tgt_train_dataset, batch_size=bs)
                # pass
            else:
                src_train_loader = get_dataloader(tgt_train_dataset, batch_size=bs)
            tgt_train_loader = get_dataloader(tgt_train_dataset, batch_size=bs)

            tgt_valid_loader = get_dataloader(tgt_test_dataset, batch_size=bs * 4)
            src_valid_loader = get_dataloader(src_test_dataset, batch_size=bs * 4)

            plat = FederatedDAANLearner(model=model,
                                        source_da_train_loader=src_train_loader,
                                        source_val_loader=src_valid_loader,
                                        target_da_train_loader=tgt_train_loader,
                                        target_val_loader=tgt_valid_loader,
                                        epoch_patience=epoch_patience,
                                        validation_batch_interval=5)
            plat.set_model_save_info(exp_dir)

            # optimizer_param_dict = {"src": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay},
            #                         "tgt": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}}
            plat.train_wo_adaption(epochs=max_epochs,
                                   lr=lr,
                                   train_source=is_train_source,
                                   valid_source=is_valid_source,
                                   metric=metrics,
                                   task_id=task_id)
