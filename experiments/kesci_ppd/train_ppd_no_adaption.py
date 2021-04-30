from datasets.ppd_dataloader import get_datasets, get_dataloader
from experiments.kesci_ppd.train_ppd_dann import wire_global_model
from models.experiment_dann_learner import FederatedDAANLearner
from utils import get_timestamp

if __name__ == "__main__":

    exp_dir = "ppd_dann"
    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    src_tgt_train_file_name = data_dir + "PPD_2014_train.csv"
    target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    # source_test_file_name = data_dir + 'PPD_2014_1to9_test.csv'
    target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'

    print(f"[INFO] load data from {data_dir}")
    # all_train_ds = get_dataset(dir=all_dir, data_mode="train")
    # target_train_ds = get_dataset(dir=target_dir, data_mode="train")
    # test_ds = get_dataset(dir=target_dir, data_mode="test")

    split_ratio = 1.0
    src_tgt_train_dataset, _ = get_datasets(ds_file_name=src_tgt_train_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_train_dataset, _ = get_datasets(ds_file_name=target_train_file_name, shuffle=True, split_ratio=split_ratio)
    # src_test_dataset, _ = get_datasets(ds_file_name=source_test_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_test_dataset, _ = get_datasets(ds_file_name=target_test_file_name, shuffle=True, split_ratio=split_ratio)

    # batch_size_list = [512]
    # learning_rate_list = [2e-4, 3e-4, 4e-4]
    batch_size_list = [256]
    learning_rate_list = [5e-4]
    # learning_rate_list = [5e-4]
    is_all_list = [True]
    tries = 2
    pos_class_weight = 1.0

    param_comb_list = list()
    for is_all in is_all_list:
        for lr in learning_rate_list:
            for bs in batch_size_list:
                param_comb_list.append((is_all, lr, bs))

    timestamp = get_timestamp()
    date = "20200122_pdd_no_da"
    for param_comb in param_comb_list:
        is_all, lr, bs = param_comb
        for version in range(tries):
            train_data_tag = "all" if is_all else "tgt"
            task_id = date + "_" + train_data_tag + "_pw" + str(pos_class_weight) + "_bs" + str(bs) + "_lr" + str(
                lr) + "_v" + str(version) + "_t" + str(timestamp)
            print("[INFO] perform task:{0}".format(task_id))

            daan_model = wire_global_model(pos_class_weight=pos_class_weight)
            print("[INFO] model created.")

            if is_all:
                train_loader = get_dataloader(src_tgt_train_dataset, batch_size=bs)
            else:
                train_loader = get_dataloader(tgt_train_dataset, batch_size=bs)

            test_loader = get_dataloader(tgt_test_dataset, batch_size=bs * 4)

            plat = FederatedDAANLearner(model=daan_model,
                                        source_train_loader=None,
                                        source_val_loader=None,
                                        target_train_loader=train_loader,
                                        target_val_loader=test_loader,
                                        epoch_patience=25,
                                        number_validations=6)

            plat.set_model_save_info(exp_dir)
            plat.train_wo_adaption(epochs=500, lr=lr, source=False, task_id=task_id)
