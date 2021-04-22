from datasets.cell_manage_dataloader import get_dataset, get_cell_manager_dataloader
from experiments.tencent_cell_manage.train_cell_dann import create_global_daan_model
from models.experiment_dann_learner import FederatedDAANLearner
from utils import get_timestamp

if __name__ == "__main__":

    all_dir = "/Users/yankang/Documents/Data/cell_manager/A_B_train_data/"
    # target_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_3/"
    target_dir = "/Users/yankang/Documents/Data/cell_manager/B_train_data_4/"
    exp_dir = "cell_dann"

    # print(f"[INFO] load source data from {source_dir}")
    # src_train_ds = get_dataset(dir=source_dir, data_mode="train")
    # src_test_ds = get_dataset(dir=source_dir, data_mode="test")

    print(f"[INFO] load target data from {target_dir}")
    all_train_ds = get_dataset(dir=all_dir, data_mode="train")
    target_train_ds = get_dataset(dir=target_dir, data_mode="train")
    test_ds = get_dataset(dir=target_dir, data_mode="test")

    batch_size_list = [512]
    learning_rate_list = [4e-4, 6e-4]
    # batch_size_list = [512]
    # learning_rate_list = [3e-4]
    is_all_list = [False, True]
    tries = 1
    pos_class_weight = 1.0

    param_comb_list = list()
    for is_all in is_all_list:
        for lr in learning_rate_list:
            for bs in batch_size_list:
                param_comb_list.append((is_all, lr, bs))

    timestamp = get_timestamp()
    date = "20200120_cell_no_da"
    for param_comb in param_comb_list:
        is_all, lr, bs = param_comb
        for version in range(tries):
            train_data_tag = "all" if is_all else "tgt"
            epoch_patience = 3 if is_all else 10
            number_validations = 20 if is_all else 6
            task_id = date + "_pw" + str(pos_class_weight) + "_bs" + str(bs) + "_lr" + str(lr) + "_v" + str(
                version) + "_t" + str(timestamp)
            print("[INFO] perform task:{0}".format(task_id))

            daan_model = create_global_daan_model(pos_class_weight=pos_class_weight)
            print("[INFO] model created.")

            if is_all:
                train_loader = get_cell_manager_dataloader(all_train_ds, batch_size=bs)
            else:
                train_loader = get_cell_manager_dataloader(target_train_ds, batch_size=bs)

            test_loader = get_cell_manager_dataloader(test_ds, batch_size=bs * 4)

            plat = FederatedDAANLearner(model=daan_model,
                                        source_train_loader=None,
                                        source_val_loader=None,
                                        target_train_loader=train_loader,
                                        target_val_loader=test_loader,
                                        epoch_patience=epoch_patience,
                                        number_validations=number_validations)

            plat.set_model_save_info(exp_dir)
            plat.train_wo_adaption(epochs=500, lr=lr, source=False, task_id=task_id)
