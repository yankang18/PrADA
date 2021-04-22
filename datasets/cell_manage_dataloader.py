import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from datasets.cell_manage_dataset import MultiPartitionDataset

# table_names = ["demographics.csv", "equipment.csv", "risk.csv",
#                "cell_number_appears.csv", "app_finance_install.csv", "app_finance_usage.csv",
#                "app_life_install.csv", "app_life_usage.csv", "fraud.csv", "tgi.csv", "target.csv"]

table_names = ["demo.csv", "asset.csv", "equipment.csv", "risk.csv",
               "cell_number_appears.csv", "app_finance_install.csv", "app_finance_usage.csv",
               "app_life_install.csv", "app_life_usage.csv", "fraud.csv", "tgi.csv", "target.csv"]


def shuffle_data(data):
    len = data.shape[0]
    perm_idxs = np.random.permutation(len)
    return data[perm_idxs]


def get_data(dir, file_name_list, data_mode, is_shuffle=True, suffix=None):
    data_mode_dict = {"train_bal": "train_bal_",
                      "train_nus": "train_nus_",
                      "train": "train_",
                      "val": "val_",
                      "test": "test_"}
    prefix = data_mode_dict.get(data_mode)
    if prefix is None:
        raise RuntimeError(f"Does not support data_mode:{data_mode}")

    data_list = []
    target = None
    perm_idx = None
    for idx, file_name in enumerate(file_name_list):
        tokens = file_name.split(".")
        file_name = tokens[0]
        file_ext = "." + tokens[1]
        file_name = prefix + file_name + file_ext if suffix is None else prefix + file_name + suffix + file_ext
        df = pd.read_csv(dir + file_name)
        values = df.values
        print(f"[INFO] loaded {file_name} table, which has shape:{values.shape}")
        if is_shuffle:
            if perm_idx is None:
                num_samples = df.shape[0]
                perm_idx = np.random.permutation(num_samples)
            values = df.values[perm_idx]
            print(f"[INFO] -- data in {file_name} has been shuffled")
        # values = values[:10000] if data_mode == 'train_nus' else values
        if idx == len(file_name_list) - 1:
            num_pos = np.sum(values)
            num_neg = len(values) - num_pos
            print(f"[INFO] ---- number of positive sample:{num_pos}")
            print(f"[INFO] ---- number of negative sample:{num_neg}")
            target = values
        else:
            data_list.append(values)
    return data_list, target


def get_dataset(dir, data_mode, is_shuffle=True, suffix=None, num_samples=None):
    data_list, target = get_data(dir, table_names, data_mode, is_shuffle=is_shuffle, suffix=suffix)
    if num_samples is not None:
        target_2 = target[:num_samples]
        data_list_2 = [data[:num_samples] for data in data_list]
        print("data_list_2:", data_list_2[0].shape, data_list_2[1].shape, data_list_2[2].shape)
        print("target_2:", target_2.shape)
        return MultiPartitionDataset(data_list_2, target_2)
    else:
        return MultiPartitionDataset(data_list, target)


def get_cell_manager_dataloader(dataset, batch_size=128, num_workers=2, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_cell_manager_dataloader_ob(dir, batch_size=128, num_workers=2, data_mode="train", shuffle=False, suffix=None):
    dataset = get_dataset(dir=dir, data_mode=data_mode, suffix=suffix)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def show_dataset(dataloader):
    data_batch, target = next(iter(dataloader))
    for data in data_batch:
        print(data.shape)
    print(f"target shape:{target.shape}")
    # print(f"number of positive samples:{np.sum(target)}")


if __name__ == "__main__":
    dir = "../../../data/cell_manager/A_train_data/"
    A_train_loader = get_cell_manager_dataloader_ob(dir=dir, data_mode="train")
    A_test_loader = get_cell_manager_dataloader_ob(dir=dir, data_mode="test")
    print(f"A_train_loader {len(A_train_loader.dataset)}")
    show_dataset(A_train_loader)
    print(f"A_test_loader {len(A_test_loader.dataset)}")
    show_dataset(A_test_loader)

    # dir = "../../../data/cell_manager/C_train_data_2/"
    dir = "../../../data/cell_manager/B_train_data_2/"
    B_train_loader = get_cell_manager_dataloader_ob(dir=dir, data_mode="train")
    # B_val_loader = get_cell_manager_dataloader(dir=dir, data_mode="val")
    B_test_loader = get_cell_manager_dataloader_ob(dir=dir, data_mode="test")

    print(f"B_train_loader {len(B_train_loader.dataset)}")
    show_dataset(B_train_loader)
    # print(f"B_val_loader {len(B_val_loader.dataset)}")
    # show_dataset(B_val_loader)
    print(f"B_test_loader {len(B_test_loader.dataset)}")
    show_dataset(B_test_loader)
