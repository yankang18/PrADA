from torch.utils.data import DataLoader

from datasets.cell_manage_dataloader import get_dataset, show_dataset

# table_names = ["p_wide_col.csv", "p_debt_feat.csv", "p_payment_feat.csv",
#                "p_payment_debt_cross_feat.csv", "p_multi_acc_feat.csv", "p_mal_behavior_feat.csv",
#                "p_qualify_feat.csv", "p_loan_feat.csv", "target.csv"]

table_names = ["p_wide_col.csv", "p_debt_feat.csv", "p_payment_feat.csv",
               "p_payment_debt_cross_feat.csv", "p_multi_acc_feat.csv", "p_mal_behavior_feat.csv",
               "p_qualify_feat.csv", "target.csv"]


def get_lending_dataloader(dir, batch_size=512, num_workers=2, data_mode="train", shuffle=False, suffix=None):
    dataset = get_dataset(dir=dir, file_name_list=table_names, data_mode=data_mode, suffix=suffix)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    # source_dir = "../../../data/lending_club_bundle_archive/loan_processed_2015_17/"
    # target_dir = "../../../data/lending_club_bundle_archive/loan_processed_2018/"

    # source_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_17/"
    source_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2016_17/"
    target_dir = "../../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"

    src_train_data_loader = get_lending_dataloader(dir=source_dir, data_mode="train")
    src_test_data_loader = get_lending_dataloader(dir=source_dir, data_mode="test")
    print(f"src_train_data_loader {len(src_train_data_loader.dataset)}")
    show_dataset(src_train_data_loader)
    print(f"src_test_data_loader {len(src_test_data_loader.dataset)}")
    show_dataset(src_test_data_loader)

    tgt_train_data_loader = get_lending_dataloader(dir=target_dir, data_mode="train")
    tgt_val_data_loader = get_lending_dataloader(dir=target_dir, data_mode="val")
    tgt_test_data_loader = get_lending_dataloader(dir=target_dir, data_mode="test")

    # print(f"tgt_train_data_loader {len(tgt_train_data_loader.dataset)}")
    # show_dataset(tgt_train_data_loader)
    # print(f"tgt_val_data_loader {len(tgt_val_data_loader.dataset)}")
    # show_dataset(tgt_val_data_loader)
    # print(f"tgt_test_data_loader {len(tgt_test_data_loader.dataset)}")
    # show_dataset(tgt_test_data_loader)
