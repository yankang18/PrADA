import pandas as pd

from data_process import data_process_utils

if __name__ == "__main__":
    data_dir = "/Users/yankang/Documents/Data/census/output/"

    # tag = "all4000pos004v4"
    # tag = "all4000pos002"
    tag = 'all4000pos001'
    source_train_file_name = data_dir + f'undergrad_census9495_da_{tag}_train.csv'
    target_train_file_name = data_dir + f'grad_census9495_ft_{tag}_train.csv'

    df_src_data = pd.read_csv(source_train_file_name, skipinitialspace=True)
    df_tgt_data = pd.read_csv(target_train_file_name, skipinitialspace=True)
    print("[INFO] df_src_data shape:", df_src_data.shape)
    print("[INFO] df_tgt_data shape:", df_tgt_data.shape)

    df_data = data_process_utils.combine_src_tgt_data(df_src_data, df_tgt_data)
    print("[INFO] df_src_tgt_data shape:", df_data.shape)

    file_full_name = "{}/degree_src_tgt_census9495_da_{}_train.csv".format(data_dir, tag)

    data_process_utils.save_df_data(df_data, file_full_name)
