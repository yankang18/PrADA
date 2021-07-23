import pandas as pd

from data_process import data_process_utils

if __name__ == "__main__":

    ts = '20210522'
    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"
    to_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"
    # data_tag = 'lbl004tgt4000v4'
    # data_tag = 'lbl002tgt4000'
    data_tag = 'lbl001tgt4000'
    source_train_file_name = data_dir + f"PPD_2014_src_1to9_da_{data_tag}_train.csv"
    target_train_file_name = data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_train.csv'

    df_src_data = pd.read_csv(source_train_file_name, skipinitialspace=True)
    df_tgt_data = pd.read_csv(target_train_file_name, skipinitialspace=True)
    print("[INFO] df_src_data shape:", df_src_data.shape)
    print("[INFO] df_tgt_data shape:", df_tgt_data.shape)

    df_data = data_process_utils.combine_src_tgt_data(df_src_data, df_tgt_data)
    print("[INFO] df_src_tgt_data shape:", df_data.shape)

    file_full_name = "{}/PPD_2014_src_tgt_{}_{}.csv".format(to_dir, data_tag, "train")
    data_process_utils.save_df_data(df_data, file_full_name)
