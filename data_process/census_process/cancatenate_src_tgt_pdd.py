import pandas as pd


def save_df(df, to_dir, file_name):
    file_path = to_dir + file_name
    df.to_csv(file_path, index=False)
    print(f"save {file_name} with shape {df.shape} to {file_path}")


if __name__ == "__main__":
    source_train_file = "PPD_2014_1to9_train.csv"
    target_train_file = 'PPD_2014_10to12_train.csv'
    source_test_file = 'PPD_2014_1to9_test.csv'
    target_test_file = 'PPD_2014_10to12_test.csv'

    src_tgt_train_file = 'PPD_2014_train.csv'
    src_tgt_test_file = 'PPD_2014_test.csv'

    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    to_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"

    df_train_src = pd.read_csv(data_dir + source_train_file)
    df_train_tgt = pd.read_csv(data_dir + target_train_file)
    df_train_all = pd.concat([df_train_src, df_train_tgt], axis=0)
    print("df_train_src shape:{0}".format(df_train_src.shape))
    print("df_train_tgt shape:{0}".format(df_train_tgt.shape))
    print("df_train_all shape:{0}".format(df_train_all.shape))

    df_test_src = pd.read_csv(data_dir + source_test_file)
    df_test_tgt = pd.read_csv(data_dir + target_test_file)
    df_test_all = pd.concat([df_test_src, df_test_tgt], axis=0)
    print("df_test_src shape:{0}".format(df_test_src.shape))
    print("df_test_tgt shape:{0}".format(df_test_tgt.shape))
    print("df_test_all shape:{0}".format(df_test_all.shape))

    save_df(df_train_all, to_dir, src_tgt_train_file)
    save_df(df_test_all, to_dir, src_tgt_test_file)
