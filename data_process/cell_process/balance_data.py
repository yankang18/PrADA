import numpy as np
import pandas as pd


def create_balanced_training_data(dir, to_dir, table_names, pos_number, neg_number):
    prefix = "train_nus_"
    table_name = prefix + "target.csv"
    target_df = pd.read_csv(dir + table_name)
    num_pos_samples = target_df[target_df["target"] == 1].shape[0]
    num_neg_samples = target_df[target_df["target"] == 0].shape[0]
    print("num_pos_samples:", num_pos_samples)
    print("num_neg_samples:", num_neg_samples)

    pos_idx_list = np.random.choice(num_pos_samples, pos_number)
    neg_idx_list = np.random.choice(num_neg_samples, neg_number)
    all_perm_idx_list = np.random.permutation(pos_number + neg_number)
    do_create_balanced_B_train_tables(dir, to_dir, table_names, pos_idx_list, neg_idx_list, all_perm_idx_list,
                                      target_df)


def do_create_balanced_B_train_tables(dir, to_dir, table_names, pos_idx_list, neg_idx_list, all_perm_idx_list,
                                      target_df):
    prefix = "train_nus_"
    for table_name in table_names:
        df = pd.read_csv(dir + prefix + table_name)
        print(f"{table_name} table shape: {df.shape}")

        pos_samples = df[target_df["target"] == 1]
        neg_samples = df[target_df["target"] == 0]

        pos_values = pos_samples.values
        neg_values = neg_samples.values
        print("before balance pos_samples:", pos_values.shape, len(pos_idx_list))
        print("before balance neg_samples:", neg_values.shape, len(neg_idx_list))

        pos_values = pos_values if pos_values.shape[0] == len(pos_idx_list) else pos_values[pos_idx_list]
        neg_values = neg_values if neg_values.shape[0] == len(neg_idx_list) else neg_values[neg_idx_list]

        print("after balance pos_samples:", pos_values.shape)
        print("after balance neg_samples:", neg_values.shape)

        all_values = np.concatenate([pos_values, neg_values], axis=0)
        all_values = all_values[all_perm_idx_list]

        print("after balance all_values:", all_values.shape)

        all_df = pd.DataFrame(data=all_values, columns=df.columns)
        tokens = table_name.split(".")
        table_name = tokens[0]
        table_name_extension = "." + tokens[1]
        table_name = 'train_bal_' + table_name + "_" + str(pos_values.shape[0]) + "_" + str(
            neg_values.shape[0]) + table_name_extension
        df_file_path = to_dir + table_name
        all_df.to_csv(df_file_path, index=False)
        print(f"save named {table_name} to: {df_file_path}")


def combine_B_train_and_test_data():
    B_train = pd.read_csv("../../../data/cell_manager/B_train.csv")
    B_test = pd.read_csv("../../../data/cell_manager/B_test.csv")
    print("B train pos:", B_train[B_train['target'] == 1].shape)
    print("B test pos", B_test[B_test['target'] == 1].shape)

    B_train_values = B_train.values
    B_test_values = B_test.values
    B_cols = B_train.columns
    B_combine = np.concatenate([B_train_values, B_test_values], axis=0)
    B_combine_df = pd.DataFrame(data=B_combine, columns=B_cols)
    print(f"B_combine_df shape:{B_combine_df.shape}")
    file_path = "../../../data/cell_manager/B_combine.csv"
    B_combine_df.to_csv(file_path, index=False)
    print(f"save B_combine_df to {file_path}")


def balance_data(dir, to_dir, file_name):
    data = pd.read_csv(dir + file_name)
    print("loaded data")

    pos_samples = data[data["target"] == 1]
    neg_samples = data[data["target"] == 0]

    print(f"shape of pos_samples:{pos_samples.shape}")
    print(f"shape of neg_samples:{neg_samples.shape}")

    pos_values = pos_samples.values
    neg_values = neg_samples.values

    comb_values = np.concatenate([pos_values, pos_values, pos_values, pos_values, neg_values], axis=0)

    print(f"shape of comb_values:{comb_values.shape}")

    bal_file_name = file_name.split(".")[0] + "_bal.csv"
    comb_df = pd.DataFrame(data=comb_values, columns=data.columns)
    bal_file_path = to_dir + bal_file_name
    comb_df.to_csv(bal_file_path, index=False)
    print(f"save balanced data to {bal_file_path}")


if __name__ == "__main__":
    # combine_B_data()

    # dir = "../../../data/cell_manager/"
    # # file_name = "A_train.csv"
    # file_name = "B_combine.csv"
    # balance_data(dir, dir, file_name)

    data = pd.read_csv("../../../data/cell_manager/C_train_Data_2/train_nus_target.csv")
    num_samples = data.shape[0]

    pos_samples = data[data["target"] == 1]
    neg_samples = data[data["target"] == 0]
    num_pos_samples = pos_samples.shape[0]
    num_neg_samples = neg_samples.shape[0]
    print(f"shape of pos_samples:{pos_samples.shape}")
    print(f"shape of neg_samples:{neg_samples.shape}")

    # dir = "../../../data/cell_manager/B_train_Data_2/"
    # to_dir = "../../../data/cell_manager/B_train_Data_2/"
    dir = "../../../data/cell_manager/C_train_Data_2/"
    to_dir = "../../../data/cell_manager/C_train_Data_2/"
    table_names = ["p_named_demographics.csv", "p_named_equipment.csv", "p_named_risk.csv",
                   "p_named_cell_number_appears.csv", "p_tgi.csv", "p_fraud.csv", "p_app_finance_install.csv",
                   "p_app_finance_usage.csv", "p_app_life_install.csv", "p_app_life_usage.csv", "target.csv"]
    # pos_neg_num_pair = [(316, 316), (316, 500), (316, 1000), (316, 24691), (500, 2000), (1000, 2000), (1000, 4000),
    #                     (2000, 4000),
    #                     (2000, 12000), {4000, 12000}, (2000, 24691), {4000, 24691}, (6000, 24691), (10000, 24691)]
    pos_neg_num_pair = [(205, 4795), (700, 4795), (1000, 4795), (1200, 4795), (1500, 4795)]
    for pair in pos_neg_num_pair:
        pos_number, neg_number = pair
        print(f"==> create training data with pos samples:{pos_number} and neg samples:{neg_number}")
        create_balanced_training_data(dir, dir, table_names, pos_number, neg_number)
