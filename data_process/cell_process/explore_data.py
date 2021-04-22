import numpy as np
import pandas as pd


def show_data_info(file_full_name):
    data = pd.read_csv(file_full_name)
    print("data:{0}, pos:{1}".format(data.shape, data[data['target'] == 1].shape))


def show_party_data_info(data_dir, id, show_test=False):
    file_full_name = data_dir + id + "_train.csv"
    show_data_info(file_full_name)

    if show_test:
        file_full_name = data_dir + id + "_test.csv"
        show_data_info(file_full_name)


if __name__ == "__main__":
    data_dir = "../../../data/cell_manager/"

    # print("[INFO] party A data:")
    # show_party_data_info(data_dir, "A", show_test=False)

    print("[INFO] party B data:")
    show_party_data_info(data_dir, "B", show_test=True)
    party_b_data_name = "B" + "_combine.csv"
    show_data_info(data_dir + party_b_data_name)
    party_b_data_name = "B" + "_combine_bal.csv"
    show_data_info(data_dir + party_b_data_name)

    # print("[INFO] party C data:")
    # party_c_data_name = "C" + "_combine.csv"
    # show_data_info(data_dir + party_c_data_name)
