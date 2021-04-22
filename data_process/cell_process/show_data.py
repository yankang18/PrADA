from collections import OrderedDict

import numpy as np
import pandas as pd


def split_demographics_data(load_dir, to_dir, load_file_name, data_mode="train"):
    df = pd.read_csv(load_dir + data_mode + "_" + load_file_name)
    cols = list(df.columns)

    demo_list = ['2000026105', '2000026480', '2000045217']
    asset_list = [x for x in cols if x not in demo_list]

    print("cols:{0} {1}".format(cols, len(cols)))
    print("demo_list", demo_list, len(demo_list))
    print("asset_list", asset_list, len(asset_list))

    # col_list = ['2000000136', '2000026105', '2000026114', '2000026480',
    #             '2000000134', '2000026497', '2000040011',
    #             '2000045217', '2000045218', '2000054989', '2000054990']
    # for col in col_list:
    #     if col in cols:
    #         print("{0} is in table".format(col))
    #     else:
    #         print("{0} is not in table".format(col))

    df_demo = df[demo_list]
    df_asset = df[asset_list]

    demo_file_path = to_dir + data_mode + "_demo.csv"
    asset_file_path = to_dir + data_mode + "_asset.csv"
    df_demo.to_csv(demo_file_path, index=False)
    print(f"save {demo_file_path} with shape {df_demo.shape} to {demo_file_path}")
    df_asset.to_csv(asset_file_path, index=False)
    print(f"save {asset_file_path} with shape {df_asset.shape} to {asset_file_path}")


if __name__ == "__main__":
    source_dir = "/Users/yankang/Documents/Data/cell_manager/A_train_data_3/"
    target_dir = "../../../data/cell_manager/B_train_data_4/"

    split_demographics_data(source_dir, source_dir, "demographics.csv", data_mode="train")
    split_demographics_data(source_dir, source_dir, "demographics.csv", data_mode="test")

    split_demographics_data(target_dir, target_dir, "demographics.csv", data_mode="train")
    split_demographics_data(target_dir, target_dir, "demographics.csv", data_mode="test")
