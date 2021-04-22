import pandas as pd

from data_process.cell_process.create_data_table_util import create_tables, save_df_tables, get_all_tag_ids
from data_process.cell_process.preprocess_data_util import TransformPipeLine, StandardTransformer, MissingFillTransformer

feature_group_conf_list = [("demographics", ['人口属性', '社会属性', '资产', '资产分属性'], None),
                           ("fraud", ['v1版反欺诈分', 'v2反欺诈分', 'v3反欺诈分', 'v3版反欺诈分底层特征'], "fraudv3_wifi_ids.csv"),
                           ("risk", ['贷款', '高危', '逾期类', '贷款行为', '高危行为'], None),
                           ("equipment", None, "equipment_ids.txt"),
                           ("tgi", None, "tgi_ids.txt"),
                           ("cell_number_appears", ['号码出现'], None),
                           ("app_finance_install", None, "app_finance_install_ids.txt"),
                           ("app_finance_usage", None, "app_finance_usage_ids.txt"),
                           ("app_life_install", None, "app_life_install_ids.txt"),
                           ("app_life_usage", None, "app_life_usage_ids.txt")]


def create_data_tables_pipeline(dir_dict, id_mean_v5_df, all_tag_ids):
    data_dir = dir_dict['data_dir']
    resource_dir = dir_dict['resource_dir']
    to_dir = dir_dict['save_dir']
    data_to_split = pd.read_csv(data_dir)
    feat_group_df_list, target_df = create_tables(feature_group_conf_list,
                                                  resource_dir,
                                                  id_mean_v5_df,
                                                  all_tag_ids,
                                                  data_to_split,
                                                  target_col_name="target")

    # perform transform
    transform_pipeline = TransformPipeLine([MissingFillTransformer(),
                                            StandardTransformer()])
    trans_fg_df_list = [transform_pipeline.transform(fg_df) for fg_df in feat_group_df_list]

    # save df tables
    named_table = [(fg_conf[0], fg_df) for fg_conf, fg_df in zip(feature_group_conf_list, trans_fg_df_list)]
    named_table.append(('target', target_df))
    save_df_tables(named_table, to_dir)


# def create_data_tables_for_B_data(resource_dir, to_dir, id_mean_v5_df, all_tag_ids):
#     print("==> create data tables for B (from target domain) data ...")
#     # B_combine is the combination of B_train and B_test
#     data_to_split = pd.read_csv("../../../data/cell_manager/B_combine.csv")
#     feat_group_df_list, target_df = create_tables(feature_group_conf_list,
#                                                   resource_dir,
#                                                   id_mean_v5_df,
#                                                   all_tag_ids,
#                                                   data_to_split,
#                                                   target_col_name="target")
#
#     # perform transform
#     transform_pipeline = TransformPipeLine([MissingFillTransformer(),
#                                             StandardTransformer()])
#     trans_fg_df_list = [transform_pipeline.transform(fg_df) for fg_df in feat_group_df_list]
#
#     # save df tables
#     named_table = [(fg_conf[0], fg_df) for fg_conf, fg_df in zip(feature_group_conf_list, trans_fg_df_list)]
#     save_df_tables(named_table, to_dir)


# def create_data_tables_for_C_data(resource_dir, to_dir, id_mean_v5_df, all_tag_ids, create_named_tag_ids=None):
#     print("==> create data tables for C train data ...")
#     # C_combine is the combination of C_train and C_test
#     data = pd.read_csv("../../../data/cell_manager/C_combine.csv")
#     print(f"C data shape:{data.shape}")
#     print(f"C label 1 data shape:{data[data['y'] == 1].shape}")
#     print(f"C label 0 data shape:{data[data['y'] == 0].shape}")
#     create_tables(resource_dir, to_dir, id_mean_v5_df, all_tag_ids, data, target_col_name='y')
#     id2name_dict, _, _ = get_tag_id2info_dicts(id_mean_v5_df)
#
#     if create_named_tag_ids:
#         table_names = ['demographics.csv', 'equipment.csv', 'risk.csv', 'cell_number_appears.csv']
#         create_named_tag_ids(to_dir, to_dir, table_names, id2name_dict)


if __name__ == "__main__":
    resource_dir = "../../../data/cell_manager/"
    all_tag_id_list = get_all_tag_ids(resource_dir)
    tag_id_meaning_v5_df = pd.read_csv(resource_dir + 'ID_meaning_v5.csv')

    data_dir = "../../../data/cell_manager/A_train_bal.csv"
    resource_dir = "../../../data/cell_manager/tag_ids_files/"
    save_dir = "../../../data/cell_manager/A_train_data_3/"
    dir_dict = {'data_dir': data_dir, 'resource_dir': resource_dir, 'save_dir': save_dir}
    create_data_tables_pipeline(dir_dict, tag_id_meaning_v5_df, all_tag_id_list)

    data_dir = "../../../data/cell_manager/B_combine.csv"
    resource_dir = "../../../data/cell_manager/tag_ids_files/"
    save_dir = "../../../data/cell_manager/B_train_data_3/"
    dir_dict = {'data_dir': data_dir, 'resource_dir': resource_dir, 'save_dir': save_dir}
    create_data_tables_pipeline(dir_dict, tag_id_meaning_v5_df, all_tag_id_list)

    # to_dir = "../../../data/cell_manager/C_train_data_3/"
    # resource_dir = "../../../data/cell_manager/tag_ids_files/"
    # create_data_tables_for_C_data(resource_dir, to_dir, tag_id_meaning_v5_df, all_tag_id_list)
