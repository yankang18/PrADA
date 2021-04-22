import pandas as pd


# def create_demographics_table(resource_dir, to_dir, all_col, data):
#     print(f" ==> creating demographics table ...")
#     matched_equipment_ids = fetch_demographics_ids(resource_dir, all_col)
#     # data = pd.read_csv(dir + data_file)
#     matched_equipment_df = data[matched_equipment_ids]
#     file_path = to_dir + 'demographics.csv'
#     matched_equipment_df.to_csv(file_path, index=False)
#     print(f" ==> save demographics table to :{file_path}")


# def create_asset_table(resource_dir, to_dir, all_col, data):
#     print(f" ==> creating asset table ...")
#     id_mean_v5_df = pd.read_csv(resource_dir + 'ID_meaning_v5.csv')
#     id2desc_dict, id2type1_dict, id2type2_dict = get_tag_id2info_dicts(id_mean_v5_df)
#
#     # get mapping from id to desc for app
#     result = get_tag_id2info_by_type(id2type2_dict, id2desc_dict, ['资产', '资产分属性'])
#     _, _, asset_ids = result
#     print(f"asset_ids:{asset_ids}")
#     matched_asset_ids = fetch_tag_ids_by_group(all_col, asset_ids)
#     print(f" matched_asset_ids:{matched_asset_ids} with length:{len(matched_asset_ids)}")
#     # data = pd.read_csv(dir + data_file)
#     asset_df = data[matched_asset_ids]
#     file_path = to_dir + 'asset.csv'
#     asset_df.to_csv(file_path, index=False)
#     print(f" ==> save asset table to :{file_path}")

def get_tag_id2info_dicts(data_df):
    """
        get mapping from tag ids to tag type and from tag ids to tag description
    :param data_df:
    :return:
    """

    id2name_dict = dict()
    id2type1_dict = dict()
    id2type2_dict = dict()
    for index, row in data_df.iterrows():
        # note that here tag code is the identical to tag id
        # print(type(row['tag_code']), row['tag_type'], row['tag_desc'])
        id2name_dict[str(row['tag_code'])] = row['tag_desc']
        id2type1_dict[str(row['tag_code'])] = row['tag_type1']
        id2type2_dict[str(row['tag_code'])] = row['tag_type2']
    return id2name_dict, id2type1_dict, id2type2_dict


def get_tag_id2info_by_types(id2type_dict, id2info_dict, type_list):
    """

    :param id2type_dict:
    :param id2info_dict:
    :param type_list:
    :return:
    """
    selected_id2info_dict = dict()
    selected_id2type_dict = dict()
    selected_ids_list = list()
    for key, val in id2type_dict.items():
        if val in type_list:
            print(key, id2info_dict[key])
            selected_id2info_dict[key] = id2info_dict[key]
            selected_id2type_dict[key] = val
            selected_ids_list.append(key)
    return selected_id2info_dict, selected_id2type_dict, selected_ids_list


def fetch_tag_ids_by_id_prefix(all_tag_id, id_prefix_list):
    matched_tag_id = []
    for tag_id in all_tag_id:
        id_prefix = tag_id[:10]
        # print(tag_id)
        if id_prefix in id_prefix_list:
            # print("add:", tag_id)
            matched_tag_id.append(tag_id)
    return matched_tag_id


def fetch_app_ids(dir, all_tag_ids, app_ids_file="app_ids.txt"):
    app_ids_df = pd.read_csv(dir + app_ids_file, header=None)
    print(app_ids_df.head())
    # print(app_ids_df.iloc[0].values, type(app_ids_df.iloc[0].values))
    app_install_ids = [str(val) for val in app_ids_df.iloc[0].values]
    app_use_ids = [str(val) for val in app_ids_df.iloc[1].values]
    print(app_install_ids, app_use_ids)
    matched_app_install_ids = fetch_tag_ids_by_id_prefix(all_tag_ids, app_install_ids)
    matched_app_use_ids = fetch_tag_ids_by_id_prefix(all_tag_ids, app_use_ids)
    return matched_app_install_ids, matched_app_use_ids


def fetch_equipment_ids(dir, all_tag_ids):
    equipment_ids_df = pd.read_csv(dir + 'equipment.txt', header=None)
    equipment_ids = [str(val) for val in equipment_ids_df.iloc[0].values]
    matched_app_install_ids = fetch_tag_ids_by_id_prefix(all_tag_ids, equipment_ids)
    return matched_app_install_ids


def fetch_demographics_ids(dir, all_tag_ids):
    demographics_ids_df = pd.read_csv(dir + 'demographics.txt', header=None)
    demographics_ids = [str(val) for val in demographics_ids_df.iloc[0].values]
    matched_demographics_ids = fetch_tag_ids_by_id_prefix(all_tag_ids, demographics_ids)
    return matched_demographics_ids


def create_app_table(dir, to_dir, all_tag_ids, data, app_ids_file="app_ids.txt"):
    print(f" ==> creating app tables ...")
    matched_app_install_ids, matched_app_usage_ids = fetch_app_ids(dir, all_tag_ids, app_ids_file)
    print("matched_app_install_ids:", matched_app_install_ids, len(matched_app_install_ids))
    print("matched_app_usage_ids:", matched_app_usage_ids, len(matched_app_usage_ids))
    # data = pd.read_csv(dir + data_file)
    app_install_df = data[matched_app_install_ids]
    app_usage_df = data[matched_app_usage_ids]
    tokens = app_ids_file.split(".")[0].split("_")[0:-1]
    file_name = ""
    for t in tokens:
        file_name = file_name + t + "_"
    app_install_file_path = to_dir + file_name + "install.csv"
    app_usage_file_path = to_dir + file_name + "usage.csv"
    app_install_df.to_csv(app_install_file_path, index=False)
    print(f" ==> save app_install table with shape:{app_install_df.shape} to :{app_install_file_path}")
    app_usage_df.to_csv(app_usage_file_path, index=False)
    print(f" ==> save app_usage table with shape:{app_usage_df.shape} to :{app_usage_file_path}")


def create_equipment_table(resource_dir, to_dir, all_tag_ids, data):
    print(f" ==> creating equipment table ...")
    matched_equipment_ids = fetch_equipment_ids(resource_dir, all_tag_ids)
    # data = pd.read_csv(dir + data_file)
    matched_equipment_df = data[matched_equipment_ids]
    file_path = to_dir + 'equipment.csv'
    matched_equipment_df.to_csv(file_path, index=False)
    print(f" ==> save equipment table with shape:{matched_equipment_df.shape} to :{file_path}")


def create_demographics_table(resource_dir, to_dir, all_tag_ids, data):
    print(f" ==> creating demographics table ...")
    id_mean_v5_df = pd.read_csv(resource_dir + 'ID_meaning_v5.csv')
    id2name_dict, id2type1_dict, id2type2_dict = get_tag_id2info_dicts(id_mean_v5_df)

    # get mapping from id to desc for app
    result = get_tag_id2info_by_types(id2type2_dict, id2name_dict, ['人口属性', '社会属性', '资产', '资产分属性'])
    _, _, demo_ids = result
    print(f"demo_ids:{demo_ids}")
    matched_demo_ids = fetch_tag_ids_by_id_prefix(all_tag_ids, demo_ids)
    print(f"matched_demo_ids:{matched_demo_ids} with length:{len(matched_demo_ids)}")
    # data = pd.read_csv(dir + data_file)
    demo_df = data[matched_demo_ids]
    file_path = to_dir + 'demographics.csv'
    demo_df.to_csv(file_path, index=False)
    print(f" ==> save demographics table with shape:{demo_df.shape} to :{file_path}")


def create_fraud_table(resource_dir, to_dir, all_tag_ids, data):
    print(f" ==> creating fraud table ...")
    id_mean_v5_df = pd.read_csv(resource_dir + 'ID_meaning_v5.csv')
    id2desc_dict, id2type1_dict, id2type2_dict = get_tag_id2info_dicts(id_mean_v5_df)

    # get mapping from id to desc for app
    result = get_tag_id2info_by_types(id2type2_dict, id2desc_dict, ['v1版反欺诈分', 'v2反欺诈分', 'v3反欺诈分', 'v3版反欺诈分底层特征'])
    _, _, fraud_ids = result
    saved_fraudv3_wifi_ids_df = pd.read_csv(resource_dir + "fraudv3_wifi_ids.csv")
    fraud_v3_wifi_ids = [str(val) for val in saved_fraudv3_wifi_ids_df.values.flatten()]
    fraud_ids = fraud_ids + fraud_v3_wifi_ids
    print(f"fraud_ids:{fraud_ids}, {len(fraud_ids)}")
    matched_fraud_ids = fetch_tag_ids_by_id_prefix(all_tag_ids, fraud_ids)
    print(f"matched_fraud_ids:{matched_fraud_ids} with length:{len(matched_fraud_ids)}")
    fraud_df = data[matched_fraud_ids]
    file_path = to_dir + 'fraud.csv'
    fraud_df.to_csv(file_path, index=False)
    print(f" ==> save fraud table with shape:{fraud_df.shape} to :{file_path}")


def create_tgi_table(resource_dir, to_dir, all_tag_ids, data):
    print(f" ==> creating tgi table ...")

    saved_tgi_ids_df = pd.read_csv(resource_dir + "tgi_ids.csv")
    tgi_ids = [str(val) for val in saved_tgi_ids_df.values.flatten()]

    matched_tgi_ids = fetch_tag_ids_by_id_prefix(all_tag_ids, tgi_ids)
    print(f"matched_tgi_ids:{matched_tgi_ids} with length:{len(matched_tgi_ids)}")
    tgi_df = data[matched_tgi_ids]
    file_path = to_dir + 'tgi.csv'
    tgi_df.to_csv(file_path, index=False)
    print(f" ==> save tgi table with shape:{tgi_df.shape} to :{file_path}")


def create_risk_table(resource_dir, to_dir, all_tag_ids, data):
    print(f" ==> creating risk table ...")
    id_mean_v5_df = pd.read_csv(resource_dir + 'ID_meaning_v5.csv')
    id2desc_dict, id2type1_dict, id2type2_dict = get_tag_id2info_dicts(id_mean_v5_df)

    # get mapping from id to desc for app
    result = get_tag_id2info_by_types(id2type2_dict, id2desc_dict, ['贷款', '高危', '逾期类', '贷款行为', '高危行为'])
    _, _, risk_ids = result
    print(f"risk_ids:{risk_ids}")
    matched_risk_ids = fetch_tag_ids_by_id_prefix(all_tag_ids, risk_ids)
    print(f"matched_risk_ids:{matched_risk_ids} with length:{len(matched_risk_ids)}")
    risk_df = data[matched_risk_ids]
    file_path = to_dir + 'risk.csv'
    risk_df.to_csv(file_path, index=False)
    print(f" ==> save risk table with shape:{risk_df.shape} to :{file_path}")


def create_cell_number_appears_table(resource_dir, to_dir, all_tag_ids, data):
    print(f" ==> creating cell_number_appears table ...")
    id_mean_v5_df = pd.read_csv(resource_dir + 'ID_meaning_v5.csv')
    id2desc_dict, id2type1_dict, id2type2_dict = get_tag_id2info_dicts(id_mean_v5_df)
    # print(id2type2_dict)

    # get mapping from id to desc for app
    result = get_tag_id2info_by_types(id2type2_dict, id2desc_dict, ['号码出现'])
    _, _, num_appears_ids = result
    print(f"num_appears_ids:{num_appears_ids}")
    matched_num_appears_ids = fetch_tag_ids_by_id_prefix(all_tag_ids, num_appears_ids)
    print(f"matched_num_appears_ids:{matched_num_appears_ids} with length:{len(matched_num_appears_ids)}")
    appears_df = data[matched_num_appears_ids]
    file_path = to_dir + 'cell_number_appears.csv'
    appears_df.to_csv(file_path, index=False)
    print(f" ==> save cell number appears table with shape:{appears_df.shape} to :{file_path}")


def get_all_tag_ids(directory):
    cols_df = pd.read_csv(directory + 'A_train_col.txt', header=None)
    all_tag_ids = []
    tags = cols_df.values.flatten()
    print(f"shape:{tags.shape}")
    for tag in tags:
        all_tag_ids.append(str(tag))
    return all_tag_ids


def create_data_tables(resource_dir, to_dir, all_tag_ids, data: pd.DataFrame, target_col_name="target"):
    create_demographics_table(resource_dir, to_dir, all_tag_ids, data)
    create_equipment_table(resource_dir, to_dir, all_tag_ids, data)
    create_risk_table(resource_dir, to_dir, all_tag_ids, data)
    create_cell_number_appears_table(resource_dir, to_dir, all_tag_ids, data)
    create_fraud_table(resource_dir, to_dir, all_tag_ids, data)
    create_tgi_table(resource_dir, to_dir, all_tag_ids, data)
    create_app_table(resource_dir, to_dir, all_tag_ids, data, app_ids_file="app_finance_ids.txt")
    create_app_table(resource_dir, to_dir, all_tag_ids, data, app_ids_file="app_life_ids.txt")
    create_target_table(to_dir, data, target_col_name)


def create_target_table(to_dir, data, target_col_name):
    print(f" ==> creating target table ...")
    target_df = data[target_col_name]
    target_df = pd.DataFrame(data=target_df.values, columns=['target'])
    file_path = to_dir + 'target.csv'
    target_df.to_csv(file_path, index=False)
    print(f" ==> save target table with shape:{target_df.shape} to :{file_path}")


def create_data_tables_for_A_data(resource_dir, all_tag_ids):
    print("==> create data tables for A data ...")
    # data_to_split = pd.read_csv("../../../data/cell_manager/A_train.csv")
    data_to_split = pd.read_csv("../../../data/cell_manager/A_train_bal.csv")
    # to_dir = "../../../data/cell_manager/A_train_data/"
    to_dir = "../../../data/cell_manager/A_train_data_2/"
    create_data_tables(resource_dir, to_dir, all_tag_ids, data_to_split)


def create_data_tables_for_B_data(resource_dir, all_tag_ids):
    print("==> create data tables for B train data ...")
    # B_combine is the combination of B_train and B_test
    data_to_split = pd.read_csv("../../../data/cell_manager/B_combine.csv")
    # to_dir = "../../../data/cell_manager/B_train_data/"
    to_dir = "../../../data/cell_manager/B_train_data_2/"
    create_data_tables(resource_dir, to_dir, all_tag_ids, data_to_split)

    # print("==> create data tables for B test data ...")
    # data_to_split = pd.read_csv("../../../data/cell_manager/B_test.csv")
    # to_dir = "../../../data/cell_manager/B_test_data/"
    # create_data_tables(resource_dir, to_dir, all_col, data_to_split)


def create_data_tables_for_C_data(resource_dir, all_tag_ids):
    print("==> create data tables for C train data ...")
    # C_combine is the combination of C_train and C_test
    data_to_split = pd.read_csv("../../../data/cell_manager/C_combine.csv")
    print(f"C data shape:{data_to_split.shape}")
    print(f"C label 1 data shape:{data_to_split[data_to_split['y'] == 1].shape}")
    print(f"C label 0 data shape:{data_to_split[data_to_split['y'] == 0].shape}")
    to_dir = "../../../data/cell_manager/C_train_data_2/"
    create_data_tables(resource_dir, to_dir, all_tag_ids, data_to_split, target_col_name='y')


def create_data_tables_for_A_B_C(resource_dir):
    all_tag_id_list = get_all_tag_ids(resource_dir)
    # create_data_tables_for_A_data(resource_dir, all_tag_id_list)
    # create_data_tables_for_B_data(resource_dir, all_tag_id_list)
    create_data_tables_for_C_data(resource_dir, all_tag_id_list)


if __name__ == "__main__":
    resource_dir = "../../../data/cell_manager/"
    create_data_tables_for_A_B_C(resource_dir)
