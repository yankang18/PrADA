import pandas as pd


def get_all_tag_ids(directory):
    """

    :param directory:
    :return:
    """

    cols_df = pd.read_csv(directory + 'A_train_col.txt', header=None)
    all_tag_ids = []
    tags = cols_df.values.flatten()
    print(f"shape:{tags.shape}")
    for tag in tags:
        all_tag_ids.append(str(tag))
    return all_tag_ids


def get_tag_id2info_dicts(tag_meaning_df):
    """
        get mapping from tag ids to tag type and from tag ids to tag description
    :param tag_meaning_df:
    :return:
    """

    id2name_dict = dict()
    id2type1_dict = dict()
    id2type2_dict = dict()
    for index, row in tag_meaning_df.iterrows():
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


def fetch_tag_ids_with_prefix(all_tag_id, id_prefix_list):
    matched_tag_id = set()
    for tag_id in all_tag_id:
        id_prefix = tag_id[:10]
        # print(tag_id)
        if id_prefix in id_prefix_list:
            # print("add:", tag_id)
            matched_tag_id.add(tag_id)
    return matched_tag_id


def get_ids_via_meaning_table(id_mean_v5_df, type2_list):
    # id_mean_v5_df = pd.read_csv(resource_dir + 'ID_meaning_v5.csv')
    id2name_dict, id2type1_dict, id2type2_dict = get_tag_id2info_dicts(id_mean_v5_df)
    _, _, ids = get_tag_id2info_by_types(id2type2_dict, id2name_dict, type2_list)
    return ids


def get_ids_via_specific_table(resource_dir, file_name):
    prefix_ids_df = pd.read_csv(resource_dir + file_name, header=None)
    prefix_ids = [str(val) for val in prefix_ids_df.values.flatten()]
    return prefix_ids


def save_df_tables(named_tables, to_dir):
    for table_name, df_table in named_tables:
        file_name = table_name + '.csv'
        save_df_table(to_dir, file_name, df_table)


def save_df_table(to_dir, file_name, data_df):
    file_path = to_dir + file_name
    data_df.to_csv(file_path, index=False)
    table_name = file_name.split('.')[0]
    print(f" ==> save {table_name} table with shape:{data_df.shape} to :{file_path}")


def create_tables(feature_group_list, resource_dir, id_mean_v5_df, all_tag_ids, data: pd.DataFrame,
                  target_col_name="target"):
    # create feature group tables
    feature_group_df_list = create_feature_group_tables(feature_group_list, resource_dir, id_mean_v5_df, all_tag_ids,
                                                        data)

    # create target table
    target_df = create_target_table(data, target_col_name)

    return feature_group_df_list, target_df


def create_target_table(data, target_col_name):
    print(f" ==> creating target table:{target_col_name}")
    target_df = data[target_col_name]
    target_df = pd.DataFrame(data=target_df.values, columns=['target'])
    return target_df


def create_feature_group_tables(feature_group_list, resource_dir, id_mean_v5_df, all_tag_ids, data_df):
    df_list = list()
    for feature_group_tuple in feature_group_list:
        feature_group_name = feature_group_tuple[0]
        type2_list = feature_group_tuple[1]
        ids_file_name = feature_group_tuple[2]

        print(f"=> create feature group table:{feature_group_name}")
        feat_group_df = create_feature_group_table(resource_dir, id_mean_v5_df, all_tag_ids, data_df,
                                                   type2_list=type2_list,
                                                   id_file_name=ids_file_name)
        df_list.append(feat_group_df)
    return df_list


def create_feature_group_table(resource_dir, id_mean_v5_df, all_tag_ids, data, type2_list, id_file_name):
    ids = list()
    if type2_list is not None and len(type2_list) > 0:
        ids += get_ids_via_meaning_table(id_mean_v5_df, type2_list)
    if id_file_name is not None:
        ids += get_ids_via_specific_table(resource_dir, id_file_name)

    matched_ids = list(fetch_tag_ids_with_prefix(all_tag_ids, id_prefix_list=ids))
    print(f"matched_ids:{matched_ids} with length:{len(matched_ids)}")

    data_df = data[matched_ids]
    return data_df


def create_named_data_tables(table_dir, to_dir, table_names, id2name_dict):
    for table_name in table_names:
        df = pd.read_csv(table_dir + table_name)
        print(f"[INFO ]{table_name} table shape: {df.shape}")
        named_df = df.rename(columns=id2name_dict)
        named_df_file_path = to_dir + table_name
        named_df.to_csv(named_df_file_path, index=False)
        print(f"[INFO] save named {table_name} to: {named_df_file_path}")
