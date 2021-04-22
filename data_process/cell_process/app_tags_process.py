import pandas as pd

from data_process.cell_process.create_data_table_util import get_tag_id2info_by_types, get_tag_id2info_dicts


def create_maps_between_app_install_usage_(app_install_ids, app_usage_ids):
    """
    create mapping from app install ids to app usage ids, and vice versa.
    """
    install_id2usage_id_dict = dict()
    usage_id2install_id_dict = dict()
    for install_id, usage_id in zip(app_install_ids, app_usage_ids):
        install_id_str = str(install_id)
        usage_id_str = str(usage_id)
        install_id2usage_id_dict[install_id_str] = usage_id_str
        usage_id2install_id_dict[usage_id_str] = install_id_str
    return install_id2usage_id_dict, usage_id2install_id_dict


def get_paired_id2info_dict(id2info_dict, id2paired_id_dict):
    """

    :param id2info_dict:
    :param id2paired_id_dict:
    :return:
    """

    pair_id2info_dict = dict()
    for id, info in id2info_dict.items():
        pair_id2info_dict[id2paired_id_dict[id]] = info
    return pair_id2info_dict


if __name__ == "__main__":
    app_df = pd.read_csv('../../../data/cell_manager/app.txt', header=None)
    app_install_ids = [str(val) for val in app_df.iloc[0].values]
    app_usage_ids = [str(val) for val in app_df.iloc[1].values]
    print(f"num of install app:{len(app_install_ids)}, num of usage app:{len(app_usage_ids)}")
    install_id2usage_id_dict, usage_id2install_id_dict = create_maps_between_app_install_usage_(app_install_ids,
                                                                                                app_usage_ids)

    id_mean_v5_df = pd.read_csv('../../../data/cell_manager/ID_meaning_v5.csv')
    id2name_dict, id2type1_dict, id2type2_dict = get_tag_id2info_dicts(id_mean_v5_df)

    # get mapping from id to name for tag with type1 in type1_list

    type1_list = ['app安装']
    result = get_tag_id2info_by_types(id2type_dict=id2type1_dict, id2info_dict=id2name_dict, type_list=type1_list)
    app_install_id2name_dict, _, _ = result

    type1_list = ['app使用']
    result = get_tag_id2info_by_types(id2type_dict=id2type1_dict, id2info_dict=id2name_dict, type_list=type1_list)
    app_usage_id2name_dict, _, _ = result

    print(f"num of install app:{len(app_install_id2name_dict)}, num of usage app:{len(app_usage_id2name_dict)}")

    # get mapping from id to type2 for tag with type1 in type1_list

    type1_list = ['app安装']
    result = get_tag_id2info_by_types(id2type_dict=id2type1_dict, id2info_dict=id2type2_dict, type_list=type1_list)
    app_install_id2type2_dict, _, _ = result

    type1_list = ['app使用']
    result = get_tag_id2info_by_types(id2type_dict=id2type1_dict, id2info_dict=id2type2_dict, type_list=type1_list)
    app_usage_id2type2_dict, _, _ = result

    #
    paired_app_usage_id2name_dict = get_paired_id2info_dict(app_install_id2name_dict, install_id2usage_id_dict)
    paired_app_install_id2name_dict = get_paired_id2info_dict(app_usage_id2name_dict, usage_id2install_id_dict)

    paired_app_usage_id2type2_dict = get_paired_id2info_dict(app_install_id2type2_dict, install_id2usage_id_dict)
    paired_app_install_id2type2_dict = get_paired_id2info_dict(app_usage_id2type2_dict, usage_id2install_id_dict)

    # paired_app_usage_id2name_dict.update(app_usage_id2name_dict)
    # print(paired_app_usage_id2name_dict, len(paired_app_usage_id2name_dict))
    #
    # app_install_id2name_dict.update(paired_app_install_id2name_dict)
    # print(app_install_id2name_dict, len(app_install_id2name_dict))

    #
    paired_app_usage_id2type2_dict.update(app_usage_id2type2_dict)
    print(paired_app_usage_id2type2_dict, len(paired_app_usage_id2type2_dict))
    print(paired_app_usage_id2type2_dict.keys())

    # part_1 = ["理财"]
    # part_2 = ["汽车服务", "旅游出行"]
    # part_3 = ["消费", "娱乐", "社交", "咨询", "医疗"]
    # part_4 = ["家庭", "自我成长", "商务"]
    # finance = ['消费', '理财', '汽车服务', '家庭', '医疗']
    # life = ['旅游出行', '商务', '娱乐', '通讯', '功能', '咨询', '社交', '自我成长']
    finance = ['消费', '理财', '汽车服务']
    life = ['旅游出行', '商务', '娱乐', '通讯', '功能', '咨询', '社交', '自我成长', '家庭', '医疗']
    start_finance = ""
    start_life = ""
    start_life_count = 0
    start_finance_count = 0
    for key, type in paired_app_usage_id2type2_dict.items():
        if type in finance:
            start_finance = start_finance + key + ","
            start_finance_count += 1

        if type in life:
            start_life = start_life + key + ","
            start_life_count += 1
    print("app finance usage ids: ", start_finance, start_finance_count)
    print("app life usage ids: ", start_life, start_life_count)

    app_install_id2type2_dict.update(paired_app_install_id2type2_dict)
    print(app_install_id2type2_dict, len(app_install_id2type2_dict))
    print(app_install_id2type2_dict.keys())

    start_finance = ""
    start_life = ""
    start_life_count = 0
    start_finance_count = 0
    for key, type in app_install_id2type2_dict.items():
        if type in finance:
            start_finance = start_finance + key + ","
            start_finance_count += 1

        if type in life:
            start_life = start_life + key + ","
            start_life_count += 1
    print("app finance install ids: ", start_finance, start_finance_count)
    print("app life install ids: ", start_life, start_life_count)

    type2_count_dict = dict()
    for key, val in paired_app_usage_id2type2_dict.items():
        count = type2_count_dict.get(val)
        if count is None:
            type2_count_dict[val] = 1
        else:
            type2_count_dict[val] = count + 1
    print(type2_count_dict, len(type2_count_dict))
