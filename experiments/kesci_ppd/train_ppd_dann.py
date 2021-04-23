from collections import OrderedDict

from datasets.ppd_dataloader import get_datasets, get_dataloader
from experiments.kesci_ppd.meta_data import column_name_list, df_group_ind_list, df_cat_mask_list, df_group_index
from models.classifier import CensusRegionAggregator
from models.dann_models import create_embeddings
from models.discriminator import CensusRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.model_config import wire_global_model
from utils import get_timestamp


def store_domain_data(domain_data_dict, domain_data, domain_col_list, is_cat):
    if is_cat:
        emb_dict = OrderedDict()
        for col_index, col_name in enumerate(domain_col_list):
            emb_dict[col_name] = domain_data[:, col_index]
        domain_data_dict["embeddings"] = emb_dict

    else:
        domain_data_dict["non_embedding"] = {"tabular_data": domain_data}


def parse_domain_data(data, column_name_list, df_group_ind_list, df_cat_mask_list, df_group_index_list):
    wide_feat_list = []
    domain_list = []
    for group_ind, is_cat, group_index in zip(df_group_ind_list, df_cat_mask_list, df_group_index_list):
        start_index = group_index[0]
        length = group_index[1]
        # 2 stands for wide features
        if group_ind == 2:
            wide_feat_list.append(data[:, start_index:start_index + length])
        elif group_ind == 1:
            new_domain = dict()
            new_domain["embeddings"] = None
            new_domain["non_embedding"] = None
            domain_data = data[:, start_index:start_index + length]
            domain_col_list = column_name_list[start_index:start_index + length]
            store_domain_data(new_domain, domain_data, domain_col_list, is_cat)
            domain_list.append(new_domain)
        else:
            # print("group_ind", group_ind)
            assert group_ind == 0
            domain_dict = domain_list[-1]
            domain_data = data[:, start_index:start_index + length]
            domain_col_list = column_name_list[start_index:start_index + length]
            store_domain_data(domain_dict, domain_data, domain_col_list, is_cat)
    return wide_feat_list, domain_list


def partition_data(data):
    wide_feat_list, domain_list = parse_domain_data(data,
                                                    column_name_list,
                                                    df_group_ind_list,
                                                    df_cat_mask_list,
                                                    df_group_index)
    return wide_feat_list, domain_list


def create_model_group(extractor_input_dim):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dim)
    classifier = CensusRegionAggregator(input_dim=extractor_input_dim[-1])
    discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dim[-1])
    return extractor, classifier, discriminator


def create_embedding_dict():
    embedding_map = {'UserInfo_13': (3, 3),
                     'UserInfo_12': (3, 3),
                     'UserInfo_22': (9, 9),
                     'UserInfo_17': (2, 2),
                     'UserInfo_21': (2, 2),
                     'UserInfo_5': (3, 3),
                     'UserInfo_3': (9, 9),
                     'UserInfo_11': (3, 3),
                     'UserInfo_16': (6, 6),
                     'UserInfo_1': (9, 9),
                     'UserInfo_6': (3, 3),
                     'UserInfo_23': (31, 15),
                     'UserInfo_9': (4, 4),
                     'Education_Info3': (3, 3),
                     'Education_Info5': (2, 2),
                     'Education_Info1': (2, 2),
                     'Education_Info7': (2, 2),
                     'Education_Info6': (6, 6),
                     'Education_Info8': (7, 7),
                     'Education_Info2': (7, 7),
                     'Education_Info4': (6, 6),
                     'SocialNetwork_2': (3, 3),
                     'SocialNetwork_7': (3, 3),
                     'SocialNetwork_12': (3, 3),
                     'WeblogInfo_19': (8, 8),
                     'WeblogInfo_21': (5, 5),
                     'WeblogInfo_20': (45, 15)}
    embedding_meta_dict = dict()
    for feat_name, emb_shape in embedding_map.items():
        embedding_meta_dict[feat_name] = emb_shape
    print(f"embedding_meta_dict: \n {embedding_meta_dict}")
    return create_embeddings(embedding_meta_dict)


def compute_feature_group_interaction(input_dims_list):
    input_int_dims_list = [e for e in input_dims_list]
    start_index = 1
    for fg in input_dims_list:
        for i in range(start_index, len(input_dims_list)):
            fg_2 = input_dims_list[i]
            input_int_dims_list.append([v1 + v2 for v1, v2 in zip(fg, fg_2)])
        start_index += 1
    return input_int_dims_list


def create_pdd_global_model():
    embedding_dict = create_embedding_dict()
    print("embedding_dict", embedding_dict)

    # input_dims_list = [[15, 24, 15, 6],
    #                    [85, 120, 85, 10],
    #                    [35, 50, 35, 8],
    #                    [21, 30, 21, 8],
    #                    [58, 80, 58, 10]]
    # input_dims_list = [[15, 24, 15, 6],
    #                    [85, 120, 85, 10],
    #                    [35, 50, 35, 8],
    #                    [21, 30, 21, 8],
    #                    [58, 80, 58, 10]]
    input_dims_list = [[15, 24, 15, 6],
                       [85, 120, 85, 8],
                       [35, 50, 35, 8],
                       [21, 30, 21, 8],
                       [58, 80, 58, 8]]

    # num_wide_feature = 10
    num_wide_feature = 17
    using_feature_group = True
    using_interaction = True
    using_transform_matrix = True

    global_model = wire_global_model(embedding_dict=embedding_dict,
                                     input_dims_list=input_dims_list,
                                     num_wide_feature=num_wide_feature,
                                     using_feature_group=using_feature_group,
                                     using_interaction=using_interaction,
                                     using_transform_matrix=using_transform_matrix,
                                     partition_data_fn=partition_data,
                                     create_model_group_fn=create_model_group,
                                     pos_class_weight=1.0)

    return global_model


if __name__ == "__main__":
    # data_dir = "../../../Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data/"
    # import os
    #
    # dirpath = os.path.dirname(__file__)
    # print("dirpath", dirpath)

    exp_dir = "ppd_dann"
    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    source_train_file_name = data_dir + "PPD_2014_1to9_train.csv"
    target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    source_test_file_name = data_dir + 'PPD_2014_1to9_test.csv'
    target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'

    split_ratio = 1.0
    src_train_dataset, _ = get_datasets(ds_file_name=source_train_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_train_dataset, _ = get_datasets(ds_file_name=target_train_file_name, shuffle=True, split_ratio=split_ratio)
    src_test_dataset, _ = get_datasets(ds_file_name=source_test_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_test_dataset, _ = get_datasets(ds_file_name=target_test_file_name, shuffle=True, split_ratio=split_ratio)

    # batch_size_list = [512]
    batch_size_list = [256]
    learning_rate_list = [1.5e-3]
    # learning_rate_list = [1.2e-3]
    # learning_rate_list = [6e-4]
    # batch_size_list = [256, 512]
    # learning_rate_list = [3e-4, 8e-4, 1e-3]
    tries = 1
    pos_class_weight = 1.0
    param_comb_list = list()
    for lr in learning_rate_list:
        for bs in batch_size_list:
            param_comb_list.append((lr, bs))
    date = "20210423_PDD"

    # date = "20200118_PDD"; batch_size = 512; lr = 1e-3; version = 3
    # task_id = date + "_" + str(batch_size) + "_" + str(lr) + "_" + str(version)
    # exp_dir = "ppd_dann"

    timestamp = get_timestamp()
    for param_comb in param_comb_list:
        lr, bs = param_comb
        for version in range(tries):
            task_id = date + "_pw" + str(pos_class_weight) + "_bs" + str(bs) + "_lr" + str(lr) + "_v" + str(
                version) + "_t" + str(timestamp)
            print("[INFO] perform task:{0}".format(task_id))

            global_model = create_pdd_global_model()
            print("[INFO] model created.")

            src_train_loader = get_dataloader(src_train_dataset, batch_size=bs)
            tgt_train_loader = get_dataloader(tgt_train_dataset, batch_size=bs)
            print("[INFO] train data loaded.")

            src_test_loader = get_dataloader(src_test_dataset, batch_size=bs * 2)
            tgt_test_loader = get_dataloader(tgt_test_dataset, batch_size=bs * 2)
            print("[INFO] test data loaded.")

            plat = FederatedDAANLearner(model=global_model,
                                        source_train_loader=src_train_loader,
                                        source_val_loader=src_test_loader,
                                        target_train_loader=tgt_train_loader,
                                        target_val_loader=tgt_test_loader,
                                        max_epochs=400,
                                        epoch_patience=10)
            # wrapper.print_global_classifier_param()
            plat.set_model_save_info(exp_dir)
            plat.train_dann(epochs=120, lr=lr, task_id=task_id)

            global_model.print_parameters()
