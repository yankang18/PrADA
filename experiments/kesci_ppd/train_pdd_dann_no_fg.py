from collections import OrderedDict

from datasets.pdd_dataloader import get_pdd_dataloaders_ob
from experiments.kesci_ppd.meta_data import column_name_list, df_group_ind_list, df_cat_mask_list, df_group_index
from models.classifier import GlobalClassifier, CensusRegionAggregator, IdentityRegionAggregator
from models.dann_models import GlobalModel, RegionalModel, create_embeddings
from models.discriminator import CensusRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.discriminator import LendingRegionDiscriminator
import numpy as np
import torch


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


def aggregate_domains(domain_list):
    agg_domain = dict({'embeddings': None, 'non_embedding': dict()})
    agg_embed_dict = dict()
    non_embed_list = []
    for domain in domain_list:
        embed_dict = domain['embeddings']
        if embed_dict:
            agg_embed_dict.update(embed_dict)
        non_embed = domain['non_embedding']
        if non_embed:
            non_embed_list.append(non_embed['tabular_data'])

    agg_domain['embeddings'] = agg_embed_dict
    agg_domain['non_embedding']['tabular_data'] = torch.cat(non_embed_list, dim=1)
    return agg_domain


def partition_data(data):
    wide_feat_list, domain_list = parse_domain_data(data,
                                                    column_name_list,
                                                    df_group_ind_list,
                                                    df_cat_mask_list,
                                                    df_group_index)
    agg_domain = aggregate_domains(domain_list)
    return wide_feat_list, [agg_domain]


def create_region_model_wrapper(extractor_input_dims_list):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims_list)
    region_aggregator = IdentityRegionAggregator()
    discriminator = LendingRegionDiscriminator(input_dim=extractor_input_dims_list[-1])
    return RegionalModel(extractor=extractor,
                         aggregator=region_aggregator,
                         discriminator=discriminator)


def create_region_model_wrappers(input_dims_list):
    wrapper_list = list()
    for input_dim in input_dims_list:
        wrapper_list.append(create_region_model_wrapper(input_dim))
    return wrapper_list


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


def create_global_model_wrapper(pos_class_weight=1.0):
    embedding_dict = create_embedding_dict()
    print("embedding_dict", embedding_dict)
    # input_dims_list = [[285, 404, 285, 52]]
    input_dims_list = [[214, 304, 214, 42]]
    # global_input_dim = 56
    global_input_dim = 59

    region_wrapper_list = create_region_model_wrappers(input_dims_list)

    classifier = GlobalClassifier(input_dim=global_input_dim)
    wrapper = GlobalModel(classifier, region_wrapper_list, embedding_dict, partition_data,
                          pos_class_weight=pos_class_weight, loss_name="BCE")
    return wrapper


if __name__ == "__main__":
    pos_class_weight = 1.0
    wrapper = create_global_model_wrapper(pos_class_weight=pos_class_weight)
    # data_dir = "../../../Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data/"
    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    source_train_file_name = data_dir + "PPD_2014_1to9_train.csv"
    target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    source_test_file_name = data_dir + 'PPD_2014_1to9_test.csv'
    target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'
    batch_size = 512
    # batch_size = 256

    print("[INFO] Load train data")
    source_train_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=source_train_file_name, batch_size=batch_size, split_ratio=1.0)
    target_train_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    source_valid_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=source_test_file_name, batch_size=batch_size * 2, split_ratio=1.0)
    target_valid_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_test_file_name, batch_size=batch_size * 2, split_ratio=1.0)

    plat = FederatedDAANLearner(model=wrapper,
                                source_train_loader=source_train_loader,
                                source_val_loader=source_valid_loader,
                                target_train_loader=target_train_loader,
                                target_val_loader=target_valid_loader,
                                max_epochs=400,
                                epoch_patience=5)

    # wrapper.print_global_classifier_param()
    plat.set_model_save_info("ppd_dann")
    # plat.set_model_save_info("lending_dann")
    version = 2
    lr = 1e-3
    # lr = 8e-4
    tag = "20200122_no_fg"
    task_id = tag + "_pw" + str(pos_class_weight) + "_bs" + str(batch_size) + "_lr" + str(lr) + "_v" + str(version)
    plat.train_dann(epochs=100, lr=lr, task_id=task_id)

    wrapper.print_parameters()
