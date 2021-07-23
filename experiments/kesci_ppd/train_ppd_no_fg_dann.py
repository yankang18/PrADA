from collections import OrderedDict

import numpy as np
import torch

from datasets.ppd_dataloader import get_pdd_dataloaders_ob
from experiments.kesci_ppd.global_config import feature_extractor_architecture_list
from experiments.kesci_ppd.meta_data import column_name_list, group_ind_list, group_info
from experiments.kesci_ppd.train_ppd_fg_dann import parse_domain_data, create_embedding_dict
from models.classifier import GlobalClassifier, CensusFeatureAggregator
from models.dann_models import GlobalModel, RegionalModel
from models.discriminator import LendingRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from utils import get_timestamp, get_current_date


def store_domain_data(domain_data_dict, domain_data, domain_col_list, is_cat):
    if is_cat:
        emb_dict = OrderedDict()
        for col_index, col_name in enumerate(domain_col_list):
            emb_dict[col_name] = domain_data[:, col_index]
        domain_data_dict["embeddings"] = emb_dict

    else:
        domain_data_dict["non_embedding"] = {"tabular_data": domain_data}


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
                                                    group_ind_list,
                                                    group_info)
    agg_domain = aggregate_domains(domain_list)
    return wide_feat_list, [agg_domain]


def create_region_model(extractor_input_dims_list, aggregation_dim):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims_list)
    aggregator = CensusFeatureAggregator(input_dim=extractor_input_dims_list[-1], output_dim=aggregation_dim)
    discriminator = LendingRegionDiscriminator(input_dim=extractor_input_dims_list[-1])
    return RegionalModel(extractor=extractor,
                         aggregator=aggregator,
                         discriminator=discriminator)


def create_region_model_list(feature_extractor_arch_list, aggregation_dim):
    model_list = list()
    for feature_extractor_arch in feature_extractor_arch_list:
        model_list.append(create_region_model(feature_extractor_arch, aggregation_dim))
    return model_list


def create_global_model(aggregation_dim, num_wide_feature, pos_class_weight=1.0):
    embedding_dict = create_embedding_dict()
    print("embedding_dict", embedding_dict)

    # feature_extractor_architecture = list(np.sum(np.array(feature_extractor_architecture_list), axis=0))
    feature_extractor_architecture = [203, 210, 70, 20]
    print(f"[INFO] feature_extractor_architecture list:{[feature_extractor_architecture]}")

    region_wrapper_list = create_region_model_list([feature_extractor_architecture], aggregation_dim)
    global_input_dim = aggregation_dim + num_wide_feature
    print(f"[INFO] global_input_dim: {global_input_dim}")
    source_classifier = GlobalClassifier(input_dim=global_input_dim)
    target_classifier = GlobalClassifier(input_dim=global_input_dim)
    wrapper = GlobalModel(source_classifier=source_classifier,
                          target_classifier=target_classifier,
                          regional_model_list=region_wrapper_list,
                          embedding_dict=embedding_dict,
                          partition_data_fn=partition_data,
                          pos_class_weight=pos_class_weight,
                          loss_name="BCE")
    return wrapper


if __name__ == "__main__":

    # hyper-parameters
    apply_target_classification = False
    monitor_source = False
    momentum = 0.99
    weight_decay = 0.0
    # weight_decay = 0.00001
    apply_global_domain_adaption = False
    global_domain_adaption_lambda = 1.0
    tgt_lr = 5e-4
    lr = 5e-4
    batch_size = 64
    src_batch_size = batch_size
    tgt_batch_size = batch_size

    # num_tgt_clz_train_iter = 1
    num_tgt_clz_train_iter = None

    epoch_patience = 2
    max_epochs = 600

    date = get_current_date()
    timestamp = get_timestamp()

    pos_class_weight = 3.0
    metrics = ('ks', 'auc')
    ppd_no_fg_dann_dir = "ppd_no_fg_dann"

    # initialize model
    model = create_global_model(aggregation_dim=5,
                                num_wide_feature=6,
                                pos_class_weight=pos_class_weight)

    # load data
    ts = '20210522'
    data_tag = 'lbl004tgt4000v4'
    # data_tag = 'lbl002tgt4000'
    # data_tag = 'lbl001tgt4000'
    data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_{ts}/"
    source_train_file_name = data_dir + f"PPD_2014_src_1to9_da_{data_tag}_train.csv"
    source_test_file_name = data_dir + f'PPD_2014_src_1to9_da_{data_tag}_test.csv'
    target_train_file_name = data_dir + f'PPD_2014_tgt_10to12_da_{data_tag}_train.csv'
    target_test_file_name = data_dir + f'PPD_2014_tgt_10to12_ft_{data_tag}_test.csv'

    print(f"load source train from: {source_train_file_name}.")
    print(f"load target train from: {target_train_file_name}.")
    print(f"load source test from: {source_test_file_name}.")
    print(f"load target test from: {target_test_file_name}.")

    # data_dir = f"/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_output_1620085151/"
    # source_train_file_name = data_dir + f"PPD_2014_src_1to9_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_train.csv"
    # source_test_file_name = data_dir + f'PPD_2014_src_1to9_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_test.csv'
    # target_train_file_name = data_dir + f'PPD_2014_tgt_10to11_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_train.csv'
    # target_test_file_name = data_dir + f'PPD_2014_tgt_10to11_{tgt_data_tag}_{src_data_tag}_{tgt_tag}_test.csv'

    print("[INFO] Load train data")
    source_train_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=source_train_file_name, batch_size=src_batch_size, split_ratio=1.0)
    target_train_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_train_file_name, batch_size=src_batch_size, split_ratio=1.0)
    target_clz_train_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_train_file_name, batch_size=tgt_batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    source_valid_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=source_test_file_name, batch_size=src_batch_size * 4, split_ratio=1.0)
    target_valid_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_test_file_name, batch_size=src_batch_size * 4, split_ratio=1.0)

    plat = FederatedDAANLearner(model=model,
                                source_da_train_loader=source_train_loader,
                                source_val_loader=source_valid_loader,
                                target_da_train_loader=target_train_loader,
                                target_val_loader=target_valid_loader,
                                target_clz_train_loader=target_clz_train_loader,
                                max_epochs=max_epochs,
                                epoch_patience=epoch_patience)
    plat.set_model_save_info(ppd_no_fg_dann_dir)

    task_tag = date + "_no_fg"
    task_id = task_tag + "_" + data_tag + "_pw" + str(
        pos_class_weight) + "_bs" + str(src_batch_size) + "_lr" + str(lr) + "_v" + str(timestamp)

    optimizer_param_dict = {"src": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay},
                            "tgt": {"lr": tgt_lr, "momentum": momentum, "weight_decay": weight_decay}}
    plat.train_dann(epochs=100,
                    task_id=task_id,
                    metric=metrics,
                    apply_global_domain_adaption=apply_global_domain_adaption,
                    global_domain_adaption_lambda=global_domain_adaption_lambda,
                    use_target_classifier=apply_target_classification,
                    optimizer_param_dict=optimizer_param_dict,
                    monitor_source=monitor_source,
                    num_tgt_clz_train_iter=num_tgt_clz_train_iter)

    model.print_parameters()
