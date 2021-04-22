from collections import OrderedDict

import torch

from models.classifier import GlobalClassifier, IdentityRegionAggregator
from datasets.lending_dataloader import get_lending_dataloader
from models.discriminator import LendingRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from experiments.lending_loan.train_lending_dann import create_embeddings
from models.dann_models import GlobalModel, RegionalModel


def create_embedding_dict():
    COLUMNS = {'home_ownership': (4, 3), 'purpose': (4, 3)}

    embedding_meta_dict = dict()
    for col, num_values in COLUMNS.items():
        embedding_meta_dict[col] = (num_values[0], num_values[1])
    print(f"[INFO] Create embedding_meta_dict: \t {embedding_meta_dict}")
    return create_embeddings(embedding_meta_dict)


def partition_data(data):

    wide_data = data[0]
    wide_feat = [wide_data[:, 0].reshape(-1, 1),
                 wide_data[:, 1].reshape(-1, 1),
                 wide_data[:, 2].reshape(-1, 1),
                 wide_data[:, 3].reshape(-1, 1)]

    combine_feat = []
    for idx in range(1, len(data) - 1):
        combine_feat.append(data[idx])

    # qualify_feat = [
    #     'grade',
    #     'emp_length',
    #     'home_ownership',
    #     'verification_status',
    #     'annual_inc_comp',
    #     'purpose',
    #     'application_type',
    #     'disbursement_method'
    # ]
    qualify_data = data[6]
    embedding_feat = OrderedDict({"home_ownership": qualify_data[:, 0],
                                  "purpose": qualify_data[:, 1]})
    combine_feat.append(qualify_data[:, 2:])

    deep_feat = list()
    cat_data = torch.cat(combine_feat, dim=1)
    # print(f"cat_data shape:{cat_data.shape}")
    feat = {"embeddings": embedding_feat,
            "non_embedding": cat_data}
    deep_feat.append(feat)
    return wide_feat, deep_feat


def create_region_model_wrapper(extractor_input_dims_list):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims_list)
    region_aggregator = IdentityRegionAggregator()
    discriminator = LendingRegionDiscriminator(input_dim=extractor_input_dims_list[-1])
    return RegionalModel(extractor=extractor,
                         aggregator=region_aggregator,
                         discriminator=discriminator)


def create_region_model_wrappers(input_dims_list):
    wrapper_list = list()
    for input_dims in input_dims_list:
        wrapper_list.append(create_region_model_wrapper(input_dims))
    return wrapper_list


def create_global_model_wrapper(pos_class_weight=1.0, beta=1.0):
    # embedding_dict = dict()
    embedding_dict = create_embedding_dict()
    # input_dims_list = [[24, 32, 24, 6],
    #                    [17, 24, 17, 6],
    #                    [41, 50, 41, 8],
    #                    [17, 24, 17, 6],
    #                    [16, 24, 16, 6],
    #                    [12, 20, 11, 4]]

    # input_dims_list = [[127, 150, 100, 26]]
    input_dims_list = [[127, 174, 127, 36]]
    # input_dims_list = [[127, 200, 150, 36]]
    # input_dims_list = [[127, 200, 150, 80, 36]]
    # global_input_dim = 30
    global_input_dim = 40

    region_wrapper_list = create_region_model_wrappers(input_dims_list=input_dims_list)

    print(f"[INFO] global_input_dim:{global_input_dim}\t beta:{beta}\t pos_class_weight:{pos_class_weight}")
    classifier = GlobalClassifier(input_dim=global_input_dim)
    wrapper = GlobalModel(classifier, region_wrapper_list, embedding_dict, partition_data_fn=partition_data,
                          beta=beta, pos_class_weight=pos_class_weight, loss_name="BCE")
    return wrapper


if __name__ == "__main__":
    wrapper = create_global_model_wrapper(pos_class_weight=1.0, beta=1.0)
    print("[INFO] model created ...")

    # loan_201517_dir = "../../data/lending_club_bundle_archive/loan_processed_2015_18/"
    # loan_2018_dir = "../../data/lending_club_bundle_archive/loan_processed_2018/"
    # loan_201517_dir = "../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2015_17/"
    src_dir = "../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2016_17/"
    loan_2018_dir = "../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"

    batch_size = 512
    print(f"[INFO] load loan_201617 data from {src_dir}")
    src_train_loader = get_lending_dataloader(dir=src_dir, batch_size=batch_size, data_mode="train")
    src_test_loader = get_lending_dataloader(dir=src_dir, batch_size=batch_size * 2, data_mode="test")

    print(f"[INFO] load loan_2018 data from {loan_2018_dir}")
    tgt_train_loader = get_lending_dataloader(dir=loan_2018_dir, batch_size=batch_size, data_mode="train")
    tgt_val_loader = get_lending_dataloader(dir=loan_2018_dir, batch_size=batch_size * 2, data_mode="val")
    tgt_test_loader = get_lending_dataloader(dir=loan_2018_dir, batch_size=batch_size * 2, data_mode="test")
    print("[INFO] data loaded ...")

    plat = FederatedDAANLearner(model=wrapper,
                                source_train_loader=src_train_loader,
                                source_val_loader=src_test_loader,
                                target_train_loader=tgt_train_loader,
                                target_val_loader=tgt_test_loader,
                                max_epochs=500)

    # wrapper.print_global_classifier_param()
    plat.set_model_save_info("lending_dann")
    task_id = "20200901_no_XAI_BCE_03_lr_0003_w_36"
    plat.set_patience(300)
    plat.train_dann(epochs=300, lr=2e-3, task_id=task_id)

    wrapper.print_parameters()
