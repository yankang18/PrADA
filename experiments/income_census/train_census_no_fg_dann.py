from collections import OrderedDict

import numpy as np

from data_process.census_process.mapping_resource import embedding_dim_map
from datasets.census_dataloader import get_income_census_dataloaders, get_census_adult_dataloaders
from experiments.income_census.global_config import feature_extractor_architecture_list, \
    pre_train_dann_hypterparameters
from experiments.income_census.train_census_fg_dann import create_embedding_dict
from models.classifier import GlobalClassifier, CensusFeatureAggregator
from models.dann_models import GlobalModel, RegionalModel
from models.discriminator import LendingRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from utils import get_timestamp, get_current_date


def partition_data(data):
    wide_feat = [data[:, 0].reshape(-1, 1),
                 data[:, 1].reshape(-1, 1),
                 # data[:, 2].reshape(-1, 1),
                 data[:, 3].reshape(-1, 1),
                 data[:, 4].reshape(-1, 1),
                 data[:, 5].reshape(-1, 1)]

    deep_feat = {"embeddings": OrderedDict({"class_worker": data[:, 6],
                                            "major_ind_code": data[:, 7],
                                            "major_occ_code": data[:, 8],
                                            "unemp_reason": data[:, 9],
                                            "full_or_part_emp": data[:, 10],
                                            "own_or_self": data[:, 11],
                                            "education": data[:, 12],
                                            "race": data[:, 13],
                                            "age_index": data[:, 14],
                                            "gender_index": data[:, 15],
                                            "marital_stat": data[:, 16],
                                            "union_member": data[:, 17],
                                            "vet_benefits": data[:, 18],
                                            "vet_question": data[:, 19],
                                            "region_prev_res": data[:, 20],
                                            "state_prev_res": data[:, 21],
                                            "mig_chg_msa": data[:, 22],
                                            "mig_chg_reg": data[:, 23],
                                            "mig_move_reg": data[:, 24],
                                            "mig_same": data[:, 25],
                                            "mig_prev_sunbelt": data[:, 26],
                                            "tax_filer_stat": data[:, 27],
                                            "det_hh_fam_stat": data[:, 28],
                                            "det_hh_summ": data[:, 29],
                                            "fam_under_18": data[:, 30]
                                            # "hisp_origin": data[:, 31],
                                            # "country_father": data[:, 32],
                                            # "country_mother": data[:, 33],
                                            # "country_self": data[:, 34],
                                            # "citizenship": data[:, 35]
                                            })}

    deep_partition = [deep_feat]
    return wide_feat, deep_partition


def create_region_model(extractor_input_dims_list, aggregation_dim):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims_list)
    # region_aggregator = IdentityRegionAggregator()
    aggregator = CensusFeatureAggregator(input_dim=extractor_input_dims_list[-1], output_dim=aggregation_dim)
    discriminator = LendingRegionDiscriminator(input_dim=extractor_input_dims_list[-1], hidden_dim=36)
    return RegionalModel(extractor=extractor,
                         aggregator=aggregator,
                         discriminator=discriminator)


# def create_embedding_dict():
#     feat_org2dim = {'age_bucket': 11, 'marital_status': 7, 'gender': 2, 'native_country': 43,
#                     'race': 5, 'workclass': 9, 'occupation': 15, 'education': 17, 'relationship': 6}
#     feat_emb2dim = {'age_bucket': 8, 'marital_status': 8, 'gender': 8, 'native_country': 8,
#                     'race': 8, 'workclass': 8, 'occupation': 8, 'education': 8, 'relationship': 8}
#     embedding_meta_dict = dict()
#     for feat_name, num_values in feat_org2dim.items():
#         embedding_dim = feat_emb2dim[feat_name]
#         embedding_meta_dict[feat_name] = (num_values, embedding_dim)
#     print(f"embedding_meta_dict: \n {embedding_meta_dict}")
#     return create_embeddings(embedding_meta_dict)


def create_region_model_list(feature_extractor_arch_list, aggregation_dim):
    model_list = list()
    for feature_extractor_arch in feature_extractor_arch_list:
        model_list.append(create_region_model(feature_extractor_arch, aggregation_dim))
    return model_list


def create_global_model(aggregation_dim, num_wide_feature, pos_class_weight):
    embedding_dict = create_embedding_dict(embedding_dim_map)
    # kk = np.sum(np.array(feature_extractor_architecture_list), axis=0) / 2
    # print("kk:", kk)
    # feature_extractor_architecture = list(np.sum(np.array(feature_extractor_architecture_list), axis=0))
    # feature_extractor_architecture = list(kk)
    # feature_extractor_architecture = [68, 104, 68, 16]
    feature_extractor_architecture = [116, 160, 32, 16]
    print(f"[INFO] feature_extractor_architecture list:{[feature_extractor_architecture]}")
    region_model_list = create_region_model_list([feature_extractor_architecture], aggregation_dim)

    global_input_dim = aggregation_dim + num_wide_feature
    print(f"[INFO] global_input_dim:{global_input_dim}")
    classifier = GlobalClassifier(input_dim=global_input_dim)
    wrapper = GlobalModel(classifier, region_model_list, embedding_dict, partition_data,
                          pos_class_weight=pos_class_weight, loss_name="BCE")
    return wrapper


if __name__ == "__main__":
    momentum = 0.99
    weight_decay = 0.0001
    lr = pre_train_dann_hypterparameters['lr']
    batch_size = pre_train_dann_hypterparameters['batch_size']
    apply_global_domain_adaption = pre_train_dann_hypterparameters['apply_global_domain_adaption']
    global_domain_adaption_lambda = pre_train_dann_hypterparameters['global_domain_adaption_lambda']
    pos_class_weight = pre_train_dann_hypterparameters['pos_class_weight']
    epoch_patience = pre_train_dann_hypterparameters['epoch_patience']
    valid_metric = pre_train_dann_hypterparameters['valid_metric']

    census_no_fg_dann_dir = "census_no_fg_dann"

    date = get_current_date()
    timestamp = get_timestamp()

    model = create_global_model(aggregation_dim=4,
                                num_wide_feature=5,
                                pos_class_weight=pos_class_weight)

    data_dir = "/Users/yankang/Documents/Data/census/output/"
    source_adult_train_file_name = data_dir + 'undergrad_census9495_da_300_train.csv'
    target_adult_train_file_name = data_dir + 'grad_census9495_da_300_train.csv'
    source_adult_test_file_name = data_dir + 'undergrad_census9495_da_300_test.csv'
    target_adult_test_file_name = data_dir + 'grad_census9495_da_300_test.csv'
    # source_adult_train_file_name = data_dir + 'undergrad_census9495_da_200_train.csv'
    # target_adult_train_file_name = data_dir + 'grad_census9495_da_200_train.csv'
    # source_adult_test_file_name = data_dir + 'undergrad_census9495_da_200_test.csv'
    # target_adult_test_file_name = data_dir + 'grad_census9495_da_200_test.csv'

    print("[INFO] Load train data")
    source_adult_train_loader, _ = get_census_adult_dataloaders(
        ds_file_name=source_adult_train_file_name, batch_size=batch_size, split_ratio=1.0)
    target_adult_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_adult_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    source_adult_valid_loader, _ = get_census_adult_dataloaders(
        ds_file_name=source_adult_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)
    target_adult_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_adult_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)

    plat = FederatedDAANLearner(model=model,
                                source_train_loader=source_adult_train_loader,
                                source_val_loader=source_adult_valid_loader,
                                target_train_loader=target_adult_train_loader,
                                target_val_loader=target_adult_valid_loader,
                                max_epochs=400,
                                epoch_patience=epoch_patience)
    plat.set_model_save_info(census_no_fg_dann_dir)

    task_id = date + "_DEGREE_no_fg_dann_" + str(lr) + "_" + str(timestamp)
    plat.train_dann(epochs=200,
                    lr=lr,
                    task_id=task_id,
                    metric=valid_metric,
                    momentum=momentum,
                    weight_decay=weight_decay)

    model.print_parameters()
