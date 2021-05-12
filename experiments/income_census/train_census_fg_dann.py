from collections import OrderedDict

from data_process.census_process.mapping_resource import embedding_dim_map
from datasets.census_dataloader import get_census_adult_dataloaders
from experiments.income_census.global_config import feature_extractor_architecture_list, \
    pre_train_dann_hypterparameters
from models.classifier import CensusFeatureAggregator
from models.dann_models import create_embedding
from models.discriminator import CensusRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.model_config import wire_fg_dann_global_model
from utils import get_timestamp, get_current_date


# def partition_data(data):
#     # COLUMNS_TO_LOAD = ['age',  # 0
#     #                    'education_year',  # 1
#     #                    'capital_gain',  # 2
#     #                    'capital_loss',  # 3
#     #                    'age_bucket',  # 4 cat
#     #                    'marital_status',  # 5 cat
#     #                    'gender',  # 6 cat
#     #                    'native_country',  # 7 cat
#     #                    'race',  # 8 cat
#     #                    'workclass',  # 9 cat
#     #                    'occupation',  # 10 cat
#     #                    'education',  # 11 cat
#     #                    "hours_per_week",  # 12
#     #                    "relationship",  # 13 cat
#     #                    "income_label"]
#
#     wide_feat = [data[:, 0].reshape(-1, 1),
#                  data[:, 1].reshape(-1, 1),
#                  data[:, 2].reshape(-1, 1),
#                  data[:, 3].reshape(-1, 1)]
#     # demo_feat = {"non_embedding": OrderedDict({"gender": data[:, 6]}),
#     #              "embeddings": OrderedDict({"age_bucket": data[:, 4],
#     #                                         "marital_status": data[:, 5],
#     #                                         "relationship": data[:, 13],
#     #                                         "race": data[:, 8]})}
#     # emp_feat = {"non_embedding": OrderedDict({"hours_per_week": data[:, 12]}),
#     #             "embeddings": OrderedDict({"age_bucket": data[:, 4],
#     #                                        "workclass": data[:, 9],
#     #                                        "occupation": data[:, 10],
#     #                                        "education": data[:, 11]})}
#     # demo_emp_feat = {"non_embedding": OrderedDict({"gender": data[:, 6]}),
#     #                  "embeddings": OrderedDict({"relationship": data[:, 13],
#     #                                             "marital_status": data[:, 5],
#     #                                             "occupation": data[:, 10],
#     #                                             "education": data[:, 11]})}
#
#     demo_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
#                                             "marital_status": data[:, 5],
#                                             "gender": data[:, 6],
#                                             "race": data[:, 8]})}
#     emp_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
#                                            "workclass": data[:, 9],
#                                            "occupation": data[:, 10],
#                                            "education": data[:, 11]})}
#     demo_emp_feat = {"embeddings": OrderedDict({"gender": data[:, 6],
#                                                 "marital_status": data[:, 5],
#                                                 "occupation": data[:, 10],
#                                                 "education": data[:, 11]})}
#
#     # demo_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
#     #                                         "marital_status": data[:, 5],
#     #                                         "gender": data[:, 6],
#     #                                         "native_country": data[:, 7],
#     #                                         "race": data[:, 8]})}
#     # emp_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
#     #                                        "workclass": data[:, 9],
#     #                                        "occupation": data[:, 10],
#     #                                        "education": data[:, 11]})}
#     # demo_emp_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
#     #                                             "gender": data[:, 6],
#     #                                             "occupation": data[:, 10],
#     #                                             "education": data[:, 11]})}
#     deep_partition = [demo_feat, emp_feat, demo_emp_feat]
#     return wide_feat, deep_partition


def partition_data(data):
    wide_feat = [data[:, 0].reshape(-1, 1),
                 data[:, 1].reshape(-1, 1),
                 # data[:, 2].reshape(-1, 1),
                 data[:, 3].reshape(-1, 1),
                 data[:, 4].reshape(-1, 1),
                 data[:, 5].reshape(-1, 1)]

    emp_feat = {"embeddings": OrderedDict({"class_worker": data[:, 6],
                                           "major_ind_code": data[:, 7],
                                           "major_occ_code": data[:, 8],
                                           "unemp_reason": data[:, 9],
                                           "full_or_part_emp": data[:, 10],
                                           "own_or_self": data[:, 11]})}
    demo_feat = {"embeddings": OrderedDict({"education": data[:, 12],
                                            "race": data[:, 13],
                                            "age_index": data[:, 14],
                                            "gender_index": data[:, 15],
                                            "marital_stat": data[:, 16],
                                            "union_member": data[:, 17],
                                            "vet_benefits": data[:, 18],
                                            "vet_question": data[:, 19]})}
    residence_feat = {"embeddings": OrderedDict({"region_prev_res": data[:, 20],
                                                 "state_prev_res": data[:, 21],
                                                 "mig_chg_msa": data[:, 22],
                                                 "mig_chg_reg": data[:, 23],
                                                 "mig_move_reg": data[:, 24],
                                                 "mig_same": data[:, 25],
                                                 "mig_prev_sunbelt": data[:, 26],
                                                 "hisp_origin": data[:, 31],
                                                 "country_father": data[:, 32],
                                                 "country_mother": data[:, 33],
                                                 "country_self": data[:, 34],
                                                 "citizenship": data[:, 35]
                                                 })}
    household_feat = {"embeddings": OrderedDict({"tax_filer_stat": data[:, 27],
                                                 "det_hh_fam_stat": data[:, 28],
                                                 "det_hh_summ": data[:, 29],
                                                 "fam_under_18": data[:, 30]})}
    # origin_feat = {"embeddings": OrderedDict({"hisp_origin": data[:, 31],
    #                                           "country_father": data[:, 32],
    #                                           "country_mother": data[:, 33],
    #                                           "country_self": data[:, 34],
    #                                           "citizenship": data[:, 35]})}

    # deep_partition = [emp_feat, demo_feat, residence_feat, household_feat, origin_feat]
    # deep_partition = [emp_feat, demo_feat, residence_feat, household_feat]
    deep_partition = [emp_feat, demo_feat, residence_feat, household_feat]
    return wide_feat, deep_partition


def create_model_group(extractor_input_dim):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dim)
    aggregator = CensusFeatureAggregator(input_dim=extractor_input_dim[-1])
    discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dim[-1])
    return extractor, aggregator, discriminator


def create_embedding_dict(embedding_dim_map):
    # tag embedding map is used for embedding reuse. features with the same tag use the same embedding.
    tag_embedding_map = dict()
    feat_embedding_dict = dict()
    for feat_name, val in embedding_dim_map.items():
        tag = val[2]
        embedding = tag_embedding_map.get(tag)
        if embedding is None:
            embedding = create_embedding((val[0], val[1]))
            tag_embedding_map[tag] = embedding
        feat_embedding_dict[feat_name] = embedding
    return feat_embedding_dict


def create_global_model(pos_class_weight=1.0):
    embedding_dict = create_embedding_dict(embedding_dim_map)

    num_wide_feature = 5
    using_feature_group = True
    using_interaction = False
    using_transform_matrix = False

    global_model = wire_fg_dann_global_model(embedding_dict=embedding_dict,
                                             feature_extractor_architecture_list=feature_extractor_architecture_list,
                                             num_wide_feature=num_wide_feature,
                                             using_feature_group=using_feature_group,
                                             using_interaction=using_interaction,
                                             using_transform_matrix=using_transform_matrix,
                                             partition_data_fn=partition_data,
                                             create_model_group_fn=create_model_group,
                                             pos_class_weight=pos_class_weight)

    return global_model


if __name__ == "__main__":
    # lr = pre_train_dann_hypterparameters['lr']
    lr = 5e-4
    momentum = 0.99
    weight_decay = 0.0001

    batch_size = pre_train_dann_hypterparameters['batch_size']
    apply_global_domain_adaption = pre_train_dann_hypterparameters['apply_global_domain_adaption']
    global_domain_adaption_lambda = pre_train_dann_hypterparameters['global_domain_adaption_lambda']
    pos_class_weight = pre_train_dann_hypterparameters['pos_class_weight']
    epoch_patience = pre_train_dann_hypterparameters['epoch_patience']
    valid_metric = pre_train_dann_hypterparameters['valid_metric']

    census_dann_dir = "census_dann"

    date = get_current_date()
    timestamp = get_timestamp()

    # load data
    data_dir = "/Users/yankang/Documents/Data/census/output/"
    source_adult_train_file_name = data_dir + 'undergrad_census9495_da_300_train.csv'
    target_adult_train_file_name = data_dir + 'grad_census9495_da_300_train.csv'
    source_adult_test_file_name = data_dir + 'undergrad_census9495_da_300_test.csv'
    target_adult_test_file_name = data_dir + 'grad_census9495_da_300_test.csv'
    # source_adult_train_file_name = data_dir + 'undergrad_census9495_da_200_train.csv'
    # target_adult_train_file_name = data_dir + 'grad_census9495_da_200_train.csv'
    # source_adult_test_file_name = data_dir + 'undergrad_census9495_da_200_test.csv'
    # target_adult_test_file_name = data_dir + 'grad_census9495_da_200_test.csv'
    tag = "DEGREE"

    print("[INFO] Load train data")
    source_adult_train_loader, _ = get_census_adult_dataloaders(
        ds_file_name=source_adult_train_file_name, batch_size=batch_size, split_ratio=1.0)
    target_adult_train_loader, _ = get_census_adult_dataloaders(
        ds_file_name=target_adult_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    source_adult_valid_loader, _ = get_census_adult_dataloaders(
        ds_file_name=source_adult_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)
    target_adult_valid_loader, _ = get_census_adult_dataloaders(
        ds_file_name=target_adult_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)

    model = create_global_model(pos_class_weight=pos_class_weight)
    plat = FederatedDAANLearner(model=model,
                                source_train_loader=source_adult_train_loader,
                                source_val_loader=source_adult_valid_loader,
                                target_train_loader=target_adult_train_loader,
                                target_val_loader=target_adult_valid_loader,
                                max_epochs=400,
                                epoch_patience=epoch_patience)

    plat.set_model_save_info(census_dann_dir)

    task_id = date + "_" + tag + "_" + str(lr) + "_" + str(batch_size) + "_gd" + str(
        apply_global_domain_adaption) + "_" + str(timestamp)
    plat.train_dann(epochs=200,
                    lr=lr,
                    task_id=task_id,
                    metric=valid_metric,
                    apply_global_domain_adaption=apply_global_domain_adaption,
                    global_domain_adaption_lambda=global_domain_adaption_lambda,
                    momentum=momentum,
                    weight_decay=weight_decay)

    model.print_parameters()
