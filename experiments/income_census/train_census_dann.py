from collections import OrderedDict

from datasets.census_dataloader import get_census_adult_dataloaders
from models.classifier import CensusRegionAggregator
from models.dann_models import create_embedding
from models.discriminator import CensusRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from data_process.census_process.mapping_resource import embedding_dim_map
from models.model_config import wire_global_model


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
    # feature_group_map = {"employment": {"class_worker", "major_ind_code", "major_occ_code", "unemp_reason",
    #                                     "full_or_part_emp", "own_or_self"},
    #                      "demo": {"education", "race", "age_index", "gender_index"},
    #                      "residence": {"region_prev_res", "state_prev_res", "mig_chg_msa",
    #                                    "mig_chg_reg", "mig_move_reg",
    #                                    "mig_same", "mig_prev_sunbelt"},
    #                      "household": {"marital_stat", "tax_filer_stat", "det_hh_fam_stat", "det_hh_summ",
    #                                    "fam_under_18"},
    #                      "Origin": {"hisp_origin", "country_father", "country_mother", "country_self", "citizenship"},
    #                      "social_status": {"union_member", "vet_benefits", "vet_question"}}
    wide_feat = [data[:, 0].reshape(-1, 1),
                 data[:, 1].reshape(-1, 1),
                 data[:, 2].reshape(-1, 1),
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
                                                 "mig_prev_sunbelt": data[:, 26]})}
    household_feat = {"embeddings": OrderedDict({"tax_filer_stat": data[:, 27],
                                                 "det_hh_fam_stat": data[:, 28],
                                                 "det_hh_summ": data[:, 29],
                                                 "fam_under_18": data[:, 30]})}
    origin_feat = {"embeddings": OrderedDict({"hisp_origin": data[:, 31],
                                              "country_father": data[:, 32],
                                              "country_mother": data[:, 33],
                                              "country_self": data[:, 34],
                                              "citizenship": data[:, 35]})}
    # social_status_feat = {"embeddings": OrderedDict({"union_member": data[:, 33],
    #                                                  "vet_benefits": data[:, 34],
    #                                                  "vet_question": data[:, 35]})}

    deep_partition = [emp_feat, demo_feat, residence_feat, household_feat, origin_feat]
    return wide_feat, deep_partition


# def partition_data(data):
#     wide_feat = [data[:, 0].reshape(-1, 1),
#                  data[:, 1].reshape(-1, 1),
#                  data[:, 2].reshape(-1, 1),
#                  data[:, 3].reshape(-1, 1)]
#     demo_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
#                                             "marital_status": data[:, 5],
#                                             "gender": data[:, 6],
#                                             "native_country": data[:, 7],
#                                             "race": data[:, 8],
#                                             "relationship": data[:, 12]})}
#     emp_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
#                                            "workclass": data[:, 9],
#                                            "occupation": data[:, 10],
#                                            "education": data[:, 11]})}
#     demo_emp_feat = {"embeddings": OrderedDict({"age_bucket": data[:, 4],
#                                                 "relationship": data[:, 12],
#                                                 "occupation": data[:, 10],
#                                                 "education": data[:, 11]})}
#     deep_partition = [demo_feat, emp_feat, demo_emp_feat]
#     return wide_feat, deep_partition


# def create_model_group(extractor_input_dim):
#     extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dim)
#     region_classifier = CensusRegionAggregator(input_dim=extractor_input_dim[-1])
#     discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dim[-1])
#     return RegionalModel(extractor=extractor,
#                          aggregator=region_classifier,
#                          discriminator=discriminator)

def create_model_group(extractor_input_dim):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dim)
    classifier = CensusRegionAggregator(input_dim=extractor_input_dim[-1])
    discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dim[-1])
    return extractor, classifier, discriminator


def create_embedding_dict(embedding_dim_map):
    # feat_org2dim = {'age_bucket': 11, 'marital_status': 7, 'gender': 2, 'native_country': 43, 'relationship': 6,
    #                 'race': 5, 'workclass': 9, 'occupation': 15, 'education': 17}
    # feat_emb2dim = {'age_bucket': 11, 'marital_status': 7, 'gender': 2, 'native_country': 43, 'relationship': 6,
    #                 'race': 5, 'workclass': 9, 'occupation': 15, 'education': 17}
    # feat_org2dim = {'age_bucket': 11, 'marital_status': 7, 'relationship': 6, 'race': 5, 'workclass': 9,
    #                 'occupation': 15, 'education': 17}
    # feat_emb2dim = {'age_bucket': 6, 'marital_status': 5, 'relationship': 4, 'race': 3, 'workclass': 7,
    #                 'occupation': 8, 'education': 8}
    # feat_org2dim = {'age_bucket': 11, 'marital_status': 7, 'gender': 2, 'race': 5, 'workclass': 9,
    #                 'occupation': 15, 'education': 17}
    # feat_emb2dim = {'age_bucket': 6, 'marital_status': 5, 'gender': 2, 'race': 3, 'workclass': 6,
    #                 'occupation': 8, 'education': 8}

    # feat_org2dim = {'age_bucket': 11, 'marital_status': 7, 'gender': 2, 'native_country': 43,
    #                 'race': 5, 'workclass': 9, 'occupation': 15, 'education': 17, 'relationship': 6}
    # feat_emb2dim = {'age_bucket': 8, 'marital_status': 8, 'gender': 8, 'native_country': 8,
    #                 'race': 8, 'workclass': 8, 'occupation': 8, 'education': 8, 'relationship': 8}
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


def create_region_model_wrappers(input_dims_list):
    wrapper_list = list()
    for input_dim in input_dims_list:
        wrapper_list.append(create_model_group(input_dim))
    return wrapper_list


def create_global_model_model(pos_class_weight=1.0):
    # embedding_dim = 8
    embedding_dict = create_embedding_dict(embedding_dim_map)
    # input_dims_list = [[16, 24, 16, 8],
    #                    [28, 40, 28, 8],
    #                    [23, 36, 23, 8]]
    # input_dims_list = [[48, 50, 48, 6],
    #                    [32, 40, 32, 6],
    #                    [32, 40, 32, 6]]
    # input_dims_list = [[48, 50, 48, 6],
    #                    [32, 40, 32, 6],
    #                    [32, 40, 32, 6]]
    input_dims_list = [[28, 40, 20, 5],
                       [25, 40, 20, 5],
                       [36, 52, 20, 5],
                       [27, 40, 20, 5],
                       [20, 36, 20, 5]]
    # input_dims_list = [[28, 40, 10],
    #                    [25, 40, 10],
    #                    [46, 60, 10],
    #                    [27, 40, 10],
    #                    [20, 40, 10]]
    # input_dim_list = [5 * embedding_dim, 4 * embedding_dim, 4 * embedding_dim]
    # region_wrapper_list = create_region_model_wrappers(input_dims_list)
    #
    # global_input_dim = 6 + len(input_dims_list)
    # classifier = GlobalClassifier(input_dim=global_input_dim)
    # wrapper = GlobalModel(classifier, region_wrapper_list, embedding_dict, partition_data,
    #                       pos_class_weight=pos_class_weight, loss_name="BCE")

    num_wide_feature = 6
    using_transform_matrix = True
    using_feature_group = True

    global_model = wire_global_model(embedding_dict=embedding_dict,
                                     input_dims_list=input_dims_list,
                                     num_wide_feature=num_wide_feature,
                                     using_feature_group=using_feature_group,
                                     using_transform_matrix=using_transform_matrix,
                                     partition_data_fn=partition_data,
                                     create_model_group_fn=create_model_group,
                                     pos_class_weight=pos_class_weight)

    return global_model


if __name__ == "__main__":
    model = create_global_model_model(pos_class_weight=2.0)

    # source_adult_train_file_name = '../../datasets/census_processed/degree_source_train.csv'
    # target_adult_train_file_name = '../../datasets/census_processed/degree_target_train.csv'
    # source_adult_test_file_name = '../../datasets/census_processed/degree_source_test.csv'
    # target_adult_test_file_name = '../../datasets/census_processed/degree_target_test.csv'
    # tag = "DEGREE"

    source_adult_train_file_name = '../../datasets/census_processed/undergrad_census9495_da_train.csv'
    target_adult_train_file_name = '../../datasets/census_processed/grad_census9495_da_train.csv'
    source_adult_test_file_name = '../../datasets/census_processed/undergrad_census9495_da_test.csv'
    target_adult_test_file_name = '../../datasets/census_processed/grad_census9495_da_test.csv'
    tag = "DEGREE"

    # source_adult_train_file_name = '../../datasets/census_processed/adult_source_train.csv'
    # target_adult_train_file_name = '../../datasets/census_processed/adult_target_train.csv'
    # source_adult_test_file_name = '../../datasets/census_processed/adult_source_test.csv'
    # target_adult_test_file_name = '../../datasets/census_processed/adult_target_test.csv'
    # tag = "ASIA"

    # target_adult_train_file_name = '../../datasets/census_processed/adult_target_train.csv'
    # source_adult_train_file_name = '../../datasets/census_processed/adult_source_train.csv'
    # target_adult_test_file_name = '../../datasets/census_processed/adult_target_test.csv'
    # source_adult_test_file_name = '../../datasets/census_processed/adult_source_test.csv'

    # source_adult_train_file_name = '../../datasets/census_processed/standardized_adult.csv'
    # target_adult_train_file_name = '../../datasets/census_processed/sampled_standardized_census95_train.csv'
    # source_adult_test_file_name = '../../datasets/census_processed/standardized_adult_test.csv'
    # target_adult_test_file_name = '../../datasets/census_processed/sampled_standardized_census95_test.csv'
    # target_adult_test_file_name = '../../datasets/census_processed/standardized_census95_test.csv'

    # date = "20201215"; lr = 8e-4; batch_size = 128; version = 5
    # date = "20201215"; lr = 1e-3; batch_size = 64; version = 5
    date = "20210418"
    lr = 8e-4
    batch_size = 64
    version = 5

    print("[INFO] Load train data")
    source_adult_train_loader, _ = get_census_adult_dataloaders(
        ds_file_name=source_adult_train_file_name, batch_size=batch_size, split_ratio=1.0)
    target_adult_train_loader, _ = get_census_adult_dataloaders(
        ds_file_name=target_adult_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    # census_adult_valid_loader, census_adult_test_loader = get_census_adult_dataloaders(
    #     ds_file_name=census_adult_test_file_name, batch_size=batch_size * 2, split_ratio=0.7)
    # census95_valid_loader, census95_test_loader = get_census_95_dataloaders(
    #     ds_file_name=census_95_test_file_name, batch_size=batch_size * 2, split_ratio=0.7)
    source_adult_valid_loader, _ = get_census_adult_dataloaders(
        ds_file_name=source_adult_test_file_name, batch_size=batch_size * 2, split_ratio=1.0)
    target_adult_valid_loader, _ = get_census_adult_dataloaders(
        ds_file_name=target_adult_test_file_name, batch_size=batch_size * 2, split_ratio=1.0)

    plat = FederatedDAANLearner(model=model,
                                source_train_loader=source_adult_train_loader,
                                source_val_loader=source_adult_valid_loader,
                                target_train_loader=target_adult_train_loader,
                                target_val_loader=target_adult_valid_loader,
                                max_epochs=400,
                                epoch_patience=10)

    # wrapper.print_global_classifier_param()
    plat.set_model_save_info("census_dann")
    # plat.set_model_save_info("lending_dann")

    task_id = date + "_" + tag + "_" + str(lr) + "_" + str(batch_size) + "_" + str(version)
    # 5e-4
    plat.train_dann(epochs=400, lr=lr, task_id=task_id)

    model.print_parameters()
