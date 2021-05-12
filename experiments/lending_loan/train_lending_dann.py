from collections import OrderedDict

from models.classifier import GlobalClassifier, CensusFeatureAggregator
from datasets.lending_dataloader import get_lending_dataloader
from models.discriminator import CensusRegionDiscriminator
from models.experiment_dann_learner import FederatedDAANLearner
from models.feature_extractor import CensusRegionFeatureExtractorDense
from models.dann_models import GlobalModel, RegionalModel, create_embeddings


# def partition_data(data):
#     # table_names = ["p_wide_col.csv", "p_debt_feat.csv", "p_payment_feat.csv",
#     #                "p_payment_debt_cross_feat.csv", "p_multi_acc_feat.csv", "p_mal_behavior_feat.csv",
#     #                "p_qualify_feat.csv", "p_loan_feat.csv"]
#
#     deep_feat = list()
#
#     wide_data = data[0]
#     wide_feat = [wide_data[:, 0].reshape(-1, 1),
#                  wide_data[:, 1].reshape(-1, 1),
#                  wide_data[:, 2].reshape(-1, 1),
#                  wide_data[:, 3].reshape(-1, 1)]
#
#     for idx in range(1, len(data) - 2):
#         feat = {"non_embedding": data[idx]}
#         deep_feat.append(feat)
#
#     # qualify_feat = [
#     #     'grade',
#     #     'emp_length',
#     #     'home_ownership',
#     #     'verification_status',
#     #     'annual_inc_comp',
#     # ]
#     qualify_data = data[6]
#     qualify_feat = {"embeddings": OrderedDict({"grade": qualify_data[:, 0],
#                                                "emp_length": qualify_data[:, 1],
#                                                "home_ownership": qualify_data[:, 2],
#                                                "verification_status": qualify_data[:, 3]}),
#                     "non_embedding": qualify_data[:, 4:]}
#     deep_feat.append(qualify_feat)
#
#     # loan_feat = [
#     #     'term',
#     #     'initial_list_status',
#     #     'purpose',
#     #     'application_type',
#     #     'disbursement_method'
#     # ]
#     loan_data = data[7]
#     loan_feat = {"embeddings": OrderedDict({"term": loan_data[:, 0],
#                                             "initial_list_status": loan_data[:, 1],
#                                             "purpose": loan_data[:, 2],
#                                             "application_type": loan_data[:, 3],
#                                             "disbursement_method": loan_data[:, 4]})}
#     deep_feat.append(loan_feat)
#
#     return wide_feat, deep_feat

def partition_data(data):
    # table_names = ["p_wide_col.csv", "p_debt_feat.csv", "p_payment_feat.csv",
    #                "p_payment_debt_cross_feat.csv", "p_multi_acc_feat.csv", "p_mal_behavior_feat.csv",
    #                "p_qualify_feat.csv", "p_loan_feat.csv"]

    deep_feat = list()

    wide_data = data[0]
    wide_feat = [wide_data[:, 0].reshape(-1, 1),
                 wide_data[:, 1].reshape(-1, 1),
                 wide_data[:, 2].reshape(-1, 1),
                 wide_data[:, 3].reshape(-1, 1)]

    for idx in range(1, len(data) - 1):
        # feat = {"non_embedding": data[idx]}
        feat = {"non_embedding": OrderedDict({"tabular_data": data[idx]})}
        deep_feat.append(feat)

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
    qualify_feat = {"embeddings": OrderedDict({"home_ownership": qualify_data[:, 0],
                                               "purpose": qualify_data[:, 1]}),
                    "non_embedding": OrderedDict({"tabular_data": qualify_data[:, 2:]})}
                    # "non_embedding": qualify_data[:, 2:]}
    deep_feat.append(qualify_feat)

    return wide_feat, deep_feat


def create_region_model_wrapper(extractor_input_dims_list):
    extractor = CensusRegionFeatureExtractorDense(input_dims=extractor_input_dims_list)
    region_classifier = CensusFeatureAggregator(input_dim=extractor_input_dims_list[-1])
    discriminator = CensusRegionDiscriminator(input_dim=extractor_input_dims_list[-1])
    return RegionalModel(extractor=extractor,
                         aggregator=region_classifier,
                         discriminator=discriminator)


def create_embedding_dict():
    # grade_map = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1, 'G': 0}
    # emp_length_map = {np.nan: 0, '< 1 year': 1, '1 year': 2, '2 years': 2, '3 years': 2, '4 years': 3, '5 years': 3,
    #                   '6 years': 3, '7 years': 4, '8 years': 4, '9 years': 4, '10+ years': 5}
    # home_ownership_map = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'ANY': 3, 'NONE': 3, 'OTHER': 3}
    # verification_status_map = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
    # term_map = {' 36 months': 0, ' 60 months': 1}
    # initial_list_status_map = {'w': 0, 'f': 1}
    # purpose_map = {'debt_consolidation': 0, 'credit_card': 0, 'small_business': 1, 'educational': 2,
    #                'car': 3, 'other': 3, 'vacation': 3, 'house': 3, 'home_improvement': 3, 'major_purchase': 3,
    #                'medical': 3, 'renewable_energy': 3, 'moving': 3, 'wedding': 3}
    # application_type_map = {'Individual': 0, 'Joint App': 1}
    # disbursement_method_map = {'Cash': 0, 'DirectPay': 1}

    # COLUMNS = {'grade': (7, 8), 'emp_length': (6, 8), 'home_ownership': (4, 6), 'verification_status': (3, 4),
    #            'purpose': (4, 6), 'term': (2, 3), 'initial_list_status': (2, 3), 'application_type': (2, 3),
    #            'disbursement_method': (2, 3)}

    # COLUMNS = {'grade': (7, 6), 'emp_length': (6, 5), 'home_ownership': (4, 3), 'verification_status': (3, 2),
    #            'purpose': (4, 3), 'term': (2, 2), 'initial_list_status': (2, 2), 'application_type': (2, 2),
    #            'disbursement_method': (2, 2)}

    COLUMNS = {'home_ownership': (4, 3), 'purpose': (4, 3)}

    embedding_meta_dict = dict()
    for col, num_values in COLUMNS.items():
        embedding_meta_dict[col] = (num_values[0], num_values[1])
    print(f"[INFO] Create embedding_meta_dict: \t {embedding_meta_dict}")
    return create_embeddings(embedding_meta_dict)


def create_region_model_wrappers(input_dims_list):
    wrapper_list = list()
    for input_dims in input_dims_list:
        wrapper_list.append(create_region_model_wrapper(input_dims))
    return wrapper_list


def compute_feature_group_interaction(input_dims_list):
    input_int_dims_list = [e for e in input_dims_list]
    start_index = 1
    for fg in input_dims_list:
        for i in range(start_index, len(input_dims_list)):
            fg_2 = input_dims_list[i]
            input_int_dims_list.append([v1 + v2 for v1, v2 in zip(fg, fg_2)])
        start_index += 1
    return input_int_dims_list


def create_global_model_wrapper(pos_class_weight=1.0, beta=1.0):
    # embedding_dim = 8
    embedding_dict = create_embedding_dict()
    # # architecture 1
    # input_dims_list = [[24, 36, 24, 8],
    #                    [17, 28, 17, 8],
    #                    [41, 54, 41, 10],
    #                    [17, 28, 17, 8],
    #                    [16, 28, 16, 8],
    #                    [12, 24, 11, 6]]
    # # architecture 2
    # input_dims_list = [[24, 32, 24, 8],
    #                    [17, 24, 17, 8],
    #                    [41, 50, 41, 10],
    #                    [17, 24, 17, 8],
    #                    [16, 24, 16, 8],
    #                    [12, 20, 11, 6]]
    # architecture 3
    input_dims_list = [[24, 32, 24, 6],
                       [17, 24, 17, 6],
                       [41, 50, 41, 8],
                       [17, 24, 17, 6],
                       [16, 24, 16, 6],
                       [12, 20, 12, 4]]
    # architecture 4
    # input_dims_list = [[24, 30, 24, 4],
    #                    [17, 20, 17, 4],
    #                    [41, 50, 41, 6],
    #                    [17, 20, 17, 4],
    #                    [16, 20, 16, 4],
    #                    [12, 18, 12, 4]]
    # architecture 5
    # input_dims_list = [[24, 30, 20, 4],
    #                    [17, 20, 15, 4],
    #                    [41, 50, 30, 6],
    #                    [17, 20, 15, 4],
    #                    [16, 20, 15, 4],
    #                    [12, 18, 12, 4]]

    input_dims_list = compute_feature_group_interaction(input_dims_list)
    print(f"input_dims_list:{input_dims_list}, len:{len(input_dims_list)}")
    region_wrapper_list = create_region_model_wrappers(input_dims_list=input_dims_list)

    global_input_dim = 4 + len(input_dims_list)
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
                                epoch_patience=2,
                                max_epochs=500)

    # wrapper.print_global_classifier_param()
    plat.set_model_save_info("lending_dann")
    task_id = "20200909_w_XAI_BCE_01_lr_0001"
    plat.train_dann(epochs=300, lr=3e-3, task_id=task_id)

    wrapper.print_parameters()
