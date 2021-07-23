
# feature_extractor_architecture_list = [[28, 56, 28, 14],
#                                        [25, 50, 25, 12],
#                                        [36, 72, 36, 18],
#                                        [27, 54, 27, 13],
#                                        [20, 40, 20, 10]
#                                        ]
feature_extractor_architecture_list = [[28, 56, 28, 14],
                                       [25, 50, 25, 12],
                                       [56, 86, 56, 18],
                                       [27, 54, 27, 13]]

# feature_extractor_for_intr_architecture_list = [[53, 106, 53, 26],
#                                                 [84, 142, 84, 32],
#                                                 [55, 110, 55, 27],
#                                                 [81, 136, 81, 30],
#                                                 [52, 104, 52, 25],
#                                                 [83, 140, 83, 31]]

feature_extractor_for_intr_architecture_list = [[53, 78, 53, 15],
                                                [84, 120, 84, 20],
                                                [55, 81, 55, 15],
                                                [81, 120, 81, 20],
                                                [52, 78, 52, 15],
                                                [83, 120, 83, 20]]

pre_train_dann_hypterparameters = {
    "lr": 6e-4,
    "batch_size": 128,
    "apply_global_domain_adaption": False,
    "global_domain_adaption_lambda": 1.0,
    "pos_class_weight": 1.0,
    "epoch_patience": 2,
    "valid_metric": ('ks', 'auc'),
    # "data_tag": "all4000pos004v4"
    "data_tag": "all4000pos002"
}
