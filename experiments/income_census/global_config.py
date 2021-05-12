# feature_extractor_architecture_list = [[28, 40, 28, 6],
#                                        [25, 40, 25, 6],
#                                        [36, 52, 36, 8],
#                                        [27, 40, 27, 6],
                                       # [20, 36, 20, 6]
                                       # ]
# feature_extractor_architecture_list = [[28, 56, 28, 14],
#                                        [25, 50, 25, 12],
#                                        [36, 72, 36, 18],
#                                        [27, 54, 27, 13],
#                                        [20, 40, 20, 10]
#                                        ]
feature_extractor_architecture_list = [[28, 56, 28, 14],
                                       [25, 50, 25, 12],
                                       [56, 86, 56, 18],
                                       [27, 54, 27, 13]
                                       ]

pre_train_dann_hypterparameters = {
    "lr": 5e-4,
    "batch_size": 64,
    "apply_global_domain_adaption": False,
    "global_domain_adaption_lambda": 1.0,
    "pos_class_weight": 5,
    "epoch_patience": 1.5,
    "valid_metric": ('ks', 'auc')}
