from datasets.ppd_dataloader import get_datasets, get_dataloader
from datasets.ppd_dataloader import get_pdd_dataloaders_ob
from experiments.ppd_loan.train_ppd_no_fg_dann import create_no_fg_pdd_global_model
from models.experiment_dann_learner import FederatedDAANLearner
from models.experiment_target_learner import FederatedTargetLearner
from utils import get_timestamp, get_current_date, create_id_from_hyperparameters
from utils import test_classifier


def pretrain_ppd_dann(data_tag,
                      dann_root_dir,
                      learner_hyperparameters,
                      data_hyperparameters,
                      create_ppd_global_model_func):
    using_interaction = learner_hyperparameters['using_interaction']
    momentum = learner_hyperparameters['momentum']
    weight_decay = learner_hyperparameters['weight_decay']
    batch_size = learner_hyperparameters['batch_size']
    lr = learner_hyperparameters['lr']
    epoch_patience = learner_hyperparameters['epoch_patience']
    max_epochs = learner_hyperparameters['max_epochs']
    valid_metric = learner_hyperparameters['valid_metric']
    pos_class_weight = learner_hyperparameters['pos_class_weight']
    date = get_current_date()
    timestamp = get_timestamp()

    optimizer_param_dict = {"src": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}}

    # load data
    source_train_file_name = data_hyperparameters['source_train_file_name']
    target_train_file_name = data_hyperparameters['target_train_file_name']
    source_test_file_name = data_hyperparameters['source_test_file_name']
    target_test_file_name = data_hyperparameters['target_test_file_name']

    print(f"[INFO] load source train from: {source_train_file_name}.")
    print(f"[INFO] load target train from: {target_train_file_name}.")
    print(f"[INFO] load source test from: {source_test_file_name}.")
    print(f"[INFO] load target test from: {target_test_file_name}.")

    split_ratio = 1.0
    src_train_dataset, _ = get_datasets(ds_file_name=source_train_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_train_dataset, _ = get_datasets(ds_file_name=target_train_file_name, shuffle=True, split_ratio=split_ratio)
    src_test_dataset, _ = get_datasets(ds_file_name=source_test_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_test_dataset, _ = get_datasets(ds_file_name=target_test_file_name, shuffle=True, split_ratio=split_ratio)

    tries = 1
    for version in range(tries):
        hyperparameter_dict = {"pw": pos_class_weight, "lr": lr, "bs": batch_size, "me": max_epochs, "ts": timestamp,
                               "ve": version}
        using_intr_tag = "intr" + str(True) if using_interaction else str(False)
        task_id = date + "_ppd_fg_dann_" + data_tag + "_" + using_intr_tag + "_" + create_id_from_hyperparameters(
            hyperparameter_dict)

        print("[INFO] perform task:{0}".format(task_id))

        print("[INFO] create model.")
        global_model = create_ppd_global_model_func(pos_class_weight=pos_class_weight,
                                                    using_interaction=using_interaction)

        print("[INFO] load train data.")
        src_train_loader = get_dataloader(src_train_dataset, batch_size=batch_size)
        tgt_train_loader = get_dataloader(tgt_train_dataset, batch_size=batch_size)

        print("[INFO] load test data.")
        src_test_loader = get_dataloader(src_test_dataset, batch_size=batch_size * 4)
        tgt_test_loader = get_dataloader(tgt_test_dataset, batch_size=batch_size * 4)

        plat = FederatedDAANLearner(model=global_model,
                                    source_da_train_loader=src_train_loader,
                                    source_val_loader=src_test_loader,
                                    target_da_train_loader=tgt_train_loader,
                                    target_val_loader=tgt_test_loader,
                                    max_epochs=max_epochs,
                                    epoch_patience=epoch_patience)
        plat.set_model_save_info(dann_root_dir)

        plat.train_dann(epochs=120,
                        task_id=task_id,
                        metric=valid_metric,
                        optimizer_param_dict=optimizer_param_dict)


def finetune_ppd_dann(dann_task_id,
                      ppd_pretain_model_root_dir,
                      ppd_finetune_target_root_dir,
                      dann_hyperparameters,
                      data_hyperparameters,
                      create_ppd_global_model_func):
    load_global_classifier = dann_hyperparameters['load_global_classifier']
    using_interaction = dann_hyperparameters['using_interaction']
    momentum = dann_hyperparameters['momentum']
    weight_decay = dann_hyperparameters['weight_decay']
    batch_size = dann_hyperparameters['batch_size']
    lr = dann_hyperparameters['lr']
    pos_class_weight = dann_hyperparameters['pos_class_weight']
    valid_metric = dann_hyperparameters['valid_metric']

    date = get_current_date()
    timestamp = get_timestamp()

    glr = "ft_glr" if load_global_classifier else "rt_glr"
    hyperparameter_dict = {"pw": pos_class_weight, "lr": lr, "bs": batch_size, "ts": timestamp}
    appendix = create_id_from_hyperparameters(hyperparameter_dict)
    target_task_id = dann_task_id + "@target_" + date + "-" + glr + "_" + appendix

    # initialize model
    model = create_ppd_global_model_func(pos_class_weight=pos_class_weight, using_interaction=using_interaction)

    # load pre-trained model
    model.load_model(root=ppd_pretain_model_root_dir,
                     task_id=dann_task_id,
                     load_global_classifier=load_global_classifier,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # load data
    target_train_file_name = data_hyperparameters['target_train_file_name']
    target_test_file_name = data_hyperparameters['target_test_file_name']

    print(f"[INFO] load target train from: {target_train_file_name}.")
    print(f"[INFO] load target test from: {target_test_file_name}.")

    print("[INFO] Load train data")
    target_train_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_valid_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    # perform target training
    plat_target = FederatedTargetLearner(model=model,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=400)
    plat_target.set_model_save_info(ppd_finetune_target_root_dir)

    plat_target.train_target_with_alternating(global_epochs=400,
                                              top_epochs=1,
                                              bottom_epochs=1,
                                              lr=lr,
                                              task_id=target_task_id,
                                              metric=valid_metric,
                                              momentum=momentum,
                                              weight_decay=weight_decay)

    # load best model
    model.load_model(root=ppd_finetune_target_root_dir,
                     task_id=target_task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classifier(model, target_valid_loader, "test")
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")


def train_no_adaptation(data_tag,
                        ppd_no_ad_root_dir,
                        learner_hyperparameters,
                        data_hyperparameters):
    train_data_tag = learner_hyperparameters['train_data_tag']
    momentum = learner_hyperparameters['momentum']
    weight_decay = learner_hyperparameters['weight_decay']
    batch_size = learner_hyperparameters['batch_size']
    lr = learner_hyperparameters['lr']
    epoch_patience = learner_hyperparameters['epoch_patience']
    max_epochs = learner_hyperparameters['max_epochs']
    valid_metric = learner_hyperparameters['valid_metric']
    pos_class_weight = learner_hyperparameters['pos_class_weight']

    # load data
    source_train_file_name = data_hyperparameters['source_train_file_name']
    target_train_file_name = data_hyperparameters['target_train_file_name']
    source_test_file_name = data_hyperparameters['source_test_file_name']
    target_test_file_name = data_hyperparameters['target_test_file_name']
    src_tgt_train_file_name = data_hyperparameters['src_tgt_train_file_name']

    print(f"[INFO] load source train from: {source_train_file_name}.")
    print(f"[INFO] load target train from: {target_train_file_name}.")
    print(f"[INFO] load source test from: {source_test_file_name}.")
    print(f"[INFO] load target test from: {target_test_file_name}.")
    print(f"[INFO] load src+tgt test from: {src_tgt_train_file_name}.")

    split_ratio = 1.0
    src_tgt_train_dataset, _ = get_datasets(ds_file_name=src_tgt_train_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_train_dataset, _ = get_datasets(ds_file_name=target_train_file_name, shuffle=True, split_ratio=split_ratio)
    src_test_dataset, _ = get_datasets(ds_file_name=source_test_file_name, shuffle=True, split_ratio=split_ratio)
    tgt_test_dataset, _ = get_datasets(ds_file_name=target_test_file_name, shuffle=True, split_ratio=split_ratio)
    dataset_dict = {"tgt": tgt_train_dataset,
                    "all": src_tgt_train_dataset}

    timestamp = get_timestamp()
    date = get_current_date() + "_" + data_tag + "_ppd_no_ad_wo_fg"
    tries = 1
    for version in range(tries):
        hyperparameter_dict = {"pw": pos_class_weight, "lr": lr, "bs": batch_size, "ts": timestamp, "ve": version}
        task_id = date + "_" + train_data_tag + "_" + create_id_from_hyperparameters(hyperparameter_dict)
        print("[INFO] perform task:{0}".format(task_id))

        model = create_no_fg_pdd_global_model(aggregation_dim=5,
                                              num_wide_feature=6,
                                              pos_class_weight=pos_class_weight)
        print("[INFO] model created.")
        src_train_loader = get_dataloader(dataset_dict[train_data_tag], batch_size=batch_size)
        tgt_train_loader = get_dataloader(tgt_train_dataset, batch_size=batch_size)

        tgt_valid_loader = get_dataloader(tgt_test_dataset, batch_size=batch_size * 4)
        src_valid_loader = get_dataloader(src_test_dataset, batch_size=batch_size * 4)

        plat = FederatedDAANLearner(model=model,
                                    source_da_train_loader=src_train_loader,
                                    source_val_loader=src_valid_loader,
                                    target_da_train_loader=tgt_train_loader,
                                    target_val_loader=tgt_valid_loader,
                                    epoch_patience=epoch_patience,
                                    validation_batch_interval=5)
        plat.set_model_save_info(ppd_no_ad_root_dir)

        plat.train_wo_adaption(epochs=max_epochs,
                               lr=lr,
                               train_source=True,
                               metric=valid_metric,
                               task_id=task_id,
                               momentum=momentum,
                               weight_decay=weight_decay)

        # load best model
        model.load_model(root=ppd_no_ad_root_dir,
                         task_id=task_id,
                         load_global_classifier=True,
                         timestamp=None)

        print("[DEBUG] Global classifier Model Parameter After train:")
        model.print_parameters()

        acc, auc, ks = test_classifier(model, tgt_valid_loader, 'test')
        print(f"acc:{acc}, auc:{auc}, ks:{ks}")
