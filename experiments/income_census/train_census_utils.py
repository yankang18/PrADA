from datasets.census_dataloader import get_income_census_dataloaders
from experiments.income_census import train_census_fg_dann as fg_dann
from experiments.income_census import train_census_no_fg_dann as no_fg_dann
from models.experiment_dann_learner import FederatedDAANLearner
from models.experiment_target_learner import FederatedTargetLearner
from utils import get_timestamp, get_current_date, create_id_from_hyperparameters, test_classifier


def pretrain_census_dann(data_tag,
                         census_dann_root_dir,
                         learner_hyperparameters,
                         data_hyperparameters,
                         create_census_global_model_func):
    # hyper-parameters
    using_interaction = learner_hyperparameters['using_interaction']
    momentum = learner_hyperparameters['momentum']
    weight_decay = learner_hyperparameters['weight_decay']
    batch_size = learner_hyperparameters['batch_size']
    lr = learner_hyperparameters['lr']
    epoch_patience = learner_hyperparameters['epoch_patience']
    max_epochs = learner_hyperparameters['max_epochs']
    valid_metric = learner_hyperparameters['valid_metric']

    date = get_current_date()
    timestamp = get_timestamp()

    optimizer_param_dict = {"src": {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}}

    # create pre-train task id
    hyperparameter_dict = {"lr": lr, "bs": batch_size, "me": max_epochs, "ts": timestamp}
    using_intr_tag = "intr" + str(True) if using_interaction else str(False)
    task_id = date + "_census_fg_dann_" + data_tag + using_intr_tag + "_" + create_id_from_hyperparameters(
        hyperparameter_dict)

    # load data
    source_train_file_name = data_hyperparameters['source_train_file_name']
    target_train_file_name = data_hyperparameters['target_train_file_name']
    source_test_file_name = data_hyperparameters['source_test_file_name']
    target_test_file_name = data_hyperparameters['target_test_file_name']

    print(f"[INFO] load source train from: {source_train_file_name}.")
    print(f"[INFO] load target train from: {target_train_file_name}.")
    print(f"[INFO] load source test from: {source_test_file_name}.")
    print(f"[INFO] load target test from: {target_test_file_name}.")

    print("[INFO] Load train data.")
    source_da_census_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=source_train_file_name, batch_size=batch_size, split_ratio=1.0)
    target_da_census_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data.")
    source_census_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=source_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)
    target_census_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)

    model = create_census_global_model_func(num_wide_feature=5, using_interaction=using_interaction)
    plat = FederatedDAANLearner(model=model,
                                source_da_train_loader=source_da_census_train_loader,
                                source_val_loader=source_census_valid_loader,
                                target_da_train_loader=target_da_census_train_loader,
                                target_val_loader=target_census_valid_loader,
                                max_epochs=max_epochs,
                                epoch_patience=epoch_patience)
    plat.set_model_save_info(census_dann_root_dir)

    plat.train_dann(epochs=200,
                    task_id=task_id,
                    metric=valid_metric,
                    optimizer_param_dict=optimizer_param_dict)


def finetune_census_dann(dann_task_id,
                         census_pretain_model_root_dir,
                         census_finetune_target_root_dir,
                         learner_hyperparameters,
                         data_hyperparameters,
                         create_census_global_model_func):
    # hyper-parameters
    load_global_classifier = learner_hyperparameters['load_global_classifier']
    using_interaction = learner_hyperparameters['using_interaction']
    momentum = learner_hyperparameters['momentum']
    weight_decay = learner_hyperparameters['weight_decay']
    batch_size = learner_hyperparameters['batch_size']
    lr = learner_hyperparameters['lr']
    valid_metric = learner_hyperparameters['valid_metric']

    date = get_current_date()
    timestamp = get_timestamp()

    # create fine-tune task id
    hyperparameter_dict = {"lr": lr, "bs": batch_size, "ts": timestamp}
    appendix = create_id_from_hyperparameters(hyperparameter_dict)
    glr = "ft_glr" if load_global_classifier else "rt_glr"
    target_task_id = dann_task_id + "@target_" + date + "-" + glr + "_" + appendix

    # initialize model
    model = create_census_global_model_func(using_interaction=using_interaction)

    # load pre-trained model
    model.load_model(root=census_pretain_model_root_dir,
                     task_id=dann_task_id,
                     load_global_classifier=load_global_classifier,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    model.print_parameters()

    # Load data
    target_train_file_name = data_hyperparameters['target_train_file_name']
    target_test_file_name = data_hyperparameters['target_test_file_name']
    print(f"[INFO] load target train data from {target_train_file_name}.")
    print(f"[INFO] load target test data from {target_test_file_name}.")

    print("[INFO] Load train data")
    target_train_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_valid_loader, _ = get_income_census_dataloaders(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    # perform target training
    plat_target = FederatedTargetLearner(model=model,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=500)
    plat_target.set_model_save_info(census_finetune_target_root_dir)

    plat_target.train_target_with_alternating(global_epochs=500,
                                              top_epochs=1,
                                              bottom_epochs=1,
                                              lr=lr,
                                              task_id=target_task_id,
                                              metric=valid_metric,
                                              momentum=momentum,
                                              weight_decay=weight_decay)

    # load best model
    model.load_model(root=census_finetune_target_root_dir,
                     task_id=target_task_id,
                     load_global_classifier=True,
                     timestamp=None)

    print("[DEBUG] Global classifier Model Parameter After train:")
    model.print_parameters()

    acc, auc, ks = test_classifier(model, target_valid_loader, 'test')
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")


def train_no_adaptation(data_tag,
                        census_no_ad_root_dir,
                        learner_hyperparameters,
                        data_hyperparameters):
    # hyper-parameters
    apply_feature_group = learner_hyperparameters['apply_feature_group']
    train_data_tag = learner_hyperparameters['train_data_tag']
    momentum = learner_hyperparameters['momentum']
    weight_decay = learner_hyperparameters['weight_decay']
    batch_size = learner_hyperparameters['batch_size']
    lr = learner_hyperparameters['lr']
    epoch_patience = learner_hyperparameters['epoch_patience']
    epochs = learner_hyperparameters['epochs']
    valid_metric = learner_hyperparameters['valid_metric']

    source_train_file_name = data_hyperparameters['source_train_file_name']
    target_train_file_name = data_hyperparameters['target_train_file_name']
    source_test_file_name = data_hyperparameters['source_test_file_name']
    target_test_file_name = data_hyperparameters['target_test_file_name']
    src_tgt_train_file_name = data_hyperparameters['src_tgt_train_file_name']
    data_file_name_dict = {"src": source_train_file_name,
                           "tgt": target_train_file_name,
                           "all": src_tgt_train_file_name}

    print(f"[INFO] load source train from: {source_train_file_name}.")
    print(f"[INFO] load target train from: {target_train_file_name}.")
    print(f"[INFO] load source test from: {source_test_file_name}.")
    print(f"[INFO] load target test from: {target_test_file_name}.")

    timestamp = get_timestamp()
    date = get_current_date() + "_" + data_tag + "_census_no_da_w_fg" if apply_feature_group else "_census_no_da_wo_fg"
    tries = 1
    for version in range(tries):
        hyperparameter_dict = {"lr": lr, "bs": batch_size, "ts": timestamp, "ve": version}
        task_id = date + "_" + train_data_tag + "_" + create_id_from_hyperparameters(hyperparameter_dict)
        print("[INFO] perform task:{0}".format(task_id))

        if apply_feature_group:
            print("[INFO] feature grouping applied")
            model = fg_dann.create_fg_census_global_model(num_wide_feature=5)
        else:
            print("[INFO] no feature grouping applied")
            model = no_fg_dann.create_no_fg_census_global_model(aggregation_dim=4, num_wide_feature=5)
        print("[INFO] model created.")
        src_train_loader, _ = get_income_census_dataloaders(
            ds_file_name=data_file_name_dict[train_data_tag], batch_size=batch_size, split_ratio=1.0)
        tgt_train_loader, _ = get_income_census_dataloaders(
            ds_file_name=data_file_name_dict['tgt'], batch_size=batch_size, split_ratio=1.0)

        src_valid_loader, _ = get_income_census_dataloaders(
            ds_file_name=source_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)
        tgt_valid_loader, _ = get_income_census_dataloaders(
            ds_file_name=target_test_file_name, batch_size=batch_size * 4, split_ratio=1.0)

        plat = FederatedDAANLearner(model=model,
                                    source_da_train_loader=src_train_loader,
                                    source_val_loader=src_valid_loader,
                                    target_da_train_loader=tgt_train_loader,
                                    target_val_loader=tgt_valid_loader,
                                    epoch_patience=epoch_patience,
                                    validation_batch_interval=5)
        plat.set_model_save_info(census_no_ad_root_dir)

        plat.train_wo_adaption(epochs=epochs,
                               lr=lr,
                               train_source=True,
                               metric=valid_metric,
                               task_id=task_id,
                               momentum=momentum,
                               weight_decay=weight_decay)

        # load best model
        model.load_model(root=census_no_ad_root_dir,
                         task_id=task_id,
                         load_global_classifier=True,
                         timestamp=None)

        print("[DEBUG] Global classifier Model Parameter After train:")
        model.print_parameters()

        acc, auc, ks = test_classifier(model, tgt_valid_loader, 'test')
        print(f"acc:{acc}, auc:{auc}, ks:{ks}")
