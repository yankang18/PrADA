from experiments.income_census.global_config import fine_tune_fg_dann_hyperparameters, data_hyperparameters
from experiments.income_census.train_census_utils import finetune_census_dann
from experiments.income_census.train_census_fg_dann import create_global_model

if __name__ == "__main__":
    dann_task_id = ''
    census_pretain_model_root_dir = data_hyperparameters['census_no-fg_dann_model_dir']
    census_finetune_target_root_dir = data_hyperparameters['census_no-fg_ft_target_model_dir']
    finetune_census_dann(dann_task_id,
                         census_pretain_model_root_dir,
                         census_finetune_target_root_dir,
                         fine_tune_fg_dann_hyperparameters,
                         data_hyperparameters,
                         create_global_model)

    # # Hyper-parameters
    # load_global_classifier = pre_train_fg_dann_hyperparameters['load_global_classifier']
    # using_interaction = pre_train_fg_dann_hyperparameters['using_interaction']
    # momentum = pre_train_fg_dann_hyperparameters['momentum']
    # weight_decay = pre_train_fg_dann_hyperparameters['weight_decay']
    # batch_size = pre_train_fg_dann_hyperparameters['batch_size']
    # lr = pre_train_fg_dann_hyperparameters['lr']
    # epoch_patience = pre_train_fg_dann_hyperparameters['epoch_patience']
    # max_epochs = pre_train_fg_dann_hyperparameters['max_epochs']
    # valid_metric = pre_train_fg_dann_hyperparameters['valid_metric']
    #
    # data_tag = data_hyperparameters['data_tag']
    # data_dir = data_hyperparameters['data_dir']
    # census_dann_root_dir = data_hyperparameters['census_fg_dann_model_dir']
    # census_ft_target_root_dir = data_hyperparameters['census_fg_ft_target_model_dir']
    #
    # date = get_current_date()
    # timestamp = get_timestamp()
    #
    # # Load models
    # model = create_global_model(using_interaction=using_interaction)
    #
    # # load pre-trained model
    # model.load_model(root=census_dann_root_dir,
    #                  task_id=dann_task_id,
    #                  load_global_classifier=load_global_classifier,
    #                  timestamp=None)
    #
    # print("[DEBUG] Global classifier Model Parameter Before train:")
    # model.print_parameters()
    #
    # # Load data
    # target_train_file_name = data_hyperparameters['target_train_file_name']
    # target_test_file_name = data_hyperparameters['target_test_file_name']
    # print(f"[INFO] load target train data from {target_train_file_name}.")
    # print(f"[INFO] load target test data from {target_test_file_name}.")
    #
    # print("[INFO] Load train data")
    # target_train_loader, _ = get_income_census_dataloaders(
    #     ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)
    #
    # print("[INFO] Load test data")
    # target_valid_loader, _ = get_income_census_dataloaders(
    #     ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)
    #
    # # perform target training
    # plat_target = FederatedTargetLearner(model=model,
    #                                      target_train_loader=target_train_loader,
    #                                      target_val_loader=target_valid_loader,
    #                                      patience=800,
    #                                      max_global_epochs=500)
    # plat_target.set_model_save_info(census_ft_target_root_dir)
    # glr = "ft_glr" if load_global_classifier else "rt_glr"
    #
    # hyperparameter_dict = {"lr": lr, "bs": batch_size, "ts": timestamp}
    # appendix = create_id_from_hyperparameters(hyperparameter_dict)
    # target_task_id = dann_task_id + "@target_" + date + "-" + glr + "_" + appendix
    # plat_target.train_target_with_alternating(global_epochs=500,
    #                                           top_epochs=1,
    #                                           bottom_epochs=1,
    #                                           lr=lr,
    #                                           task_id=target_task_id,
    #                                           dann_exp_result=None,
    #                                           metric=valid_metric,
    #                                           weight_decay=weight_decay)
    #
    # # load best model
    # model.load_model(root=census_ft_target_root_dir,
    #                  task_id=target_task_id,
    #                  load_global_classifier=True,
    #                  timestamp=None)
    #
    # print("[DEBUG] Global classifier Model Parameter After train:")
    # model.print_parameters()
    #
    # acc, auc, ks = test_classifier(model, target_valid_loader, 'test')
    # print(f"acc:{acc}, auc:{auc}, ks:{ks}")
