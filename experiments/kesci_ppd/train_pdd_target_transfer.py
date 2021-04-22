from datasets.pdd_dataloader import get_pdd_dataloaders_ob
from models.experiment_target_learner import FederatedTargetLearner
from experiments.kesci_ppd.train_pdd_dann import construct_global_model
from utils import test_classification, get_timestamp


if __name__ == "__main__":
    dann_root_folder = "ppd_dann"
    # dann_task_id = '20200118_PDD_512_0.001_1'  # DA: 75.5, 39.8; TA: 75.7, 39.4
    # dann_task_id = '20200118_PDD_512_0.001_0_1610923436'  # DA: 75.6, 39.9; TA: 75.8, 40.1
    # dann_task_id = '20200118_PDD_256_0.0012_0_1610938417'  # DA: 75.9, 39.6; TA: 76.0, 39.7; TA_ft: 76.0, 40.3
    # dann_task_id = '20200118_PDD_512_0.0012_0_1610938417'  # DA: 75.7, 39.8; TA: 76.0, 40.7; TA_ft: 76.2, 41.5
    # dann_task_id = '20200119_PDD_256_0.001_0_1611043561'  # DA: 76.0, 40.7; TA:75.9, 41.4, TA_ft: 76.2, 41.8
    # dann_task_id = '20200118_PDD_512_0.001_2'  # DA: 75.8, 41.2; TA: 75.8, 41.3; TA_ft: 76.2, 42.5

    # DA: 75.7, 40.5; TA: 76.0, 42.6; 76.2, 42.5
    # dann_task_id = '20200121_PDD_pw1.0_bs512_lr0.0012_v0_t1614542852'

    # DA: 75.3, 40.8; TA: 75.7, 41.2
    # dann_task_id = '20200301_PDD_pw1.0_bs256_lr0.001_v0_t1614548613'

    # DA: 75.25, 41.0; TA: 75.4, 40.2
    # dann_task_id = '20200301_PDD_pw1.0_bs512_lr0.001_v0_t1614545680'

    # DA: 75.4, 40.8;
    # dann_task_id = '20210401_PDD_pw1.0_bs256_lr0.0015_v0_t1617224419'

    # dann_task_id = '20210401_PDD_pw1.0_bs256_lr0.0015_v0_t1617246075'

    # # DA: 75.3, 40.39;
    # dann_task_id = '20210331_PDD_pw1.0_bs256_lr0.0015_v0_t1617217008'

    dann_task_id = '20210407_PDD_pw1.0_bs256_lr0.001_v0_t1617735343'

    # DA: 75.9, 41.33;
    # dann_task_id = '20210331_PDD_pw1.0_bs256_lr0.0015_v0_t1617157138'

    # Load models
    wrapper = construct_global_model(pos_class_weight=1.0)

    # load pre-trained model,
    # If load_global_classifier = True, meaning we are going to fine-tune the global classier.
    # Otherwise, we train the global classifier from the scratch.
    load_global_classifier = False
    wrapper.load_model(root=dann_root_folder,
                       task_id=dann_task_id,
                       load_global_classifier=load_global_classifier,
                       timestamp=None)
    # dann_exp_result = load_dann_experiment_result(root=dann_root_folder, task_id=dann_task_id)
    dann_exp_result = None

    print("[DEBUG] Global classifier Model Parameter Before train:")
    wrapper.print_parameters()

    # Load data
    data_dir = "/Users/yankang/Documents/Data/Data_Open_Analysis_master/Kesci_PPD/PPD_data_v1/"
    target_train_file_name = data_dir + 'PPD_2014_10to12_train.csv'
    target_test_file_name = data_dir + 'PPD_2014_10to12_test.csv'

    batch_size = 128
    print("[INFO] Load train data")
    target_train_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_valid_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    # perform target training
    plat_target = FederatedTargetLearner(wrapper=wrapper,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=500)
    plat_target.set_model_save_info("pdd_target")

    timestamp = get_timestamp()
    version = 1
    # lr = 2e-4
    lr = 3e-4
    # lr = 8e-5
    date = "20210407"
    glr = "ft_glr" if load_global_classifier else "rt_glr"
    tag = "_target_" + date + "_" + glr + "_" + str(batch_size) + "_" + str(lr) + "_v" + str(version)
    target_task_id = dann_task_id + tag
    plat_target.train_target_with_alternating(global_epochs=400, top_epochs=1, bottom_epochs=1, lr=lr,
                                              task_id=target_task_id, dann_exp_result=dann_exp_result)
    # plat_target.train_target_as_whole(global_epochs=200, lr=1e-3, task_id=target_task_id,
    #                                   dann_exp_result=dann_exp_result)

    print("[DEBUG] Global classifier Model Parameter After train:")
    wrapper.print_parameters()

    acc, auc, ks = test_classification(wrapper, target_valid_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
