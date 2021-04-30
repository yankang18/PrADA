from datasets.ppd_dataloader import get_pdd_dataloaders_ob
from models.experiment_target_learner import FederatedTargetLearner
from experiments.kesci_ppd.train_ppd_dann_no_fg import create_global_model_wrapper
from utils import test_classification

if __name__ == "__main__":
    dann_root_folder = "ppd_dann"
    # dann_task_id = '20200118_no_fg_512_0.0006_5'  # DA: 73.5, 35.5; TA: 73.8, 36.7
    # dann_task_id = '20200118_no_fg_512_0.0007_1' # DA: 73.8, 35.6; TA: 74.4, 37.0
    # dann_task_id = '20200118_no_fg_512_0.0008_1'  # DA: 74.6, 38.0; TA: 74.4, 37.6; TA_ft:75.0, 38.5
    # dann_task_id = '20200122_no_fg_pw1.0_bs512_lr0.001_v2'  # DA: 73.99, 37.74; TA: 74.1, 38.9
    dann_task_id = '20200122_no_fg_pw1.0_bs512_lr0.001_v2'  # DA: 73.97, 37.45; TA: 74.3, 38.4

    # Load models
    wrapper = create_global_model_wrapper(pos_class_weight=1.0)

    # load pre-trained model
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

    batch_size = 256
    # batch_size = 128
    # batch_size = 64
    print("[INFO] Load train data")
    target_train_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_train_file_name, batch_size=batch_size, split_ratio=1.0)

    print("[INFO] Load test data")
    target_valid_loader, _ = get_pdd_dataloaders_ob(
        ds_file_name=target_test_file_name, batch_size=batch_size, split_ratio=1.0)

    # perform target training
    plat_target = FederatedTargetLearner(model=wrapper,
                                         target_train_loader=target_train_loader,
                                         target_val_loader=target_valid_loader,
                                         patience=800,
                                         max_global_epochs=400)
    plat_target.set_model_save_info("pdd_target")

    lr = 3e-4
    version = 4
    target_task_id = dann_task_id + "_target_finetune_" + str(batch_size) + "_" + str(lr) + "_" + str(version)
    plat_target.train_target_with_alternating(global_epochs=400, top_epochs=1, bottom_epochs=1, lr=lr,
                                              task_id=target_task_id, dann_exp_result=dann_exp_result)
    # plat_target.train_target_as_whole(global_epochs=100, lr=4e-4, task_id=target_task_id,
    #                                   dann_exp_result=dann_exp_result)

    print("[DEBUG] Global classifier Model Parameter After train:")
    wrapper.print_parameters()

    acc, auc, ks = test_classification(wrapper, target_valid_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
