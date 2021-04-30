from datasets.lending_dataloader import get_lending_dataloader
from models.experiment_target_learner import FederatedTargetLearner
from experiments.lending_loan.train_lending_dann_no_fg import create_global_model_wrapper
from utils import test_classification

if __name__ == "__main__":
    dann_root_folder = "lending_dann"
    dann_task_id = "20200901_no_XAI_BCE_04_lr_0003_w_36"

    # Load models
    wrapper = create_global_model_wrapper(pos_class_weight=0.3)

    #
    # load pre-trained model
    #
    wrapper.load_model(root=dann_root_folder, task_id=dann_task_id, load_global_classifier=False, timestamp=None)
    # dann_exp_result = load_dann_experiment_result(root=dann_root_folder, task_id=dann_task_id)
    dann_exp_result = None

    print("[DEBUG] Global classifier Model Parameter Before train:")
    wrapper.print_parameters()

    # Load data
    batch_size = 512
    # B_dir = "../../data/cell_manager/B_train_data_2/"
    loan_2018_dir = "../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"
    print("[INFO] Load train data")
    tgt_train_loader = get_lending_dataloader(dir=loan_2018_dir, batch_size=batch_size, data_mode="train")
    # tgt_train_loader = get_lending_dataloader(dir=loan_2018_dir, batch_size=batch_size, data_mode="train_nus")
    print("[INFO] Load val data")
    tgt_val_loader = get_lending_dataloader(dir=loan_2018_dir, batch_size=batch_size * 2, data_mode="val")
    print("[INFO] Load test data")
    tgt_test_loader = get_lending_dataloader(dir=loan_2018_dir, batch_size=batch_size * 2, data_mode="test")
    print("[INFO] data loaded ...")

    acc, auc, ks = test_classification(wrapper, tgt_test_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")

    #
    # perform target training
    #
    plat_target = FederatedTargetLearner(model=wrapper,
                                         target_train_loader=tgt_train_loader,
                                         # target_val_loader=B_val_loader,
                                         target_val_loader=tgt_test_loader,
                                         patience=500,
                                         max_global_epochs=500)
    plat_target.set_model_save_info("lending_target")

    target_task_id = "20200901_target_transfer_no_XAI_BCE_01_lr_0002_w_36"
    # plat_target.set_fine_tuning_region_indices(fine_tuning_region_index_list=[3, 5])
    # plat_target.train_target_with_alternating(global_epochs=500, top_epochs=1, bottom_epochs=1, lr=2e-3,
    #                                           task_id=target_task_id, dann_exp_result=dann_exp_result)
    plat_target.train_target_as_whole(global_epochs=500, lr=1e-2, task_id=target_task_id, dann_exp_result=dann_exp_result)

    print("[DEBUG] Global classifier Model Parameter After train:")
    wrapper.print_parameters()

    acc, auc, ks = test_classification(wrapper, tgt_test_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
