from datasets.lending_dataloader import get_lending_dataloader
from experiments.lending_loan.train_lending_dann import create_global_model_wrapper

from utils import test_classifier

if __name__ == "__main__":

    # dann_root_folder = "lending_dann"
    # dann_task_id = "20200902_w_XAI_BCE_01_lr_0001"
    # dann_task_id = "20200907_w_XAI_BCE_01_lr_0001"
    # dann_task_id = "20200907_w_XAI_BCE_02_lr_0001"
    # dann_task_id = "20200907_w_XAI_BCE_03_lr_0001"
    # dann_task_id = "20200908_w_XAI_BCE_01_lr_0001"

    dann_root_folder = "lending_target"
    # dann_task_id = "20200902_w_XAI_BCE_01_lr_0001_target_finetune_2"
    # dann_task_id = "20200907_w_XAI_BCE_01_lr_0001_target_finetune_2"
    # dann_task_id = "20200907_w_XAI_BCE_02_lr_0001_target_finetune"
    # dann_task_id = "20200907_w_XAI_BCE_03_lr_0001_target_finetune_2"
    # dann_task_id = "20200908_w_XAI_BCE_01_lr_0001_target_finetune"
    dann_task_id = "20200909_w_XAI_BCE_01_lr_0001_target_finetune"


    # Load models
    wrapper = create_global_model_wrapper()

    # load trained model
    timestamp = None
    wrapper.load_model(root=dann_root_folder, task_id=dann_task_id, timestamp=timestamp)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    wrapper.print_parameters()

    # Load data
    batch_size = 1024
    loan_2018_dir = "../../data/lending_club_bundle_archive/loan_data_v2/loan_processed_2018/"
    # loan_2018_dir = "../../data/lending_club_bundle_archive/loan_processed_2018/"
    print("[INFO] Load test data")
    loan_2018_test_loader = get_lending_dataloader(dir=loan_2018_dir, batch_size=batch_size, data_mode="test")
    print("[INFO] data loaded ...")

    acc, auc, ks = test_classifier(wrapper, loan_2018_test_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
