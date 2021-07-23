from datasets.cell_manage_dataloader import get_cell_manager_dataloader_ob
from experiments.tencent_cell_manage.train_cell_dann import create_global_dann_model

from utils import test_classifier

if __name__ == "__main__":
    # dann_root_folder = "cell_manage_dann"
    # dann_task_id = "20200614_01_lr_0003_w_10"
    dann_root_folder = "cell_manage_target"
    # dann_task_id = "20200614_target_001"
    # dann_task_id = "20200615_logitic1_02_lr_0003_w_10"
    dann_task_id = "20200729_BCE_03_lr_00001_w_10"
    # Load models
    wrapper = create_global_dann_model()

    # load trained model
    # timestamp = 1592169013
    # timestamp = 1592172805
    # timestamp = 1592174603

    # timestamp = 1592179495
    # timestamp = 1592177620
    # timestamp = 1592180547
    timestamp = None
    wrapper.load_model(root=dann_root_folder, task_id=dann_task_id, timestamp=timestamp)

    print("[DEBUG] Global classifier Model Parameter Before train:")
    wrapper.print_parameters()

    # Load data
    batch_size = 512
    B_dir = "../../data/cell_manager/C_train_data_2/"
    print("[INFO] Load test data")
    B_test_loader = get_cell_manager_dataloader_ob(dir=B_dir, batch_size=batch_size * 2, data_mode="test")
    print("[INFO] data loaded ...")

    acc, auc, ks = test_classifier(wrapper, B_test_loader)
    print(f"acc:{acc}, auc:{auc}, ks:{ks}")
