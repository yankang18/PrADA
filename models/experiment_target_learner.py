import torch
import torch.optim as optim

from models.experiment_dann_learner import adjust_learning_rate
from utils import get_timestamp, test_classification, save_dann_experiment_result


class FederatedTargetLearner(object):

    def __init__(self, model, target_train_loader, target_val_loader, patience=200, max_global_epochs=500):
        self.model = model
        # self.num_regions = self.wrapper.get_num_regions()
        self.target_train_loader = target_train_loader
        self.target_val_loader = target_val_loader
        self.patience = patience
        self.patient_count = None
        self.stop_training = None
        self.best_score = None
        self.timestamp_with_best_score = None
        self.root = "census_target"
        self.task_meta_file_name = "task_meta"
        self.dann_exp_result_dict = None
        self.l2_reg = None
        self.task_id = None
        self.weight_value_0 = None
        self.weight_value_1 = None
        self.has_weight_constraints = False
        self.fine_tuning_region_idx_list = None
        self.max_global_epochs = max_global_epochs

    def set_model_save_info(self, model_root):
        self.root = model_root

    def _check_exists(self):
        if self.model.check_discriminator_exists() is False:
            raise RuntimeError('Discriminator not set.')

    def _change_to_train_mode(self):
        # for wrapper in self.wrapper_list:
        self.model.change_to_train_mode()

    def _change_to_eval_mode(self):
        # for wrapper in self.wrapper_list:
        self.model.change_to_eval_mode()

    def save_model(self, task_id=None, timestamp=None):
        """Save trained model."""
        self.model.save_model(self.root, task_id, self.task_meta_file_name, timestamp=timestamp)

    def train_target_models(self, epochs, optimizer, lr, train_bottom_only=False, curr_global_epoch=0, global_epochs=1):
        num_batch = len(self.target_train_loader)
        loss = list()
        curr_lr = lr
        for ep in range(epochs):
            start_steps = curr_global_epoch * epochs * num_batch + ep * num_batch
            total_steps = self.max_global_epochs * epochs * num_batch

            for batch_idx, (batch_data, batch_label) in enumerate(self.target_train_loader):
                self._change_to_train_mode()

                # p = float(batch_idx + start_steps) / total_steps
                # curr_lr = adjust_learning_rate(optimizer, p, lr_0=lr, beta=0.75)
                # print(f"curr_global_epoch:{curr_global_epoch}, epochs:{epochs}")
                # print(f"total_steps:{total_steps}, start_steps:{start_steps}, batch_idx:{batch_idx}, p:{p}, lr:{curr_lr}")

                class_loss = self.model.compute_classification_loss(batch_data, batch_label)
                if self.has_weight_constraints and not train_bottom_only:
                    # print("[DEBUG] applying DANN lr parameters")
                    class_loss = class_loss + self.apply_lr_constraint()
                class_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss.append(class_loss.item())
                with torch.no_grad():
                    if (batch_idx + 1) % 5 == 0:
                        # if (batch_idx) % 1 == 0:
                        self._change_to_eval_mode()
                        tgt_cls_acc, tgt_cls_auc, tgt_cls_ks = test_classification(self.model,
                                                                                   self.target_val_loader,
                                                                                   "test target")
                        batch_per_epoch = 100. * batch_idx / len(self.target_train_loader)
                        print(f'[INFO] [{curr_global_epoch}/{global_epochs}]\t [{ep}/{epochs} ({batch_per_epoch:.0f}%)]'
                              f'\t loss:{class_loss}\t val target acc: {tgt_cls_acc:.6f}')
                        print(f'current learning rate:{curr_lr}')

                        # score = (tgt_cls_auc + tgt_cls_ks) / 2
                        score = (tgt_cls_auc + tgt_cls_acc) / 2
                        # score = tgt_cls_auc
                        # score = target_cls_acc
                        if score > self.best_score:
                            self.best_score = score
                            print(f"best score:{self.best_score} "
                                  f"with acc:{tgt_cls_acc}, auc:{tgt_cls_auc}, ks:{tgt_cls_ks}")
                            param_dict = self.model.get_global_classifier_parameters()
                            metric_dict = dict()
                            metric_dict["target_cls_acc"] = tgt_cls_acc
                            metric_dict["target_cls_auc"] = tgt_cls_auc
                            metric_dict["target_cls_ks"] = tgt_cls_ks
                            timestamp = get_timestamp()
                            self.timestamp_with_best_score = timestamp
                            save_dann_experiment_result(self.root, self.task_id, param_dict, metric_dict, timestamp)
                            self.save_model(self.task_id, timestamp=timestamp)
                            print("saved last model")
                            self.patient_count = 0
                        else:
                            self.patient_count += 1
                            if self.patient_count > self.patience:
                                print(
                                    f"[INFO] Early Stopped at target_cls_acc:{self.best_score} "
                                    f"with timestamp:{self.timestamp_with_best_score}")
                                # timestamp = get_timestamp()
                                # self.save_model(self.task_id, timestamp=timestamp)
                                self.stop_training = True
                                break

            if self.stop_training:
                break

        return loss

    def get_weight_constraints(self, dann_exp_result):
        self.has_weight_constraints = True
        self.weight_value_0 = dann_exp_result["lr_param"]["classifier.0.weight"][0]
        # self.weight_value_1 = dann_exp_result["lr_param"]["classifier.0.weight"][1]
        print("weight_value_0", self.weight_value_0)
        # print("weight_value_1", self.weight_value_1)

    def apply_lr_constraint(self):
        cls_parameters = self.model.get_global_classifier_parameters(get_tensor=True)
        # print("cls_parameters:", cls_parameters)
        weight_tensor_0 = cls_parameters["classifier.0.weight"][0]
        # weight_tensor_1 = cls_parameters["classifier.0.weight"][1]
        # print("weight_tensor_0", weight_tensor_0, weight_tensor_0.requires_grad)
        # print("weight_value", self.weight_value)
        reg_0 = weight_tensor_0 - torch.tensor(self.weight_value_0, dtype=torch.float)
        # reg_1 = weight_tensor_1 - torch.tensor(self.weight_value_1, dtype=torch.float)
        lbd = [0.001] * 11
        # lbd[2] = 0.1
        # lbd[4] = 0.1
        # lbd[5] = 0.1
        # print("reg", reg, reg.shape)
        # print(f"lbd: {lbd}")
        l2_reg_0 = torch.norm(torch.dot(reg_0, torch.tensor(lbd)))
        # l2_reg_1 = torch.norm(torch.dot(reg_1, torch.tensor(lbd)))
        # l2_reg = torch.norm(lbd * reg)
        # l2_reg = l2_reg_0 + l2_reg_1
        l2_reg = l2_reg_0
        return l2_reg

    def set_fine_tuning_region_indices(self, fine_tuning_region_index_list):
        self.fine_tuning_region_idx_list = fine_tuning_region_index_list

    def train_target_with_alternating(self, global_epochs, top_epochs, bottom_epochs,
                                      lr, task_id, dann_exp_result=None):
        self.task_id = task_id

        if dann_exp_result:
            print("[DEBUG] apply DANN lr parameters")
            self.get_weight_constraints(dann_exp_result)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.90)
        else:
            print("[DEBUG] Do not apply DANN lr parameters")
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.99, weight_decay=0.001)

        curr_lr = lr
        step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
        self.model.print_parameters()

        loss_list = list()
        self.patient_count = 0
        self.stop_training = False
        self.best_score = -float('inf')
        self.model.freeze_bottom(is_freeze=True)
        for ep in range(global_epochs):

            print(f"[INFO] ===> global epoch {ep}, start fine-tuning top")
            self.model.freeze_top(is_freeze=False)
            self.model.freeze_bottom(is_freeze=True)
            loss_list += self.train_target_models(top_epochs, optimizer, curr_lr, train_bottom_only=True,
                                                  curr_global_epoch=ep, global_epochs=global_epochs)

            print(f"[INFO] ===> global epoch {ep}, start fine-tuning bottom")
            self.model.freeze_top(is_freeze=False)
            self.model.freeze_bottom(is_freeze=True)
            loss_list += self.train_target_models(bottom_epochs, optimizer, curr_lr,
                                                  curr_global_epoch=ep, global_epochs=global_epochs)
            step_lr.step()
            curr_lr = step_lr.get_last_lr()
            print("[INFO] change learning rate to {0}".format(curr_lr))

            if self.stop_training:
                break

        print(f"loss_list:{loss_list}")

    # def train_target_as_whole(self, global_epochs, lr, task_id, dann_exp_result=None):
    #     self.task_id = task_id
    #
    #     if dann_exp_result:
    #         print("[INFO] apply DANN lr parameters")
    #         self.get_weight_constraints(dann_exp_result)
    #         optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
    #         # optimizer = optim.Adam(self.wrapper.parameters(), lr=lr)
    #     else:
    #         print("[INFO] Do not apply DANN lr parameters")
    #         optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    #
    #     self.patient_count = 0
    #     self.stop_training = False
    #     self.best_score = -float('inf')
    #     self.model.freeze_bottom(is_freeze=True)
    #     # self.wrapper.freeze_bottom_aggregators(is_freeze=True)
    #     # self.wrapper.freeze_bottom(is_freeze=False, region_idx_list=self.fine_tuning_region_idx_list)
    #     # self.wrapper.print_parameters(print_all=True)
    #
    #     # self.wrapper.freeze_bottom(is_freeze=True, region_idx_list=[0, 1, 3, 5, 6, 8])
    #     # self.wrapper.freeze_bottom(is_freeze=True, region_idx_list=[0, 1, 3, 4, 5, 6, 7, 8])
    #     # self.wrapper.freeze_bottom_extractors(is_freeze=True)
    #     for ep in range(global_epochs):
    #         self.train_target_models(epochs=1, optimizer=optimizer, lr=lr,
    #                                  curr_global_epoch=ep, global_epochs=global_epochs)
