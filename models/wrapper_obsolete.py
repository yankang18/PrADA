import json
import os

import torch
import torch.nn as nn


class ModelWrapper():

    def __init__(self, extractor, classifier, discriminator=None):
        self.extractor = extractor
        self.classifier = classifier
        self.discriminator = discriminator
        self.discriminator_set = False if discriminator is None else True

        self.classifier_criterion = nn.CrossEntropyLoss()
        self.discriminator_criterion = nn.CrossEntropyLoss()

    def save_model(self, root, file_name):

        # build folder structure
        model_folder = os.path.join(root, "models")
        task_folder = os.path.join(root, "tasks")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(task_folder):
            os.makedirs(task_folder)

        timestamp = get_timestamp()

        # save models
        feature_classifier_name = "feature_classifier_" + str(timestamp)
        feature_extractor_name = "feature_extractor_" + str(timestamp)
        domain_discriminator_name = "domain_discriminator_" + str(timestamp)
        feature_classifier_path = os.path.join(model_folder, feature_classifier_name)
        feature_extractor_path = os.path.join(model_folder, feature_extractor_name)
        domain_discriminator_path = os.path.join(model_folder, domain_discriminator_name)
        torch.save(self.classifier.state_dict(), feature_classifier_path)
        torch.save(self.extractor.state_dict(), feature_extractor_path)
        torch.save(self.discriminator.state_dict(), domain_discriminator_path)

        # save task meta file
        task_meta = dict()
        task_meta["feature_classifier"] = feature_classifier_path
        task_meta["feature_extractor"] = feature_extractor_path
        task_meta["domain_discriminator"] = domain_discriminator_path
        print(f"[INFO] saved classifier model to: {feature_classifier_path}")
        print(f"[INFO] saved extractor model to: {feature_extractor_path}")
        print(f"[INFO] saved discriminator model to: {domain_discriminator_path}")

        file_name = str(file_name) + "_" + str(timestamp) + '.json'
        file_full_name = os.path.join(task_folder, file_name)
        with open(file_full_name, 'w') as outfile:
            json.dump(task_meta, outfile)
        return task_meta

    def get_num_regions(self):
        return 1

    def check_discriminator_exists(self):
        if self.discriminator_set is False:
            raise RuntimeError('Discriminator not set.')

    def change_to_train_mode(self):
        self.extractor.train()
        self.classifier.train()
        if self.discriminator_set:
            self.discriminator.train()

    def change_to_eval_mode(self):
        self.extractor.eval()
        self.classifier.eval()
        if self.discriminator_set:
            self.discriminator.eval()

    def parameters(self):
        if self.discriminator_set:
            return list(self.extractor.parameters()) + list(self.classifier.parameters()) + list(
                self.discriminator.parameters())
        else:
            return list(self.extractor.parameters()) + list(self.classifier.parameters())

    def compute_classification_loss(self, data, label):
        batch_feat = self.extractor(data)
        pred = self.classifier(batch_feat)
        return self.classifier_criterion(pred, label)

    def compute_total_loss(self, source_data, target_data, source_label, domain_source_labels, domain_target_labels,
                           **kwargs):
        alpha = kwargs["alpha"]

        # calculate loss on source domain including classification loss and domain loss
        source_feat = self.extractor(source_data)
        pred = self.classifier(source_feat)
        class_loss = self.classifier_criterion(pred, source_label)
        source_domain_pred = self.discriminator(source_feat, alpha)
        source_domain_loss = self.discriminator_criterion(source_domain_pred, domain_source_labels)

        # calculate loss on target domain including only domain loss
        target_feat = self.extractor(target_data)
        target_domain_pred = self.discriminator(target_feat, alpha)
        target_domain_loss = self.discriminator_criterion(target_domain_pred, domain_target_labels)

        # accumulate all losses
        total_loss = class_loss + source_domain_loss + target_domain_loss
        return total_loss

    def calculate_classifier_correctness(self, data, label):
        feat = self.extractor(data)
        pred = self.classifier(feat)
        pred_cls = pred.tgt_train_data.max(1)[1]
        return pred_cls.eq(label).sum().item()

    def calculate_domain_discriminator_correctness(self, data, is_source=True):
        if is_source:
            labels = torch.zeros(data.shape[0]).long()
        else:
            labels = torch.ones(data.shape[0]).long()
        feat = self.extractor(data)
        pred = self.discriminator(feat, alpha=0)
        pred_cls = pred.tgt_train_data.max(1)[1]
        return pred_cls.eq(labels).sum().item()


class GlobalModelWrapper(Wrapper):
    def __init__(self, complex_wrapper_list, classifier):
        self.complex_wrapper_list = complex_wrapper_list
        self.classifier = classifier
        self.classifier_criterion = nn.CrossEntropyLoss()

    def save_model(self, root, task_id, file_name, timestamp=None):
        """Save trained model."""

        # build folder structure
        model_folder = os.path.join(root, "models")
        task_folder = os.path.join(root, "tasks")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        if not os.path.exists(task_folder):
            os.makedirs(task_folder)

        timestamp = get_timestamp()

        global_classifier = "global_classifier_" + str(timestamp)
        global_classifier_path = os.path.join(model_folder, global_classifier)
        model_meta = dict()
        model_meta["global_part"] = dict()
        model_meta["global_part"]["classifier"] = global_classifier_path
        torch.save(self.classifier.state_dict(), global_classifier_path)
        print(f"[INFO] save global classifier model to: {global_classifier_path}")

        model_meta["region_part"] = dict()
        for wrapper in self.complex_wrapper_list:
            res = wrapper.save_model(root, timestamp)
            model_meta["region_part"]["region_1"] = res

        file_name = str(file_name) + "_" + str(timestamp) + '.json'
        file_full_name = os.path.join(task_folder, file_name)
        with open(file_full_name, 'w') as outfile:
            json.dump(model_meta, outfile)

        return model_meta

    def freeze_top(self, is_freeze=False):
        for param in self.classifier.parameters():
            param.requires_grad = not is_freeze

    def freeze_bottom(self, is_freeze=False):
        for wrapper in self.complex_wrapper_list:
            for param in wrapper.parameters():
                param.requires_grad = not is_freeze

    def get_num_regions(self):
        return len(self.complex_wrapper_list)

    def check_discriminator_exists(self):
        for wrapper in self.complex_wrapper_list:
            if wrapper.change_to_train_mode() is False:
                raise RuntimeError('Discriminator not set.')

    def change_to_train_mode(self):
        self.classifier.train()
        for wrapper in self.complex_wrapper_list:
            wrapper.change_to_train_mode()

    def change_to_eval_mode(self):
        self.classifier.eval()
        for wrapper in self.complex_wrapper_list:
            wrapper.change_to_eval_mode()

    def parameters(self):
        param_list = list(self.classifier.parameters())
        for wrapper in self.complex_wrapper_list:
            param_list += wrapper.parameters()
        return param_list

    def partition_data(self, data):
        # print("--")
        # row_dim = 32
        # col_dim = 32
        # row_stop = row_dim // 2 + 1
        # col_stop = col_dim // 2 + 1
        # partition_row_dim = 16
        # partition_col_dim = 16
        # row_stride = 4
        # col_stride = 4
        row_dim = 28
        col_dim = 28
        row_stop = row_dim // 2 + 1
        col_stop = col_dim // 2 + 1
        partition_row_dim = 14
        partition_col_dim = 14
        row_stride = 7
        col_stride = 7
        data_partition_list = list()
        for row_start_index in range(0, row_stop, row_stride):
            row_end_index = row_start_index + partition_row_dim
            for col_start_index in range(0, col_stop, col_stride):
                col_end_index = col_start_index + partition_col_dim
                # print(f"{row_start_index}:{row_end_index} {col_start_index}:{col_end_index}")
                data_partition_list.append(data[:, :, row_start_index:row_end_index, col_start_index:col_end_index])
        return data_partition_list

    def compute_total_loss(self, source_data, target_data, source_label, domain_source_labels, domain_target_labels,
                           **kwargs):

        src_data_par_list = self.partition_data(source_data)
        tgt_data_par_list = self.partition_data(target_data)

        total_domain_loss = torch.tensor(0.)
        output_list = []
        for wrapper, src_data, tgt_data in zip(self.complex_wrapper_list, src_data_par_list, tgt_data_par_list):
            domain_loss, output = wrapper.compute_loss(src_data, tgt_data, source_label,
                                                       domain_source_labels,
                                                       domain_target_labels,
                                                       **kwargs)
            output_list.append(output)
            total_domain_loss += domain_loss

        output = torch.cat(output_list, dim=1)
        # print(f"[DEBUG] {self.__class__.__name__} output shape:{output.shape}")
        pred = self.classifier(output)
        class_loss = self.classifier_criterion(pred, source_label)

        total_loss = class_loss + total_domain_loss
        return total_loss

    def compute_classification_loss(self, data, label):
        output = self.calculate_classifier_output(data)
        pred = self.classifier(output)
        class_loss = self.classifier_criterion(pred, label)
        return class_loss

    def calculate_classifier_output(self, data):
        data_partition_list = self.partition_data(data)
        output_list = []
        for wrapper, data_par in zip(self.complex_wrapper_list, data_partition_list):
            output_list.append(wrapper.compute_output_list(data_par))
        output = torch.cat(output_list, dim=1)
        return output

    def calculate_classifier_correctness(self, data, label):
        # data_partition_list = self.partition_data(data)
        # output_list = []
        # for wrapper, data_par in zip(self.complex_wrapper_list, data_partition_list):
        #     output_list.append(wrapper.compute_output(data_par))
        # output = torch.cat(output_list, dim=1)
        output = self.calculate_classifier_output(data)
        pred = self.classifier(output)
        pred_cls = pred.tgt_train_data.max(1)[1]
        return pred_cls.eq(label).sum().item()

    def calculate_domain_discriminator_correctness(self, data, is_source=True):
        data_partition_list = self.partition_data(data)
        output_list = []
        for wrapper, data_par in zip(self.complex_wrapper_list, data_partition_list):
            output_list.append(wrapper.calculate_domain_discriminator_correctness(data_par, is_source=is_source))
        # print(f"[DEBUG] is_source:{is_source}\t domain_discriminator_correctness {output_list}")
        return output_list


class RegionModelWrapper(object):

    def __init__(self, extractor, aggregator, discriminator):
        self.extractor = extractor
        self.aggregator = aggregator
        self.discriminator = discriminator
        self.discriminator_set = False if discriminator is None else True

        self.classifier_criterion = nn.CrossEntropyLoss()
        self.discriminator_criterion = nn.CrossEntropyLoss()

    def save_model(self, model_root, timestamp):
        feature_aggregator_name = "feature_aggregator_" + str(timestamp)
        feature_extractor_name = "feature_extractor_" + str(timestamp)
        domain_discriminator_name = "domain_discriminator_" + str(timestamp)
        feature_aggregator_path = os.path.join(model_root, feature_aggregator_name)
        feature_extractor_path = os.path.join(model_root, feature_extractor_name)
        domain_discriminator_path = os.path.join(model_root, domain_discriminator_name)
        torch.save(self.aggregator.state_dict(), feature_aggregator_path)
        torch.save(self.extractor.state_dict(), feature_extractor_path)
        torch.save(self.discriminator.state_dict(), domain_discriminator_path)

        task_meta = dict()
        task_meta["feature_aggregator"] = feature_aggregator_path
        task_meta["feature_extractor"] = feature_extractor_path
        task_meta["domain_discriminator"] = domain_discriminator_path
        print(f"saved classifier model to: {feature_aggregator_path}")
        print(f"saved extractor model to: {feature_extractor_path}")
        print(f"saved discriminator model to: {domain_discriminator_path}")
        return task_meta

    def check_discriminator_exists(self):
        if self.discriminator_set is False:
            raise RuntimeError('Discriminator not set.')

    def change_to_train_mode(self):
        self.extractor.train()
        self.aggregator.train()
        if self.discriminator_set:
            self.discriminator.train()

    def change_to_eval_mode(self):
        self.extractor.eval()
        self.aggregator.eval()
        if self.discriminator_set:
            self.discriminator.eval()

    def parameters(self):
        if self.discriminator_set:
            return list(self.extractor.parameters()) + list(self.aggregator.parameters()) + list(
                self.discriminator.parameters())
        else:
            return list(self.extractor.parameters()) + list(self.aggregator.parameters())

    def compute_output(self, data):
        batch_feat = self.extractor(data)
        output = self.aggregator(batch_feat)
        return output

    def compute_total_loss(self, source_data, target_data, source_label, domain_source_labels, domain_target_labels,
                           **kwargs):
        alpha = kwargs["alpha"]

        # # calculate loss on source domain including classification loss and domain loss
        # source_feat = self.extractor(source_data)
        # # print(f"[DEBUG] {self.__class__.__name__} source_feat shape:{source_feat.shape}")
        # output = self.classifier(source_feat)
        #
        # source_domain_pred = self.discriminator(source_feat, alpha)
        # source_domain_loss = self.discriminator_criterion(source_domain_pred, domain_source_labels)
        #
        # # calculate loss on target domain including only domain loss
        # target_feat = self.extractor(target_data)
        # # print(f"[DEBUG] {self.__class__.__name__} target_feat shape:{target_feat.shape}")
        # target_domain_pred = self.discriminator(target_feat, alpha)
        # target_domain_loss = self.discriminator_criterion(target_domain_pred, domain_target_labels)
        #
        # # accumulate domain loss
        # domain_loss = source_domain_loss + target_domain_loss

        num_sample = source_data.shape[0] + source_data.shape[0]
        source_feat = self.extractor(source_data)
        target_feat = self.extractor(target_data)

        output = self.aggregator(source_feat)

        domain_feat = torch.cat((source_feat, target_feat), dim=0)
        domain_labels = torch.cat((domain_source_labels, domain_target_labels), dim=0)
        perm = torch.randperm(num_sample)
        domain_feat = domain_feat[perm]
        domain_labels = domain_labels[perm]

        domain_output = self.discriminator(domain_feat, alpha)
        domain_loss = self.discriminator_criterion(domain_output, domain_labels)

        return domain_loss, output

    def calculate_domain_discriminator_correctness(self, data, is_source=True):
        if is_source:
            labels = torch.zeros(data.shape[0]).long()
        else:
            labels = torch.ones(data.shape[0]).long()
        feat = self.extractor(data)
        pred = self.discriminator(feat, alpha=0)
        pred_cls = pred.tgt_train_data.max(1)[1]
        return pred_cls.eq(labels).sum().item()
