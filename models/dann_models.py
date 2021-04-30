import json
import os

import torch
import torch.nn as nn

from utils import get_latest_timestamp


def create_embedding(size):
    # return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))
    return nn.Embedding(*size, _weight=torch.zeros(*size).normal_(0, 0.01))


def create_embeddings(embedding_meta_dict):
    embedding_dict = dict()
    for key, value in embedding_meta_dict.items():
        embedding_dict[key] = create_embedding(value)
    return embedding_dict


class FeatureCrossComputer(object):

    def compute(self, feature_group_list):
        fg_int_list = [e for e in feature_group_list]
        start_index = 1
        for fg in feature_group_list:
            for i in range(start_index, len(feature_group_list)):
                fg_2 = feature_group_list[i]
                fg_int_list.append(torch.cat((fg, fg_2), dim=1))
            start_index += 1
        return fg_int_list


def compute_feature_crossing(feature_cross_model, feature_group_list):
    if feature_cross_model:
        return feature_cross_model.compute(feature_group_list)
    return feature_group_list


class GlobalModel(object):
    def __init__(self, classifier, regional_model_list, embedding_dict, partition_data_fn, beta=1.0,
                 pos_class_weight=1.0, loss_name="CE", feature_cross_model=None, feature_interactive_model=None):
        self.classifier = classifier
        self.regional_model_list = list() if regional_model_list is None else regional_model_list
        self.feature_cross_model = feature_cross_model
        self.feature_interactive_model = feature_interactive_model
        self.embedding_dict = embedding_dict
        self.loss_name = loss_name
        if loss_name == "CE":
            self.classifier_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_class_weight]))
        elif loss_name == "BCE":
            self.classifier_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_class_weight))
        else:
            raise RuntimeError(f"Does not support loss:{loss_name}")
        self.beta = beta
        self.partition_data_fn = partition_data_fn

    def print_parameters(self, print_all=False):
        print("-" * 50)
        print("Global models:")
        for name, param in self.classifier.named_parameters():
            # if param.requires_grad:
            print(f"{name}: {param.data}, {param.requires_grad}")
            # print(f"{name}: {param.requires_grad}")
        if print_all:
            print("Region models:")
            for wrapper in self.regional_model_list:
                wrapper.print_parameters()
                # for param in wrapper.parameters():
                #     if param.requires_grad:
                #         print(f"{param.train_data}, {param.requires_grad}")
            print("Embedding models:")
            for emb in self.embedding_dict.values():
                for name, param in emb.named_parameters():
                    # if param.requires_grad:
                    # print(f"{name}: {param.data}, {param.requires_grad}")
                    print(f"{name}: {param.requires_grad}")
        print("-" * 50)

    def get_global_classifier_parameters(self, get_tensor=False):
        param_dict = dict()
        for name, param in self.classifier.named_parameters():
            # print("----->", name, param, param.requires_grad)
            # if param.requires_grad:
            if get_tensor:
                param_dict[name] = param
            else:
                param_dict[name] = param.data.tolist()
        return param_dict

    def load_model(self, root, task_id, task_meta_file_name="task_meta", load_global_classifier=True, timestamp=None):
        task_folder = "task_" + task_id
        task_folder_path = os.path.join(root, task_folder)
        if not os.path.exists(task_folder_path):
            raise FileNotFoundError(f"{task_folder_path} is not found.")
        print(f"[INFO] load model from:{task_folder_path}")

        if timestamp is None:
            timestamp = get_latest_timestamp("models_checkpoint", task_folder_path)
            print(f"[INFO] get latest timestamp {timestamp}")

        model_checkpoint_folder = "models_checkpoint_" + str(timestamp)
        model_checkpoint_folder = os.path.join(task_folder_path, model_checkpoint_folder)
        if not os.path.exists(model_checkpoint_folder):
            raise FileNotFoundError(f"{model_checkpoint_folder} is not found.")

        task_meta_file_name = str(task_meta_file_name) + "_" + str(timestamp) + '.json'
        task_meta_file_path = os.path.join(model_checkpoint_folder, task_meta_file_name)
        if not os.path.exists(task_meta_file_path):
            raise FileNotFoundError(f"{task_meta_file_path} is not found.")

        with open(task_meta_file_path) as json_file:
            print(f"[INFO] load task meta file from {task_meta_file_path}")
            task_meta_dict = json.load(json_file)

        if load_global_classifier:
            global_classifier_path = task_meta_dict["global_part"]["classifier"]
            self.classifier.load_state_dict(torch.load(global_classifier_path))
            print(f"[INFO] load global classifier from {global_classifier_path}")

        # load embeddings
        embedding_meta_dict = task_meta_dict["global_part"]["embeddings"]
        for key, emb_path in embedding_meta_dict.items():
            print(f"[INFO] load embedding of [{key}] from {emb_path}")
            self.embedding_dict[key].load_state_dict(torch.load(emb_path))

        # load region models
        region_part_dict = task_meta_dict["region_part"]
        num_region = len(region_part_dict)
        assert num_region == len(self.regional_model_list)

        for idx, region_model in enumerate(self.regional_model_list):
            region = "region_" + str(idx)
            region_model.load_model(region_part_dict[region]["models"])

        # load interactive model
        if self.feature_interactive_model:
            interactive_model_part_dict = task_meta_dict["interactive_part"]
            self.feature_interactive_model.load_model(interactive_model_part_dict)

    def save_model(self, root, task_id, file_name="task_meta", timestamp=None):
        """Save trained model."""

        if timestamp is None:
            raise RuntimeError("timestamp is missing.")

        task_folder = "task_" + task_id
        task_root_folder = os.path.join(root, task_folder)
        if not os.path.exists(task_root_folder):
            os.makedirs(task_root_folder)

        model_checkpoint_folder = "models_checkpoint_" + str(timestamp)
        model_checkpoint_folder = os.path.join(task_root_folder, model_checkpoint_folder)
        if not os.path.exists(model_checkpoint_folder):
            os.makedirs(model_checkpoint_folder)

        extension = ".pth"

        # save global model
        global_classifier = "global_classifier_" + str(timestamp) + extension
        global_classifier_path = os.path.join(model_checkpoint_folder, global_classifier)
        model_meta = dict()
        model_meta["global_part"] = dict()
        model_meta["global_part"]["classifier"] = global_classifier_path
        torch.save(self.classifier.state_dict(), global_classifier_path)
        print(f"[INFO] saved global classifier model to: {global_classifier_path}")

        # save embeddings
        embedding_meta_dict = dict()
        for key, emb in self.embedding_dict.items():
            emb_file_name = "embedding_" + key + "_" + str(timestamp) + extension
            emb_path = os.path.join(model_checkpoint_folder, emb_file_name)
            torch.save(emb.state_dict(), emb_path)
            print(f"[INFO] saved embedding of [{key}] to: {emb_path}")
            embedding_meta_dict[key] = emb_path
        model_meta["global_part"]["embeddings"] = embedding_meta_dict

        # save region models
        model_meta["region_part"] = dict()
        for idx, regional_model in enumerate(self.regional_model_list):
            region = "region_" + str(idx)
            res = regional_model.save_model(model_checkpoint_folder, region + "_" + str(timestamp) + extension)
            model_meta["region_part"][region] = dict()
            model_meta["region_part"][region]["order"] = idx
            model_meta["region_part"][region]["models"] = res

        # save interactive model
        if self.feature_interactive_model:
            interactive_model = self.feature_interactive_model.save_model(model_checkpoint_folder,
                                                                          str(timestamp) + extension)
            model_meta["interactive_part"] = interactive_model

        file_name = str(file_name) + "_" + str(timestamp) + '.json'
        file_full_name = os.path.join(model_checkpoint_folder, file_name)
        with open(file_full_name, 'w') as outfile:
            json.dump(model_meta, outfile)

        return model_meta

    def freeze_top(self, is_freeze=False):
        for param in self.classifier.parameters():
            param.requires_grad = not is_freeze

    def freeze_bottom(self, is_freeze=False, region_idx_list=None):
        # freeze region models
        if region_idx_list is None:
            for rg_model in self.regional_model_list:
                for param in rg_model.parameters():
                    param.requires_grad = not is_freeze
        else:
            print(f"white region idx list:{region_idx_list}")
            for region_idx in region_idx_list:
                for param in self.regional_model_list[region_idx].parameters():
                    param.requires_grad = not is_freeze

        # freeze embedding
        for emb in self.embedding_dict.values():
            for param in emb.parameters():
                param.requires_grad = not is_freeze

        # freeze interaction model
        if self.feature_interactive_model: self.feature_interactive_model.freeze(is_freeze)

    # TODO add feature_interactive_model
    def freeze_bottom_extractors(self, is_freeze=False, region_idx_list=None):
        if region_idx_list is None:
            for rg_model in self.regional_model_list:
                for param in rg_model.extractor_parameters():
                    param.requires_grad = not is_freeze
        else:
            print(f"freeze region idx list:{region_idx_list}")
            for region_idx in region_idx_list:
                for param in self.regional_model_list[region_idx].extractor_parameters():
                    param.requires_grad = not is_freeze

    # TODO add feature_interactive_model
    def freeze_bottom_aggregators(self, is_freeze=False, region_idx_list=None):
        if region_idx_list is None:
            for rg_model in self.regional_model_list:
                for param in rg_model.aggregator_parameters():
                    param.requires_grad = not is_freeze
        else:
            print(f"freeze region idx list:{region_idx_list}")
            for region_idx in region_idx_list:
                for param in self.regional_model_list[region_idx].aggregator_parameters():
                    param.requires_grad = not is_freeze

    def get_num_regions(self):
        num_feature_groups = len(self.regional_model_list)
        if self.feature_interactive_model:
            num_feature_groups += self.feature_interactive_model.get_num_feature_groups()
        return num_feature_groups

    def check_discriminator_exists(self):
        for rg_model in self.regional_model_list:
            if rg_model.has_discriminator() is False:
                raise RuntimeError('Discriminator not set.')

        if self.feature_interactive_model: self.feature_interactive_model.check_discriminator_exists()

    def change_to_train_mode(self):
        self.classifier.train()
        for rg_model in self.regional_model_list:
            rg_model.change_to_train_mode()
        for embedding in self.embedding_dict.values():
            embedding.train()
        if self.feature_interactive_model: self.feature_interactive_model.change_to_train_mode()

    def change_to_eval_mode(self):
        self.classifier.eval()
        for rg_model in self.regional_model_list:
            rg_model.change_to_eval_mode()
        for embedding in self.embedding_dict.values():
            embedding.eval()
        if self.feature_interactive_model: self.feature_interactive_model.change_to_eval_mode()

    def parameters(self):
        param_list = list(self.classifier.parameters())
        for rg_model in self.regional_model_list:
            param_list += rg_model.parameters()
        for embedding in self.embedding_dict.values():
            param_list += embedding.parameters()
        if self.feature_interactive_model:
            param_list += self.feature_interactive_model.parameters()
        return param_list

    def _combine_features(self, feat_dict):
        """

        :param feat_dict:
        :return:
        """

        features_list = []
        # print("feat_dict", feat_dict)
        embeddings = feat_dict.get("embeddings")
        if embeddings is not None:
            for key, feat in embeddings.items():
                embedding = self.embedding_dict[key]
                feat = feat.long()
                # print("@@:", key, feat)
                emb = embedding(feat)
                features_list.append(emb)
        non_embedding = feat_dict.get("non_embedding")
        if non_embedding is not None:
            for _, feat in non_embedding.items():
                # print(f"feat shape:{feat.shape}")
                if len(feat.shape) == 1:
                    features_list.append(feat.reshape(-1, 1))
                else:
                    features_list.append(feat)
        comb_feat_tensor = torch.cat(features_list, dim=1)
        # print(f"comb_feat_tensor shape:{comb_feat_tensor.shape}, {comb_feat_tensor}")
        return comb_feat_tensor

    def is_regional_model_list_empty(self):
        if self.regional_model_list is None or len(self.regional_model_list) == 0:
            return True
        return False

    def compute_feature_group_loss(self,
                                   total_domain_loss,
                                   output_list,
                                   src_feat_gp_list,
                                   tgt_feat_gp_list,
                                   domain_source_labels,
                                   domain_target_labels,
                                   **kwargs):
        if self.is_regional_model_list_empty() is False:
            for regional_model, src_fg, tgt_fg in zip(self.regional_model_list, src_feat_gp_list, tgt_feat_gp_list):
                domain_loss, output = regional_model.compute_loss(src_fg,
                                                                  tgt_fg,
                                                                  domain_source_labels,
                                                                  domain_target_labels,
                                                                  **kwargs)
                output_list.append(output)
                total_domain_loss += domain_loss
        return total_domain_loss, output_list

    def compute_feature_group_interaction_loss(self,
                                               total_domain_loss,
                                               output_list,
                                               src_feat_gp_list,
                                               tgt_feat_gp_list,
                                               domain_source_labels,
                                               domain_target_labels,
                                               **kwargs):
        if self.feature_interactive_model:
            intr_domain_loss, int_output_list = self.feature_interactive_model.compute_loss(src_feat_gp_list,
                                                                                            tgt_feat_gp_list,
                                                                                            domain_source_labels,
                                                                                            domain_target_labels,
                                                                                            **kwargs)
            total_domain_loss += intr_domain_loss
            output_list += int_output_list
        return total_domain_loss, output_list

    def compute_total_loss(self, source_data, target_data, source_label, domain_source_labels, domain_target_labels,
                           **kwargs):

        src_wide_list, src_deep_par_list = self.partition_data_fn(source_data)
        tgt_wide_list, tgt_deep_par_list = self.partition_data_fn(target_data)

        # total_domain_loss = torch.tensor(0.)
        # output_list = []
        # for regional_model, src_data, tgt_data in zip(self.regional_model_list, src_deep_par_list, tgt_deep_par_list):
        #     src_feat_group = self._combine_features(src_data)
        #     tgt_feat_group = self._combine_features(tgt_data)
        #     domain_loss, output = regional_model.compute_total_loss(src_feat_group, tgt_feat_group, source_label,
        #                                                             domain_source_labels,
        #                                                             domain_target_labels,
        #                                                             **kwargs)
        #     output_list.append(output)
        #     total_domain_loss += domain_loss

        src_feat_gp_list = list()
        tgt_feat_gp_list = list()
        for src_data, tgt_data in zip(src_deep_par_list, tgt_deep_par_list):
            src_feat_gp_list.append(self._combine_features(src_data))
            tgt_feat_gp_list.append(self._combine_features(tgt_data))

        src_feat_gp_list = compute_feature_crossing(self.feature_cross_model, src_feat_gp_list)
        tgt_feat_gp_list = compute_feature_crossing(self.feature_cross_model, tgt_feat_gp_list)

        total_domain_loss = torch.tensor(0.)
        output_list = []
        total_domain_loss, output_list = self.compute_feature_group_loss(total_domain_loss,
                                                                         output_list,
                                                                         src_feat_gp_list,
                                                                         tgt_feat_gp_list,
                                                                         domain_source_labels,
                                                                         domain_target_labels,
                                                                         **kwargs)
        # for regional_model, src_fg, tgt_fg in zip(self.regional_model_list, src_feat_gp_list, tgt_feat_gp_list):
        #     domain_loss, output = regional_model.compute_loss(src_fg,
        #                                                       tgt_fg,
        #                                                       domain_source_labels,
        #                                                       domain_target_labels,
        #                                                       **kwargs)
        #     output_list.append(output)
        #     total_domain_loss += domain_loss

        total_domain_loss, output_list = self.compute_feature_group_interaction_loss(total_domain_loss,
                                                                                     output_list,
                                                                                     src_feat_gp_list,
                                                                                     tgt_feat_gp_list,
                                                                                     domain_source_labels,
                                                                                     domain_target_labels,
                                                                                     **kwargs)

        output_list = src_wide_list + output_list if len(src_wide_list) > 0 else output_list
        output = torch.cat(output_list, dim=1)
        # print(f"[DEBUG] output shape:{output.shape}")
        pred = self.classifier(output)

        if self.loss_name == "CE":
            source_label = source_label.flatten().long()
        else:
            # using BCELogitLoss
            source_label = source_label.reshape(-1, 1).type_as(pred)

        class_loss = self.classifier_criterion(pred, source_label)

        total_loss = class_loss + self.beta * total_domain_loss
        return total_loss

    def _calculate_feature_group_output_list(self, deep_par_list):
        # print("=" * 30)
        fg_list = []
        for fg_data in deep_par_list:
            fg_list.append(self._combine_features(fg_data))

        output_list = None if self.is_regional_model_list_empty() is True \
            else [regional_model.compute_output(fg) for regional_model, fg in
                  zip(self.regional_model_list, fg_list)]

        # compute output from feature interaction model that wraps feature interactions
        fgi_output_list = None if self.feature_interactive_model is None \
            else self.feature_interactive_model.compute_output_list(fg_list)

        return output_list, fgi_output_list

    def calculate_global_classifier_input_vector(self, data):
        wide_list, deep_par_list = self.partition_data_fn(data)
        if len(deep_par_list) == 0:
            output_list = wide_list
        else:
            fg_output_list, fgi_output_list = self._calculate_feature_group_output_list(deep_par_list)
            if fg_output_list is None:
                all_fg_output_list = fgi_output_list
            else:
                all_fg_output_list = fg_output_list + fgi_output_list if fgi_output_list else fg_output_list
            output_list = wide_list + all_fg_output_list if len(wide_list) > 0 else all_fg_output_list

        # print("[DEBUG] output_list", output_list, len(output_list))
        output = torch.cat(output_list, dim=1)
        # print("[DEBUG] output", output, output.shape)
        return output

    def compute_classification_loss(self, data, label):
        output = self.calculate_global_classifier_input_vector(data)
        prediction = self.classifier(output)
        if self.loss_name == "CE":
            label = label.flatten().long()
        else:
            # using BCELogitLoss
            label = label.reshape(-1, 1).type_as(prediction)
        class_loss = self.classifier_criterion(prediction, label)
        return class_loss

    def calculate_classifier_correctness(self, data, label):
        output = self.calculate_global_classifier_input_vector(data)
        pred = self.classifier(output)
        if self.loss_name == "CE":
            # using CrossEntropyLoss
            pred_prob = torch.softmax(pred.data, dim=1)
            pos_prob = pred_prob[:, 1]
            y_pred_tag = pred_prob.max(1)[1]
        else:
            # using BCELogitLoss
            pos_prob = torch.sigmoid(pred.flatten())
            y_pred_tag = torch.round(pos_prob).long()

        correct_results_sum = y_pred_tag.eq(label).sum().item()
        return correct_results_sum, y_pred_tag, pos_prob

    def calculate_domain_discriminator_correctness(self, data, is_source=True):
        _, deep_par_list = self.partition_data_fn(data)

        fg_list = [self._combine_features(fg_feat) for fg_feat in deep_par_list]
        fg_list = compute_feature_crossing(self.feature_cross_model, fg_list)

        output_list = list()
        if self.is_regional_model_list_empty() is False:
            for regional_model, fg in zip(self.regional_model_list, fg_list):
                output_list.append(regional_model.calculate_domain_discriminator_correctness(fg, is_source=is_source))

        if self.feature_interactive_model:
            output_list = output_list + self.feature_interactive_model.calculate_domain_discriminator_correctness(
                fg_list, is_source=is_source)
        # print(f"[DEBUG] is_source:{is_source}\t domain_discriminator_correctness {output_list}")
        return output_list


class RegionalModel(object):

    def __init__(self, extractor, aggregator, discriminator):
        self.extractor = extractor
        self.aggregator = aggregator
        self.discriminator = discriminator
        self.discriminator_set = False if discriminator is None else True
        self.discriminator_criterion = nn.CrossEntropyLoss()

    def has_discriminator(self):
        return self.discriminator_set

    def load_model(self, model_dict):
        feature_aggregator_path = model_dict["feature_aggregator"]
        feature_extractor = model_dict["feature_extractor"]
        domain_discriminator = model_dict["domain_discriminator"]
        self.aggregator.load_state_dict(torch.load(feature_aggregator_path))
        self.extractor.load_state_dict(torch.load(feature_extractor))
        self.discriminator.load_state_dict(torch.load(domain_discriminator))
        print(f"[INFO] load aggregator model from {feature_aggregator_path}")
        print(f"[INFO] load extractor model from {feature_extractor}")
        print(f"[INFO] load discriminator model from {domain_discriminator}")

    def save_model(self, model_root, appendix):
        feature_aggregator_name = "feature_aggregator_" + str(appendix)
        feature_extractor_name = "feature_extractor_" + str(appendix)
        domain_discriminator_name = "domain_discriminator_" + str(appendix)
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
        print(f"[INFO] saved aggregator model to: {feature_aggregator_path}")
        print(f"[INFO] saved extractor model to: {feature_extractor_path}")
        print(f"[INFO] saved discriminator model to: {domain_discriminator_path}")
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

    def print_parameters(self):
        print("--" * 50)
        print("==> region classifiers")
        for name, param in self.aggregator.named_parameters():
            # print(f"{name}: {param.data}, {param.requires_grad}")
            print(f"{name}: {param.requires_grad}")
        print("==> region extractors")
        for name, param in self.extractor.named_parameters():
            # print(f"{name}: {param.data}, {param.requires_grad}")
            print(f"{name}: {param.requires_grad}")
        if self.discriminator_set:
            print("==> region discriminators")
            for name, param in self.discriminator.named_parameters():
                # print(f"{name}: {param.data}, {param.requires_grad}")
                print(f"{name}: {param.requires_grad}")

    def parameters(self):
        if self.discriminator_set:
            return list(self.extractor.parameters()) + list(self.aggregator.parameters()) + list(
                self.discriminator.parameters())
        else:
            return list(self.extractor.parameters()) + list(self.aggregator.parameters())

    def extractor_parameters(self):
        return list(self.extractor.parameters())

    def aggregator_parameters(self):
        return list(self.aggregator.parameters())

    def compute_output(self, data):
        batch_feat = self.extractor(data)
        output = self.aggregator(batch_feat)
        return output

    def compute_loss(self, source_data, target_data, domain_source_labels, domain_target_labels,
                     **kwargs):
        alpha = kwargs["alpha"]

        num_sample = source_data.shape[0] + target_data.shape[0]
        source_feat = self.extractor(source_data)
        target_feat = self.extractor(target_data)

        output = self.aggregator(source_feat)

        # print("[DEBUG] domain_source_labels should all be zero: \n", domain_source_labels)
        # print("[DEBUG] domain_target_labels should all be one: \n", domain_target_labels)

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
        pred_cls = pred.data.max(dim=1)[1]
        res = pred_cls.eq(labels).sum().item()
        return res
