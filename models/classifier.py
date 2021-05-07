import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import init_weights


class TransformMatrix(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformMatrix, self).__init__()
        # self.transform = nn.Sequential(
        #     nn.Linear(in_features=input_dim, out_features=output_dim),
        # )

        self.fc = nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)

        # self.fc.apply(init_weights)

    def transform(self, x):
        return self.fc(x)

    def transpose_transform(self, x):
        return F.linear(x, self.fc.weight.t())


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10)
        )

    def forward(self, x):
        x = self.classifier(x)
        return F.softmax(x, dim=1)


class RegionClassifier(nn.Module):
    def __init__(self, input_dim):
        super(RegionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=3),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.classifier(x)


activation_fn = nn.LeakyReLU()


# activation_fn = nn.Sigmoid()
# activation_fn = Mish()


class CensusRegionAggregator(nn.Module):
    def __init__(self, input_dim):
        super(CensusRegionAggregator, self).__init__()
        self.aggregator = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1),
            # activation_fn
            # nn.Sigmoid()
            # nn.LeakyReLU()
            # nn.Tanh()
        )

        # self.aggregator.apply(init_weights)

    def forward(self, x):
        return self.aggregator(x)


class IdentityRegionAggregator(nn.Module):
    def __init__(self):
        super(IdentityRegionAggregator, self).__init__()

    def forward(self, x):
        return x


class GlobalClassifier(nn.Module):
    def __init__(self, input_dim):
        super(GlobalClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1),
        )
        # self.classifier.apply(init_weights)

    def forward(self, x):
        return self.classifier(x)
