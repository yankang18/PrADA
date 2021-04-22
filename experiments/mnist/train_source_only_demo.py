from datasets.mnist_dataloader import get_mnist_dataloaders
from datasets.mnistm_dataloader import get_mnistm_dataloaders
from models.feature_extractor import MNISTExpandInputExtractor
from models.classifier import Classifier
from models.experiment_dann_learner import FederatedDAANLearner
from models.wrapper_obsolete import ModelWrapper

if __name__ == "__main__":
    extractor = MNISTExpandInputExtractor()
    classifier = Classifier(input_dim=48 * 4 * 4)

    wrapper = ModelWrapper(extractor, classifier)

    root = "./datasets/data/pytorch/MNIST"
    mnist_train_loader, mnist_valid_loader, mnist_test_loader = get_mnist_dataloaders(root=root, batch_size=64,
                                                                                      download=False)
    root = "./datasets/data/pytorch/MNIST-M"
    mnistm_train_loader, mnistm_valid_loader, mnistm_test_loader = get_mnistm_dataloaders(root=root, batch_size=64,
                                                                                          download=False)

    plat = FederatedDAANLearner(model=wrapper,
                                source_train_loader=mnist_train_loader, source_val_loader=mnist_valid_loader,
                                target_train_loader=mnistm_train_loader, target_val_loader=mnistm_valid_loader)
    plat.train_wo_adaption(epochs=100, lr=0.01)
