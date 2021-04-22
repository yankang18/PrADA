from datasets.mnist_dataloader import get_mnist_dataloaders
from datasets.mnistm_dataloader import get_mnistm_dataloaders
from models.feature_extractor import MNISTRegionExpandInputExtractor
from models.classifier import GlobalClassifier, RegionClassifier
from models.experiment_dann_learner import FederatedDAANLearner
from models.discriminator import RegionDiscriminator
from models.wrapper_obsolete import GlobalModelWrapper, RegionModelWrapper


def create_region_model_wrapper(input_dim):
    extractor = MNISTRegionExpandInputExtractor()
    region_classifier = RegionClassifier(input_dim=input_dim)
    discriminator = RegionDiscriminator(input_dim=input_dim)
    return RegionModelWrapper(extractor=extractor,
                              aggregator=region_classifier,
                              discriminator=discriminator)


if __name__ == "__main__":
    input_dim = 24 * 2 * 2
    wrapper_list = list()
    for i in range(9):
        wrapper_list.append(create_region_model_wrapper(input_dim))

    classifier = GlobalClassifier(input_dim=27)
    wrapper = GlobalModelWrapper(wrapper_list, classifier)

    root = "./datasets/data/pytorch/MNIST"
    mnist_train_loader, mnist_valid_loader, mnist_test_loader = get_mnist_dataloaders(root=root, batch_size=64,
                                                                                      download=False)
    root = "./datasets/data/pytorch/MNIST-M"
    mnistm_train_loader, mnistm_valid_loader, mnistm_test_loader = get_mnistm_dataloaders(root=root, batch_size=64,
                                                                                          download=False)

    plat = FederatedDAANLearner(model=wrapper,
                                source_train_loader=mnist_train_loader, source_val_loader=mnist_valid_loader,
                                target_train_loader=mnistm_train_loader, target_val_loader=mnistm_valid_loader)
    plat.set_model_save_info("multiple_region_dann")
    plat.train_dann(epochs=100, lr=1e-2)
