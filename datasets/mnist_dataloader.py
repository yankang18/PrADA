import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms


def get_datasets(root="data/pytorch/MNIST", download=False):
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
    #                                                      (0.32780124, 0.32292358, 0.32056796)),
    #                                 ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])
    transform = transforms.Compose([
                                    # transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))
                                    ])
    mnist_train_dataset = datasets.MNIST(root=root, train=True, download=download,
                                         transform=transform)
    mnist_valid_dataset = datasets.MNIST(root=root, train=True, download=download,
                                         transform=transform)
    mnist_test_dataset = datasets.MNIST(root=root, train=False, transform=transform)
    return mnist_train_dataset, mnist_valid_dataset, mnist_test_dataset


def get_dataloaders(mnist_train_dataset, mnist_valid_dataset, mnist_test_dataset, batch_size=64, num_workers=1):
    indices = list(range(len(mnist_train_dataset)))
    print(f"num of train data: {len(indices)}")
    validation_size = 5000
    train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    mnist_train_loader = DataLoader(
        mnist_train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        sampler=train_sampler,
        num_workers=num_workers
    )

    mnist_valid_loader = DataLoader(
        mnist_valid_dataset,
        batch_size=batch_size * 2,
        # shuffle=True,
        sampler=valid_sampler,
        num_workers=num_workers
    )

    mnist_test_loader = DataLoader(
        mnist_test_dataset,
        batch_size=batch_size * 2,
        # shuffle=True,
        num_workers=num_workers
    )

    return mnist_train_loader, mnist_valid_loader, mnist_test_loader


def get_mnist_dataloaders(root="data/pytorch/MNIST", download=False, batch_size=64, num_workers=2):
    train_dataset, valid_dataset, test_dataset = get_datasets(root=root, download=download)
    return get_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = get_datasets(download=False)
    train_loader, valid_loader, test_loader = get_dataloaders(train_dataset, valid_dataset, test_dataset)
    print(f"train_dataset {len(train_dataset)}")
    print(f"valid_dataset {len(valid_dataset)}")
    print(f"test_dataset {len(test_dataset)}")

    print(f"train_dataset {len(train_loader.dataset)}")
    print(f"valid_dataset {len(valid_loader.dataset)}")
    print(f"test_dataset {len(test_loader.dataset)}")
