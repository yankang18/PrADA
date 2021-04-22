import errno
import gzip
import os
import pickle

import torch
import torch.utils.data as data
from PIL import Image
# import essential packages
from six.moves import urllib
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import datasets
from torchvision import transforms


# MNIST-M
class MNISTM(data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'mnist_m_train.pt'
    test_file = 'mnist_m_test.pt'

    def __init__(self,
                 root, mnist_root="data",
                 train=True,
                 transform=None, target_transform=None,
                 download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = \
                torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
            print(f"train_data {self.train_data.shape}")
        else:
            self.test_data, self.test_labels = \
                torch.load(os.path.join(self.root, self.processed_folder, self.test_file))
            print(f"train_data {self.test_data.shape}")

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(type(img))
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.training_file)) and \
               os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.test_file))

    def download(self):
        """Download the MNIST data."""

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print('Downloading ' + self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        print(f"file_path {file_path}")
        if not os.path.exists(file_path.replace('.gz', '')):
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        # load MNIST-M images from pkl file
        with open(file_path.replace('.gz', ''), "rb") as f:
            mnist_m_data = pickle.load(f, encoding='bytes')
        # print(f"mnist_m_data shape:{mnist_m_data.shape}")
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root,
                                            train=True,
                                            download=True).targets
        mnist_test_labels = datasets.MNIST(root=self.mnist_root,
                                           train=False,
                                           download=True).targets
        print(f"mnist_train_labels shape:{mnist_train_labels.shape}")
        print(f"mnist_test_labels shape:{mnist_test_labels.shape}")
        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('MNISTM Done!')


def get_mnistm_datasets(root="data/pytorch/MNIST-M", download=False):
    transform = transforms.Compose([
                                    # transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))
                                    ])
    mnistm_train_dataset = MNISTM(root=root, mnist_root="data/pytorch/MNIST", train=True, download=download,
                                  transform=transform)
    mnistm_valid_dataset = MNISTM(root=root, mnist_root="data/pytorch/MNIST", train=True, download=download,
                                  transform=transform)
    mnistm_test_dataset = MNISTM(root=root, mnist_root="data/pytorch/MNIST", train=False, transform=transform)
    return mnistm_train_dataset, mnistm_valid_dataset, mnistm_test_dataset


def get_dataloaders(mnistm_train_dataset, mnistm_valid_dataset, mnistm_test_dataset, batch_size=64, num_workers=1):
    indices = list(range(len(mnistm_train_dataset)))
    print(f"num of train data: {len(indices)}")
    validation_size = 5000
    train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    mnist_train_loader = DataLoader(
        mnistm_train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        sampler=train_sampler,
        num_workers=num_workers
    )

    mnist_valid_loader = DataLoader(
        mnistm_valid_dataset,
        batch_size=batch_size * 2,
        # shuffle=True,
        sampler=valid_sampler,
        num_workers=num_workers
    )

    mnist_test_loader = DataLoader(
        mnistm_test_dataset,
        # shuffle=True,
        batch_size=batch_size * 2,
        num_workers=num_workers
    )

    return mnist_train_loader, mnist_valid_loader, mnist_test_loader


def get_mnistm_dataloaders(root="data/pytorch/MNIST-M", download=False, batch_size=64, num_workers=2):
    train_dataset, valid_dataset, test_dataset = get_mnistm_datasets(root=root, download=download)
    print("shape", train_dataset[0][0].shape)
    return get_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset = get_mnistm_dataloaders(download=True)
#
#     print(f"train_dataset {len(train_dataset)}")
#     print(f"valid_dataset {len(valid_dataset)}")
#     print(f"test_dataset {len(test_dataset)}")

    # root = os.path.expanduser(root)
    # file_path = os.path.join(root, raw_folder, filename)

    # root = "/Users/yankang/Documents/Pycharm_Projects/DANN/datasets/data/pytorch/MNIST-M/raw"
    # file_path = os.path.join(root, "keras_mnistm.pkl.gz")
    # with open(file_path.replace('.gz', ''), "rb") as f:
    #     mnist_m_data = pickle.load(f, encoding='bytes')
    # print(f"mnist_m_data:{mnist_m_data}")
    # mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
    # mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])
    #
    # print(f"mnist_m_train_data shape {mnist_m_train_data.shape}")
    # print(f"mnist_m_test_data shape {mnist_m_test_data.shape}")
