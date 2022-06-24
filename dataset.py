import os
import urllib.request
import tarfile

import torch
import torchvision
import torchvision.transforms as transforms

from Params.params_dataset import *


class Dataset:
    def __init__(self, name, split, network, additional_transform=None, adversarial_attack=None, batch_size=1000):
        self.name = name
        self.split = split
        self.network = network
        self.batch_size = batch_size
        self.additional_transform = additional_transform
        self.adversarial_attack = adversarial_attack

        self._check_accepted_dataset()
        self._check_accepted_network()
        self._check_accepted_transforms()
        self._check_accepted_attack()

        self.mean_transform = mean_transform[network]
        self.std_transform = std_transform[network]
        self._set_transforms()

        self._load_dataset()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)

    def _check_accepted_dataset(self):
        data_split = self.name + "_" + self.split
        if data_split not in accepted_datasets:
            raise ValueError("Accepted dataset/split pairs are: %s" % str(accepted_datasets)[1:-1])

    def _check_accepted_network(self):
        accepted_networks = list(mean_transform.keys())
        if self.network not in accepted_networks:
            raise ValueError("Accepted network architectures are: %s" % str(accepted_networks)[1:-1])

    def _check_accepted_transforms(self):
        accepted_transforms = list(additional_transforms.keys()) + [None]
        if self.additional_transform not in accepted_transforms:
            raise ValueError("Accepted data transforms are: %s" % str(accepted_transforms)[1:-1])

    def _check_accepted_attack(self):
        if self.adversarial_attack not in accepted_attacks + [None]:
            raise ValueError("Accepted attacks are: %s" % str(accepted_attacks)[1:-1])

    def _set_transforms(self):
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(self.mean_transform, self.std_transform)]
        if self.additional_transform is not None:
            transform_list.insert(0, additional_transforms[self.additional_transform])

        self.transform = transforms.Compose(transform_list)

    def _load_dataset(self):
        if self.name == "cifar10":
            self._load_cifar10()
        elif self.name == "cifar100":
            self._load_cifar100()
        elif self.name == "svhn":
            self._load_svhn()
        elif self.name == "tiny_imagenet":
            self._load_tinyimagenet()
        elif self.name == "lsun":
            self._load_lsun()
        else:
            pass

    def _load_cifar10(self):
        is_train = self.split == "train"
        self.dataset = torchvision.datasets.CIFAR10(root=datasets_path, train=is_train,
                                                    download=True, transform=self.transform)

    def _load_cifar100(self):
        is_train = self.split == "train"
        self.dataset = torchvision.datasets.CIFAR100(root=datasets_path, train=is_train,
                                                     download=True, transform=self.transform)

    def _load_svhn(self):
        self.dataset = torchvision.datasets.SVHN(root=datasets_path, split=self.split,
                                                 download=True, transform=self.transform)

    def _load_tinyimagenet(self):
        if not os.path.exists(path_tinyImagenet):
            u = urllib.request.urlopen(downloadUrl_tinyImagenet + "?dl=1")
            data = u.read()
            u.close()

            path_tar = datasets_path + "/Imagenet_resize.tar.gz"
            with open(path_tar, "wb") as f:
                f.write(data)

            my_tar = tarfile.open(path_tar)
            my_tar.extractall(datasets_path)  # specify which folder to extract to
            my_tar.close()

        self.dataset = torchvision.datasets.ImageFolder(path_tinyImagenet,
                                                        transform=self.transform)

    def _load_lsun(self):
        if not os.path.exists(path_lsun):
            u = urllib.request.urlopen(downloadUrl_lsun + "?dl=1")
            data = u.read()
            u.close()

            path_tar = datasets_path + "/LSUN_resize.tar.gz"
            with open(path_tar, "wb") as f:
                f.write(data)

            my_tar = tarfile.open(path_tar)
            my_tar.extractall(datasets_path)  # specify which folder to extract to
            my_tar.close()

        self.dataset = torchvision.datasets.ImageFolder(path_lsun,
                                                        transform=self.transform)
