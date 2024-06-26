import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from data.data_mono import DepthDataLoader
import os


def get_dataloader(config):
    if config.data_name == 'kitti':
        return get_kitti_dataloader(config)
    
    data_dir = config.data_dir
    batch_size = config.batch_size
    num_work = config.num_work

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.0), (1.0))])
    
    trans_cifar10 = transforms.Compose([transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.0), (1.0))])

    if config.data_name == 'mnist':
        train_dataset = dset.MNIST(root=data_dir, train=True, transform=trans, download=True)
        test_dataset = dset.MNIST(root=data_dir, train=False, transform=trans, download=True)
    elif config.data_name == 'fashion_mnist':
        train_dataset = dset.FashionMNIST(root=data_dir, train=True, transform=trans, download=True)
        test_dataset = dset.FashionMNIST(root=data_dir, train=False, transform=trans, download=True)
    elif config.data_name == 'cifar10':
        train_dataset = dset.CIFAR10(root=data_dir, train=True, transform=trans_cifar10, download=True)
        test_dataset = dset.CIFAR10(root=data_dir, train=False, transform=trans_cifar10, download=True)
    

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=num_work, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             num_workers=num_work, shuffle=False)

    data_loader = {'train': train_loader, 
                   'test': test_loader}

    return data_loader

def get_kitti_dataloader(config):
    train_loader = DepthDataLoader(config, "train").data
    test_loader = DepthDataLoader(config, "online_eval").data
    
    data_loader = {'train': train_loader, 
                   'test': test_loader}
    return data_loader
