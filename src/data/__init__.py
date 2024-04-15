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
    HOME_DIR = os.path.expanduser("~")
    DATASETS_CONFIG = {
        "save_dir": os.path.expanduser("./depth_anything_finetune"),
        "project": "ZoeDepth",
        "tags": '',
        "notes": "",
        "gpu": None,
        "root": ".",
        "uid": None,
        "print_losses": False,
        "dataset": "kitti",
        "min_depth": 0.001,
        "max_depth": 80,
        "data_path": os.path.join(HOME_DIR, "../dataset/kitti/raw_data"),
        "gt_path": os.path.join(HOME_DIR, "../dataset/kitti/data_depth_annotated_zoedepth"),
        "filenames_file": "../dataset/kitti/kitti_eigen_train_files_with_gt.txt",
        "input_height": 352,
        "input_width": 1216,  # 704
        "data_path_eval": os.path.join(HOME_DIR, "../dataset/kitti/raw_data"),
        "gt_path_eval": os.path.join(HOME_DIR, "../dataset/kitti/data_depth_annotated_zoedepth"),
        "filenames_file_eval": "../dataset/kitti/kitti_eigen_test_files_with_gt.txt",
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "do_random_rotate": True,
        "degree": 1.0,
        "do_kb_crop": True,
        "garg_crop": True,
        "eigen_crop": False,
        "use_right": False
    }
    config_dict = vars(config)
    config_dict.update(DATASETS_CONFIG)

    from argparse import Namespace

    config = Namespace(**config_dict)
    
    train_loader = DepthDataLoader(DATASETS_CONFIG, "train").data
    test_loader = DepthDataLoader(DATASETS_CONFIG, "online_eval").data
    
    data_loader = {'train': train_loader, 
                   'test': test_loader}
    return data_loader
