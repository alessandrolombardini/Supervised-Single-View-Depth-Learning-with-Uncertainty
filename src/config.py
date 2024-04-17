import argparse
import os
from distutils.util import strtobool

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--is_train", type=strtobool, default='true')
parser.add_argument("--tensorboard", type=strtobool, default='true')
parser.add_argument("--num_gpu", type=int, default=0)
parser.add_argument("--num_work", type=int, default=8)
parser.add_argument("--exp_dir", type=str, default="../results")
parser.add_argument("--exp_load", type=str, default=None)

# Data
parser.add_argument("--data_dir", type=str, default="/mnt/sda")
parser.add_argument("--data_name", type=str, default="kitti", choices=('kitti'))
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--rgb_range', type=int, default=3)         # !

# Model
parser.add_argument('--uncertainty', default='normal', choices=('normal', 'aleatoric.gaussian', 'aleatoric.tstudent', 
                                                                'aleatoric.laplacian'))
parser.add_argument('--n_feats', type=int, default=32)
parser.add_argument('--var_weight', type=float, default=1.)
parser.add_argument('--drop_rate', type=float, default=0.2)
parser.add_argument('--in_channels', type=int, default=3)       # !

# Train
parser.add_argument("--epochs", type=int, default=200) 
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--decay", type=str, default='50-100-150')
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--optimizer", type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--epsilon", type=float, default=1e-8)

# Test
parser.add_argument('--n_samples', type=int, default=25)


parser.add_argument("--dataset", type=str, default="kitti")
parser.add_argument("--distributed", type=bool, default=False)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--clip_grad", type=float, default=0.1)
parser.add_argument("--use_shared_dict", type=bool, default=False)
parser.add_argument("--shared_dict", type=str, default=None)
parser.add_argument("--use_amp", type=bool, default=False)
parser.add_argument("--aug", type=bool, default=True)
parser.add_argument("--random_crop", type=bool, default=False)
parser.add_argument("--random_translate", type=bool, default=False)
parser.add_argument("--translate_prob", type=float, default=0.2)
parser.add_argument("--max_translation", type=int, default=100)
parser.add_argument("--validate_every", type=float, default=0.25)
parser.add_argument("--log_images_every", type=float, default=0.1)
parser.add_argument("--prefetch", type=bool, default=False)
parser.add_argument("--save_dir", type=str, default="./depth_anything_finetune")
parser.add_argument("--project", type=str, default="ZoeDepth")
parser.add_argument("--tags", type=str, default="")
parser.add_argument("--notes", type=str, default="")
parser.add_argument("--gpu", type=str, default=None)
parser.add_argument("--root", type=str, default=".")
parser.add_argument("--uid", type=str, default=None)
parser.add_argument("--print_losses", type=bool, default=False)
parser.add_argument("--min_depth", type=float, default=0.001)
parser.add_argument("--max_depth", type=float, default=80)
parser.add_argument("--data_path", type=str, default=os.path.join("../dataset/kitti/raw_data"))
parser.add_argument("--gt_path", type=str, default=os.path.join("../dataset/kitti/data_depth_annotated_zoedepth"))
parser.add_argument("--filenames_file", type=str, default="../dataset/kitti/kitti_eigen_train_files_with_gt.txt")
parser.add_argument("--input_height", type=int, default=352)
parser.add_argument("--input_width", type=int, default= 1216)
parser.add_argument("--data_path_eval", type=str, default=os.path.join("../dataset/kitti/raw_data"))
parser.add_argument("--gt_path_eval", type=str, default=os.path.join("../dataset/kitti/data_depth_annotated_zoedepth"))
parser.add_argument("--filenames_file_eval", type=str, default="../dataset/kitti/kitti_eigen_test_files_with_gt.txt")
parser.add_argument("--min_depth_eval", type=float, default=1e-3)
parser.add_argument("--max_depth_eval", type=float, default=80)
parser.add_argument("--do_random_rotate", type=bool, default=True)
parser.add_argument("--degree", type=float, default=1.0)
parser.add_argument("--do_kb_crop", type=bool, default=True)
parser.add_argument("--garg_crop", type=bool, default=True)
parser.add_argument("--eigen_crop", type=bool, default=False)
parser.add_argument("--use_right", type=bool, default=False)


def save_args(obj, defaults, kwargs):
    for k,v in defaults.iteritems():
        if k in kwargs: v = kwargs[k]
        setattr(obj, k, v)

def get_config():
    config = parser.parse_args()
    config.data_dir = os.path.expanduser(config.data_dir)
    return config
