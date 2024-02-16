import torch

from config import get_config
from data import get_dataloader
from op import Operator
from util import Checkpoint


def main(config):
    # Load data_loader
    data_loader = get_dataloader(config)
    print('==>>> Total training batch number: {}'.format(len(data_loader['train'])))
    print('==>>> Total testing batch number: {}'.format(len(data_loader['test'])))
    # Define checkpoint
    check_point = Checkpoint(config)
    # Define operator
    operator = Operator(config, check_point)
    if config.is_train:
        operator.train(data_loader)
    else:
        operator.test(data_loader)

def get_available_gpu():
    for i in range(torch.cuda.device_count()):
        if torch.cuda.is_available(i):
            return i
    raise Exception('No GPU available!')

if __name__ == "__main__":
    # Get configuration
    config = get_config()
    # Device setting
    config.gpu = get_available_gpu()
    config.device = torch.device('cuda:{}'.format(config.gpu))
    print("==>>> Device: ", config.device)
    # Main
    main(config)
