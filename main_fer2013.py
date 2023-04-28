from models import segmentation
import models
import pandas as pd
import torch.multiprocessing as mp
import torch
import numpy as np
import imgaug
import json
import os
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(config_path):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()
    # configs['cwd'] = '~usr/local/lib/python3.10/dist-packages/torch/cuda/'

    # load model and data_loader
    model = get_model(configs)

    train_set, val_set, test_set = get_dataset(configs)

    # init trainer and make a training
    # from trainers.fer2013_trainer import FER2013Trainer
    from trainers.tta_trainer import FER2013Trainer

    # from trainers.centerloss_trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    if configs["distributed"] == 1:
        ngpus = torch.cuda.device_count()
        mp.spawn(trainer.train, nprocs=ngpus, args=())
    else:
        trainer.train()

    torch.save(trainer.get_model.state_dict(), './our_model')


def get_model(configs):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """
    try:
        return models.__dict__[configs["arch"]]
    except KeyError:
        return segmentation.__dict__[configs["arch"]]


def get_dataset(configs):
    """
    This function get raw dataset
    """
    from utils.datasets.fer2013dataset import fer2013

    data = pd.read_csv('./train.csv')

    train = pd.read_csv('./image_pixels_train.csv')
    val = pd.read_csv('./image_pixels_val.csv')
    test = pd.read_csv('./image_pixels_test.csv')

    # todo: add transform
    train_set = fer2013("train", configs, train)
    val_set = fer2013("val", configs, val)
    test_set = fer2013("test", configs, test, tta=True, tta_size=10)
    return train_set, val_set, test_set

# def fer2013(stage, configs=None, dataset=None, tta=False, tta_size=48):
#     return FER2013(stage, configs, dataset, tta, tta_size)


if __name__ == "__main__":
    main("./configs/fer2013_config.json")
