import torch
import os
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils.radam import RAdam
from trainers._fer2013_trainer import FER2013Trainer
from utils.datasets.fer2013dataset import FER2013
from models.fer2013_models import BaseNet
from sklearn.model_selection import train_test_split

configs = {
    "lr": 0.05,
    "batch_size": 32,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "distributed": False,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "max_epoch_num": 20,
    "max_plateau_count": 10,
    "plateau_patience": 3,
    "log_dir": "logs",
    "model_name": "my_model",
    "csv_path": os.path.join(os.getcwd(), 'Kaggle_raw_dataset', 'labels.csv'),
    "in_channels": 3,
    "num_classes": 8,
    "weighted_loss": 0,
    'image_size': 32,
}

data = pd.read_csv(
    # os.path.join(configs["cwd"], 'Kaggle_raw_dataset', "labels.csv")
    configs['csv_path']
)
# Split data into train, validation, and test sets
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)

train = FER2013('train', configs=configs, dataset=train_set, tta=True)
val = FER2013('test', configs=configs, dataset=val_set)
test = FER2013('test', configs=configs, dataset=test_set)

# class FER2013(Dataset):
#     def __init__(self, stage, configs, tta=False, tta_size=48):

model = BaseNet(in_channels=configs['in_channels'], num_classes=configs['num_classes'])


trainer = FER2013Trainer(model, train, val, test, configs)

trainer.run()
