import os

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils.augmenters.augment import seg

EMOTION_DICT = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "sad",
    6: "surprise",
    7: "contempt"
}


class FER2013(Dataset):
    def __init__(self, stage, configs, dataset, tta=False, tta_size=48):
    # def __init__(self, configs, tta=False, tta_size=48):
        self._stage = stage
        self._configs = configs
        self._tta = tta
        self._tta_size = tta_size

        self._image_size = (configs["image_size"], configs["image_size"])

        # print(f'print print print {os.path.join(configs["data_path"], "{}.csv".format(stage))}')
        # self._data = pd.read_csv(
        #     # os.path.join(configs["data_path"], "{}.csv".format(stage))
        #     configs['csv_path']
        # )
        
        self._data = dataset

        self._pixels = self._data["pixels"].tolist()
        self._emotions = pd.get_dummies(self._data["emotion"])

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def is_tta(self):
        return self._tta == True

    def __len__(self):
        return len(self._pixels)

    def __getitem__(self, idx):
        pixels = self._pixels[idx]
        pixels = list(map(int, pixels.split(" ")))
        image = np.asarray(pixels).reshape(48, 48)
        image = image.astype(np.uint8)

        image = cv2.resize(image, self._image_size)
        image = np.dstack([image] * 3)

        # if self._stage == "train":
        if self._stage == 'test':
            image = seg(image=image)

        # if self._stage == "test" and self._tta == True:
        if self._stage == "train" and self._tta == True:
            images = [seg(image=image) for i in range(self._tta_size)]
            images = [image for i in range(self._tta_size)]
            images = list(map(self._transform, images))
            target = self._emotions.iloc[idx].idxmax()
            return images, target

        image = self._transform(image)
        target = self._emotions.iloc[idx].idxmax()
        return image, target


def fer2013(stage, configs=None, tta=False, tta_size=48):
    return FER2013(stage, configs, tta, tta_size)


if __name__ == "__main__":
    data = FER2013(
        "train",
        {
            "data_path": "../../Kaggle_raw_dataset/",
            "image_size": 96,
            "in_channels": 3,
        },
    )
    import cv2

    # targets = []

    # for i in range(len(data)):
    #     image, target = data[i]
    #     cv2.imwrite("debug/{}.png".format(i), image)
    #     if i == 200:
    #         break
