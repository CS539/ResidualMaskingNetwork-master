import json
import os

import cv2
import torch
from torchvision.transforms import transforms

from models import resmasking_dropout1
from pathlib import Path

# from ..ssd_infer import ensure_color
from utils.utils import ensure_gray
from utils.utils import ensure_color

haar = r'C:\Users\Jihyun\AppData\Local\Programs\Python\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml'
# haar = Path(r"C:\Users\Jihyun\AppData\Local\Programs\Python\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")
# haar = "/usr/local/lib/python3.10/dist-packages/cv2/data/haarcascade_frontalface_alt.xml"

face_cascade = cv2.CascadeClassifier(haar)

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

# @torch.no_grad()
# def detect_emotions_and_store_in_list(self, input_path, gif_name):
#     import cv2
#     from collections import deque
#     from PIL import Image
#     import os

#     i = 0
#     output_lists = deque()
#     # input_path = f'{input_path}{gif_name}'
#     # output_path = f'./GIF_output/rmn_first_results/{gif_name}'
#     input_path = f'{input_path}{gif_name}/'
#     output_path = f'./GIF_output/rmn_first_results/{gif_name}'

#     # Check if the folder to store this function's results exists
#     # If not, make new one
#     if not os.path.exists(f'{output_path}'):
#         os.makedirs(f'{output_path}')

#     # loop for converting each image using the name of the image file
#     # And then store the result of detecting emotion for each frame to the deqeue list
#     # Also, draw results on that image (frame) and save that image on the result folder

#     for frame_name in os.listdir(input_path):

#             frame_name = f'{gif_name}_{i}.jpg'
#             frame = cv2.imread(os.path.join(input_path, f'{frame_name}'))

#             if frame is not None:  # there is a image
#                 frame_detection = self.detect_emotion_for_single_frame(frame)
#                 if len(frame_detection) == 0:
#                     frame_detection.append(
#                         {
#                             "xmin": 0,
#                             "ymin": 0,
#                             "xmax": 0,
#                             "ymax": 0,
#                             "emo_label": 0,
#                             "emo_proba": 0,
#                             "proba_list": 0,
#                         }
#                     )

#                 else:
#                     output_lists.append(frame_detection)
#                     frame = self.draw(frame, frame_detection)

#                 # store the result frame in new folder
#                 result_frame = Image.open(f'{input_path}{gif_name}_{i}.jpg').copy()
#                 result_frame.save(f'{output_path}/{gif_name}_{i}_rmnResult.jpg')
#                 cv2.imwrite(f'{output_path}/{gif_name}_{i}_rmnResult.jpg', frame)
#                 i = i + 1
#         return output_lists

def main(image_path):
    # load configs and set random seed
    configs = json.load(open("./configs/fer2013_config.json"))
    image_size = (configs["image_size"], configs["image_size"])

    # model = densenet121(in_channels=3, num_classes=7)
    model = resmasking_dropout1(in_channels=3, num_classes=7)
    model.cuda()

    # state = torch.load("./saved/checkpoints/resnet34_test_2023May01_14.34")
    state = torch.load("./checkpoint/resnet34_test_2023May01_14.34")
    model.load_state_dict(state["net"])
    model.eval()

    image = cv2.imread(image_path)    

    faces = face_cascade.detectMultiScale(image, 1.15, 5)
    gray = ensure_gray(image)
    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (179, 255, 179), 2)

        face = gray[y : y + h, x : x + w]
        face = ensure_color(face)

        face = cv2.resize(face, image_size)
        face = transform(face).cuda()
        face = torch.unsqueeze(face, dim=0)

        output = torch.squeeze(model(face), 0)
        proba = torch.softmax(output, 0)

        emo_proba, emo_idx = torch.max(proba, dim=0)
        emo_idx = emo_idx.item()
        emo_proba = emo_proba.item()

        emo_label = FER_2013_EMO_DICT[emo_idx]

        label_size, base_line = cv2.getTextSize(
            "{}: 000".format(emo_label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        cv2.rectangle(
            image,
            (x + w, y + 1 - label_size[1]),
            (x + w + label_size[0], y + 1 + base_line),
            (223, 128, 255),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            "{}: {}".format(emo_label, int(emo_proba * 100)),
            (x + w, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )


if __name__ == "__main__":
    import sys

    # argv = sys.argv[1]
    argv = "./GIF_output/gif_to_jpgs_results/the-office_9/the-office_9_32.jpg"
    assert isinstance(argv, str) and os.path.exists(argv)
    main(argv)
