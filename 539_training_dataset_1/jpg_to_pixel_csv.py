import csv
import numpy as np
import os
import random

from PIL import Image

# Define a function to convert an image to pixel values


def image_to_pixels(filepath):
    with Image.open(filepath) as im:
        # pixel_values = np.array(im)
        pixels = list(im.getdata())
        pixels = [val for pixel in pixels for val in pixel]
        pixels_str = ' '.join(str(p) for p in pixels)
        return pixels_str


emotion = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMO_DICT = {"neutral": 0,
            "anger": 1,
            "disgust": 2,
            "fear": 3,
            "happy": 4,
            "sad": 5,
            "surprise": 6,
            }

# Define a function to recursively traverse your Google Drive and get pixel values for image files


def get_image_pixels():
    data = []
    for e in emotion:
        root_dir = './' + e
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                    random_number = random.randint(0, 99)
                    if 0 <= random_number and random_number < 60:
                        usage = 'train'
                    elif 60 <= random_number and random_number < 80:
                        usage = 'val'
                    else:
                        usage = 'test'

                    filepath = os.path.join(dirpath, filename)
                    pixels = image_to_pixels(filepath)
                    data.append({'filename': filename, 'emotion': EMO_DICT[e],
                                'pixels': pixels, 'Usage': usage})
    return data


# Write the pixel data to a CSV file
data = get_image_pixels()
with open('../image_pixels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'emotion', 'pixels', 'Usage'])
    for row in data:
        writer.writerow([row['filename'], row['emotion'], row['pixels'], row['Usage']])
