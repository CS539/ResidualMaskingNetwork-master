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
        num = len(pixels)
        pixels = [val for pixel in pixels for val in pixel]
        pixels_str = ' '.join(str(p) for p in pixels)
        return pixels_str, num
        # return pixels, num


emotion = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMO_DICT = {"neutral": 0,
            "anger": 1,
            "disgust": 2,
            "fear": 3,
            "happy": 4,
            "sad": 5,
            "surprise": 6
            }

# Define a function to recursively traverse your Google Drive and get pixel values for image files


def get_image_pixels():
    data = []
    train = []
    val = []
    test = []
    tr = 0
    va = 0
    te = 0
    for e in emotion:
        root_dir = './' + e
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
                    random_number = random.randint(0, 99)
                    if 0 <= random_number and random_number < 60:
                        usage = 'train'
                    elif 60 <= random_number and random_number < 80:
                        usage = 'val'
                    else:
                        usage = 'test'

                    filepath = os.path.join(dirpath, filename)
                    pixels, len = image_to_pixels(filepath)
                    data.append({'filename': filename, 'emotion': EMO_DICT[e],
                                'pixels': pixels, 'Usage': usage})
                    
                    if usage == 'train':
                        train.append({'filename': filename, 'emotion': EMO_DICT[e],
                                'pixels': pixels, 'Usage': usage})
                        tr += 1
                    elif usage == 'val':
                        val.append({'filename': filename, 'emotion': EMO_DICT[e],
                                'pixels': pixels, 'Usage': usage})
                        va += 1
                    else:
                        test.append({'filename': filename, 'emotion': EMO_DICT[e],
                                'pixels': pixels, 'Usage': usage})
                        te += 1
    print(f'train: {tr}, validation: {va}, test: {te}, pixel len: {len}')
    return (data, train, val, test)


# Write the pixel data to a CSV file
data, train, val, test = get_image_pixels()
with open('../image_pixels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'emotion', 'pixels', 'Usage'])
    for row in data:
        writer.writerow([row['filename'], row['emotion'], row['pixels'], row['Usage']])

with open('../image_pixels_train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'emotion', 'pixels', 'Usage'])
    for row in train:
        writer.writerow([row['filename'], row['emotion'], row['pixels'], row['Usage']])
with open('../image_pixels_val.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'emotion', 'pixels', 'Usage'])
    for row in val:
        writer.writerow([row['filename'], row['emotion'], row['pixels'], row['Usage']])        
with open('../image_pixels_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'emotion', 'pixels', 'Usage'])
    for row in test:
        writer.writerow([row['filename'], row['emotion'], row['pixels'], row['Usage']])
