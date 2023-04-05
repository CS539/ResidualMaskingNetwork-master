# Reading an animated GIF file using Python Image Processing Library - Pillow
from PIL import Image
from PIL import GifImagePlugin
import csv

gif_list = []

# this will add gif directories into gif_list list
with open('../GIF_input/faces.csv', 'r') as f:
    rows = csv.reader(f)
    for row in rows:
        for i in row:
            print(i + '\n')
            gif_list.extend('../GIF_input/' + i)

print(gif_list[1])

imageObject = Image.open(gif_list[0])
print(imageObject.is_animated)
print(imageObject.n_frames)

# Display individual frames from the loaded animated GIF file
for frame in range(0,imageObject.n_frames):
    imageObject.seek(frame)
    imageObject.show()

