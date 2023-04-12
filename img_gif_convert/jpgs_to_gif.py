import os
from PIL import Image

# (255, 255, 255) = white

import imageio

def jpgs_to_gif(input_path: str, gif_name: str, save_path: str):
    images = []
    for filename in os.listdir(input_path):
        images.append(imageio.imread(filename))
    
    if not os.path.exists(f'{save_path}{gif_name}'):
        os.makedirs(f'{save_path}{gif_name}')

    imageio.mimsave(f'{save_path}{gif_name}/{gif_name}.gif', images)
    print(f'successfully convert the file: {gif_name}.gif')


# def gif_to_jpgs(file_path: str, gif_name: str, save_path: str, trans_color: tuple = (255, 255, 255)):

#     file_path_with_name = file_path + gif_name
#     # Check filename extension
#     if not(file_path_with_name[-3:] == 'gif' or file_path_with_name[-3:] == 'GIF'):
#         print(f'it is not gif file, {file_path_with_name[-3:]}')
#         return

#     # Delete filename extension '.gif'
#     # file_name contains with file path
#     # exact_file_name (which is below the code) will not store the file path
#     file_name = file_path_with_name[:-4]
#     with Image.open(file_path_with_name) as im:

#         # check frames
#         print(f'frame count: {im.n_frames}')
#         for i in range(im.n_frames):

#             # move to image[i]
#             im.seek(i)

#             # convert to RGBA
#             image = im.convert("RGBA")
#             new_data = []

#             for item in image.getdata():
#                 if item[3] == 0:
#                     new_data.append(trans_color)
#                 else:
#                     new_data.append(tuple(item[:3]))

#             # Create a new RGB blank image
#             new_image = Image.new("RGB", im.size)

#             # Add new_data into new_image
#             new_image.putdata(new_data)

#             # This is only the gif name
#             exact_file_name = file_name.split('/')[-1]

#             # check if the directory exists
#             # If not, create a new one
#             if not os.path.exists(f'{save_path}{exact_file_name}'):
#                 os.makedirs(f'{save_path}{exact_file_name}')

#             # save
#             new_image.save(f'{save_path}{exact_file_name}/{exact_file_name}_{i}.jpg')
            
#         print(f'successfully convert the file: {exact_file_name}')
