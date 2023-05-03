## Step 1: Install the RMN model
# from rmn import RMN

# m = RMN()

from legacy import demo_one_image

m = demo_one_image()

# m.video_demo()

## Step 2: convert our GIF inputs to images
import save_gif_to_imgs as sgti

## save_gif_to_imgs(csv_file_path_without_name: str, csv_file_name: str, save_filepaths: str):
# sgti.save_gif_to_imgs('./GIF_input/test/', 'faces.csv', './GIF_output/gif_to_jpgs_results/')

## Step 3: call detect_emotions_and_store_n_list function
#  detect_emotions_and_store_n_list(save_filepaths which has been used above : str, gif_file_name: str):
results = m.detect_emotions_and_store_in_list('./GIF_output/gif_to_jpgs_results/', 'the-office_9')


## Step 4: convert those images into a gif
import img_gif_convert.jpgs_to_gif as jtg

#  make_gif(input_path:str, gif_name: str):
jtg.make_gif('./GIF_output/rmn_first_results/', 'the-office_9')
