import os
from PIL import Image
import imageio

# input parameters:
#   input_path: folder path which contains folders of all frames for each git
#   gif_name:   specific gif name that user want to convert
def make_gif(input_path:str, gif_name: str):
    # Handle the exception of not given inputs
    if(input_path == "" or gif_name == ""):
        print("\nError: Given inputs are empty!\n")
        return 
    
    input_path = f'{input_path}{gif_name}/'
    save_path = f'./GIF_output/rmn_final_results/{gif_name}/'
    
    # check if the input folder exist
    if not os.path.exists(f'{input_path}'):
        print("Error: input folder doesn't exist!\n")
        return
    
    # check if the folder for saving outputs exist
    # if not, then make a new one
    if not os.path.exists(f'{save_path}'):
        os.makedirs(f'{save_path}')
    
    i = 0
    frames = []
    
    # Get each frame images from the input_path and store them in the list: frames
    for frame_name in os.listdir(input_path):
        frame_name = f'{gif_name}_{i}_rmnResult.jpg'
        frame = imageio.v2.imread(os.path.join(input_path, f'{frame_name}'))
        if frame is not None:           
            frames.append(frame)
        i = i+1
    
    # combine all images in 'frames' list and convert it to gif
    imageio.mimsave(f'{save_path}{gif_name}_result.gif', frames) 
    print(f"{gif_name}.gif has been successfuly made")