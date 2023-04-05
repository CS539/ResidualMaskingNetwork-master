from PIL import Image

# (255, 255, 255) = white
def gif_to_jpg(file_path: str, save_path: str, trans_color: tuple=(255, 255, 255)):

    # Check filename extension
    if file_path[-3:] != 'gif':
        print(f'it is not gif file, {file_path[-3:]}')
        return

    # Delete filename extension '.gif'
    file_name = file_path[:-4] 
    with Image.open(file_path) as im:

        # check frames
        print(f'frame count: {im.n_frames}')
        for i in range(im.n_frames):

            # move to image[i]
            im.seek(i)       
   
            # convert to RGBA
            image = im.convert("RGBA")
            new_data = []            

            for item in image.getdata():
                if item[3] == 0:
                    new_data.append(trans_color)
                else:
                    new_data.append(tuple(item[:3]))

            # Create a new RGB blank image
            new_image = Image.new("RGB", im.size)   
         
            # Add new_data into new_image
            new_image.putdata(new_data)

            # save
            new_image.save(f'{save_path}{file_name}_{i}.jpg')
