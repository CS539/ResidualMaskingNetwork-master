import gif_img_convert.gif_to_jpgs as gj
import csv

# make sure just put csv_file location without csv file name


def save_gif_to_imgs(csv_file_path_without_name: str, csv_file_name: str, save_filepaths: str):
    # open the csv file which saves gif file names
    with open(csv_file_path_without_name + csv_file_name) as csv_file:
        rows = csv.reader(csv_file)
        for row in rows:
            for gif_name in row:
                try:
                    gj.gif_to_jpgs(csv_file_path_without_name, gif_name, save_filepaths)
                except:
                    print(f'The file {csv_file_path_without_name}{gif_name} does not exist!')