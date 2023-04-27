from rmn import RMN
import csv
m = RMN()

import os
from collections import Counter

#import save_gif_to_imgs as sgti
#sgti.save_gif_to_imgs('./GIF_input/', 'faces.csv', './GIF_output/gif_to_jpgs_results/')

gif_list = os.listdir('./GIF_output/gif_to_jpgs_results')

gif_data_list =[]
for a in gif_list:
    results = list(m.detect_emotions_and_store_in_list('./GIF_output/gif_to_jpgs_results/', a))
    results_ = [x[0] for x in results]
    
    for item in results_:
        item['gif_name'] = a
    gif_data_list += results_ 

with open('gif_data_result.csv', mode='w', newline='') as file:
    fieldnames = ['ymax', 'xmax', 'xmin', 'ymin','emo_label','emo_proba','proba_list','gif_name']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(gif_data_list)

from collections import Counter
import pandas as pd
df = pd.read_csv('gif_data_result.csv')
gif_list = list(Counter(list(df['gif_name'])))
fieldnames = ['emo_label','times','gif_name']
result_gif = [fieldnames]
for i in range(len(gif_list)):
    df_1 = df[df['gif_name']==gif_list[i]]
    result_1 = sorted(Counter(list(df_1['emo_label'])).items(), key = lambda  x:x[1], reverse=True)[0][0]
    result_2 = sorted(Counter(list(df_1['emo_label'])).items(), key = lambda  x:x[1], reverse=True)[0][1]
    result_gif.append([result_1,result_2,gif_list[i]])

with open('gif_frequency_result.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    for row in result_gif:
        writer.writerow(row)