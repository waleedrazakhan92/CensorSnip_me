############################
## Misc utils
############################
import io
import base64
from IPython.display import HTML, display

import os
import json

def save_preds_json(out_json_path,all_frame_preds):
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_frame_preds, f, ensure_ascii=False, indent=4)

def read_preds_json(preds_json_path):
    with open(preds_json_path, 'r') as f:
        data_read = json.load(f)

    # Convert keys to integers
    data_read = {int(k): v for k, v in data_read.items()}
    return data_read

def display_video(filename,vid_res=(256,256)):

    video_data = io.open(filename, 'rb').read()
    video_b64 = base64.b64encode(video_data).decode('utf-8')
    html = f"""
    <video controls>
    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>
    """
    html = f"""
    <video controls height={vid_res[0]} width={vid_res[1]}>
    <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>
    """
    display(HTML(html))


def make_folder(in_path):
    if not os.path.isdir(in_path):
        os.mkdir(in_path)

def make_folders_multi(*in_list):
    for in_path in in_list:
        make_folder(in_path)

def make_multiple_zips(in_folder,path_zips,sub_folder=None):
    make_folder(path_zips)
    all_folders = glob(os.path.join(in_folder,'*'))

    for i in tqdm(range(0,len(all_folders))):
        one_folder = all_folders[i]
        if sub_folder!=None:
            one_folder = os.path.join(all_folders[i],sub_folder)

        one_name = os.path.join(path_zips,all_folders[i].split('/')[-1]+'.zip')

        os.command(f'zip -q -r  {one_name} {all_folders[i]}')

def print_num_images(paths_dict_all):
    for j in paths_dict_all:
        for k in paths_dict_all[j]:
            print(len(os.listdir(paths_dict_all[j][k])))
            print(paths_dict_all[j][k])
