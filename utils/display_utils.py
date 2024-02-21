import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import random
import cv2
import os
from tqdm import tqdm
import shutil


def cv2_imshow_rgb(img,resize=None, figsize=(15,15)):
    if resize!=None:
        img=cv2.resize(img,resize)

        #     cv2_imshow(cv2.cvtColor(np.uint8(img),cv2.COLOR_RGB2BGR))

    plt.figure(figsize=figsize)
    plt.imshow(np.uint8(img))


def display_multi(*images,resize=None, figsize=(15,15),bgr=False,axis=1):
    if resize!=None:
        res = np.array(cv2.resize(images[0],resize))
    else:
        res = np.array(images[0])

    for i in range(1,len(images)):
        if resize!=None:
            res_img = np.array(cv2.resize(images[i],resize))
        else:
            res_img = np.array(images[i])

        res = np.concatenate((res, res_img), axis=axis)

    if bgr==True:
        res = cv2.cvtColor(res,cv2.COLOR_BGR2RGB)

    return cv2_imshow_rgb(res,resize=None, figsize=figsize)

def display_random_images(path_images,num_imgs=4,resize=None,figsize=(3,3),bgr=False):
    all_images = glob(os.path.join(path_images, '*'))
    img_path = random.sample(all_images,num_imgs)
    all_read_imgs = []
    for i in range(0,num_imgs):
        img = cv2.imread(img_path[i])
        all_read_imgs.append(img)
        display_multi(img,resize=resize,figsize=figsize,bgr=bgr)


# def make_folder(in_path):
#     if not os.path.isdir(in_path):
#         os.mkdir(in_path)

# def make_folders_multi(*in_list):
#     for in_path in in_list:
#         make_folder(in_path)
