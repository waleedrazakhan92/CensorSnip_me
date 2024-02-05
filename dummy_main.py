from utils.display_utils import *
from utils.misc_utils import *
from utils.prediction_utils import *
from utils.video_utils import *
############################
## Inference utils
############################
import os
import cv2
import shutil

from ultralytics import YOLO
import argparse
import json

import sys
import imageio
from glob import glob
import subprocess
import ffmpegcv

def delete_unused_paths(paths_dict,keep_paths_dict):
    for k in keep_paths_dict.keys():
        if keep_paths_dict[k]==False:
            shutil.rmtree(paths_dict[k])

def custom_yolov8_inference_video(porn_model,input_path,path_write,is_video,box_color_dict=None,save_txt=True,save_original=True,save_bbox=True,save_blur=True,display_bbox=False,
                      adjust_fraction=1,num_imgs=4,figsize=(3,3),label_dict=None,img_quality=100,
                                  class_confidence_dict=None,gpu_writer=False):

    assert type(label_dict)==dict or label_dict==None

    paths_dict = {
        'write_main':path_write,
        'images':os.path.join(path_write,'images/'),
        'images_empty':os.path.join(path_write,'images_empty/'),
        'bboxes':os.path.join(path_write,'bboxes/'),
        'bboxes_empty':os.path.join(path_write,'bboxes_empty/'),
        'blur':os.path.join(path_write,'blurred/'),
        'blur_empty':os.path.join(path_write,'blurred_empty/'),
        'txt':os.path.join(path_write,'txt_files/'),
        'txt_empty':os.path.join(path_write,'txt_files_empty/'),
        'videos':os.path.join(path_write,'videos/')
    }
    keep_paths_dict = {
        'images':save_original,
        'images_empty':save_original,
        'bboxes':save_bbox,
        'bboxes_empty':save_bbox,
        'blur':save_blur,
        'blur_empty':save_blur,
        'txt':save_txt,
        'txt_empty':save_txt,
        'videos':is_video
    }

    make_folders_multi(*paths_dict.values())
    delete_unused_paths(paths_dict,keep_paths_dict)

    if is_video==False:
        # sorted_names = get_sorted_frames(input_path)
        # all_images = append_complete_path(input_path,sorted_names)
        all_images = glob(os.path.join(input_path,'*'))
        print('Total Images:',len(all_images))

        all_images = all_images[:num_imgs] if num_imgs!=None else all_images
        print('Selected Images:',len(all_images))
        total_images = len(all_images)

    elif is_video==True:
        if gpu_writer==True:
            ###################################
            ## gpu
            ###################################
            cap = ffmpegcv.VideoCaptureNV(input_path)
            fps = cap.fps
            frame_width = cap.width
            frame_height = cap.height
            total_images = cap.count
            print(cap)
        else:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(5)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            total_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_vid_name = None # dummy name in case input is a directory
    passed_images = []
    failed_images = []
    all_predictions = {}  ## for debug
    for i in tqdm(range(0,total_images),mininterval=10):
        if is_video==False:
            img_path = all_images[i]
            image_name = img_path.split('/')[-1]
            image_name = os.path.splitext(image_name)[0]
            img = cv2.imread(img_path)
        else:
            img_path = os.path.join('/dummy_name/',str(i)+'.jpg')
            image_name = img_path.split('/')[-1]
            image_name = os.path.splitext(image_name)[0]
            ret, img = cap.read()
            if not ret:
                break
            img_org = img.copy()

            if i==0:
                out_vid_name,out_vid_ext = os.path.splitext(input_path.split('/')[-1])
                out_vid_name = os.path.join(paths_dict['videos'],out_vid_name+'_filtered'+out_vid_ext)

                # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # You can change the codec as needed
                # vid_writer = cv2.VideoWriter(out_vid_name, fourcc, fps, (frame_width, frame_height))

                if gpu_writer==True:
                    vid_writer = ffmpegcv.VideoWriterNV(out_vid_name, cap.codec, fps)
                else:
                    vid_writer = imageio.get_writer(out_vid_name, fps=fps, macro_block_size=1)


        img_h,img_w = img.shape[:2]
        img_blur = cv2.blur(img.copy(),(int(img_h/2),int(img_w/2)))
        final_blur = img.copy()

        pred_porn = porn_model.predict(img)[0]

        pass_fail_dict = {}
        pass_fail_dict = pass_fail_image(pred_porn,box_color_dict,label_dict,adjust_fraction,class_confidence_dict,img_h,img_w)

        ######################
        ## for debug
        one_img_pred = []
        if len(pred_porn)!=0:
            for o in range(0,len(pred_porn)):
                one_pred_img = extract_prediction_points(pred_porn,o,box_color_dict,label_dict,adjust_fraction,img_h,img_w)
                one_img_pred.append(one_pred_img)

        all_predictions[i] = one_img_pred
        ######################

        if pass_fail_dict!={}:
            failed_images.append(i)
        else:
            passed_images.append(i)

        if pass_fail_dict!={}:#len(pred_porn)!=0:
            path_write_images = paths_dict['images']
            path_write_bboxes = paths_dict['bboxes']
            path_write_blur = paths_dict['blur']
            txt_file_path = os.path.join(paths_dict['txt'],image_name+'.txt')
        else:
            path_write_images = paths_dict['images_empty']
            path_write_bboxes = paths_dict['bboxes_empty']
            path_write_blur = paths_dict['blur_empty']
            txt_file_path = os.path.join(paths_dict['txt_empty'],image_name+'.txt')

        for p in range(0,len(pred_porn)):
            one_pred = extract_prediction_points(pred_porn,p,box_color_dict,label_dict,adjust_fraction,img_h,img_w)
            img = cv2.rectangle(img, one_pred.start_point, one_pred.end_point,one_pred.b_color, 3)
            img = cv2.putText(img, one_pred.b_txt, one_pred.start_point, cv2.FONT_HERSHEY_SIMPLEX ,
                    1, one_pred.b_color, 2, cv2.LINE_AA)

            final_blur[one_pred.start_point[1]:one_pred.end_point[1],
                       one_pred.start_point[0]:one_pred.end_point[0]] = img_blur[one_pred.start_point[1]:one_pred.end_point[1],
                                                                                 one_pred.start_point[0]:one_pred.end_point[0]]

            if save_txt==True: ##pass_fail_dict=={} and save_txt==True :#one_pred.b_conf>=filter_prob:
                ## only write predictions which model is confident on
                one_txt_line = [one_pred.c_lab,one_pred.x_c_norm,one_pred.y_c_norm,one_pred.b_w_norm,one_pred.b_h_norm]
                one_txt_line = [str(x) for x in one_txt_line]
                one_txt_line = ' '.join(one_txt_line)

                write_preds_txt_file(txt_file_path,one_txt_line)


        if ((len(pred_porn)==0) or (os.path.isfile(txt_file_path)==False)) and (save_txt==True):
            write_preds_txt_file(txt_file_path,'')

        if save_bbox==True:
            cv2.imwrite(os.path.join(path_write_bboxes,img_path.split('/')[-1]),img,[cv2.IMWRITE_JPEG_QUALITY, img_quality])

        if save_original==True:
            if is_video==False:
                shutil.copy(img_path,path_write_images)
            else:
                cv2.imwrite(os.path.join(path_write_images,img_path.split('/')[-1]),img_org,[cv2.IMWRITE_JPEG_QUALITY, img_quality])

        if save_blur==True:
            cv2.imwrite(os.path.join(path_write_blur,img_path.split('/')[-1]),final_blur,[cv2.IMWRITE_JPEG_QUALITY, img_quality])

        if display_bbox==True:
            display_multi(img,figsize=figsize,bgr=True)

        if is_video==True:
            if pass_fail_dict!={}:
                if gpu_writer==True:
                        vid_writer.write(img_blur[:,:,::-1])
                else:
                    vid_writer.append_data(img_blur[:,:,::-1])
            else:
                if gpu_writer==True:
                    vid_writer.write(img_org[:,:,::-1])
                else:
                    vid_writer.append_data(img_org[:,:,::-1])

    if is_video==True:
        try:
            vid_writer.close()
        except:
            pass

    return paths_dict,failed_images,passed_images,all_predictions,out_vid_name



parser = argparse.ArgumentParser()
parser.add_argument("--path_model",type=str, help="path pretrained model checkpoint")
parser.add_argument("--path_input",type=str, help="path of input video or a folder of images")
parser.add_argument("--path_results",type=str, help="path to save all the results", default='model_results/')
parser.add_argument("--class_confidence_dict",default=[0.5,0.5,0.5,0.5,0.5], action='store',nargs='*',
                        dest='class_confidence_dict',type=float,help="dictionary of probabilities with desired tolerance level for predictions")
parser.add_argument("--num_imgs", help="number of images you want to process", default=None, type=int)
parser.add_argument("--adjust_fraction", help="fraction [0-1] you want to manually adjust bounding box", default=1, type=float)

parser.add_argument("--img_quality", help="manually adjust the image image quality of the saved images (JPEG)", default=100, type=int)
parser.add_argument("--save_FLAG", help="flag to save (save bbox,blur,original,text file)", action="store_true")
parser.add_argument("--save_bbox", help="flag to save bbox", action="store_true")
parser.add_argument("--save_txt", help="flag to save text file", action="store_true")
parser.add_argument("--save_blur", help="flag to save blur,", action="store_true")

## trimming flags
parser.add_argument("--do_trimming", help="do video trimming or not", action="store_true")
##parser.add_argument("--write_video_trim", help="write video after trimming", action="store_true")
parser.add_argument("--write_frames_trim", help="write frames of the trimmed video", action="store_true")
parser.add_argument("--start_time", help="start time(seconds) for video trimming", default=None)
parser.add_argument("--duration", help="end time(seconds) for video trimming", default=None)

## extended flags
parser.add_argument("--gpu_writer", action="store_true")


## extras
parser.add_argument("--skip_sound", help="write back video without sound", action="store_true")

args = parser.parse_args()

print('-------------------------')
print(args)

## defaults
box_color_dict = {0:(255,0,0),
                  1:(0,255,0),
                  2:(0,0,255),
                  3:(0,255,255),
                  4:(255,0,255)}

class_confidence_dict = {0:0.5, 1:0.5, 2:0.5, 3:0.5, 4:0.5}

###########################
## assert checks
###########################
if os.path.isfile(args.path_input)==False and os.path.isdir(args.path_input)==False:
    print("------------------------------------------------------")
    print("ERROR!!! please make sure the file/directory exists. Exiting program")
    sys.exit(0)

## check if video or a directory
is_video = True if os.path.isfile(args.path_input) else False

## working with videos
if is_video==True:
    if args.do_trimming==True:
        assert is_video==True, 'make sure *path_input* is a video file as trimming can only be done on a video'
        ##assert args.write_frames_trim==True, 'when do_trimming is set to True, make sure you set the write_frames_trim'
        assert (args.start_time!=None and args.duration!=None), 'set positive integer values for the *start_time* and *duration*'
        ##assert args.save_FLAG==True, 'set save_FLAG flag == True in order to utilize do_trimming'

    if args.num_imgs!=None:
        assert is_video==False, 'To use num_imgs make sure the input path is a directiory containing images'

## working with images
elif is_video==False:
    assert args.do_trimming==False, 'do_trimming can only be used with videos'
    assert args.skip_sound==False, 'skip_sound can only be used with videos'
    ##assert args.write_video_trim==False, 'write_video_trim can only be used with videos'
    assert args.write_frames_trim==False, 'write_frames_trim can only be used with videos'
    assert args.start_time==None, 'start_time can only be used with videos'
    assert args.duration==None, 'duration can only be used with videos'


assert len(args.class_confidence_dict)<=len(class_confidence_dict), 'length of class_confidence_dict must be less than 5, as there are only 5 classes in the model '
assert type(args.img_quality)==int, 'image quality must be a integer value between 1-100'

###########################
## load model
###########################
custom_yolo = YOLO(args.path_model)

###########################
## read variables
###########################
path_input = args.path_input
path_write_main = args.path_results
adjust_fraction = args.adjust_fraction
num_imgs = args.num_imgs

save_FLAG = args.save_FLAG
img_quality = args.img_quality


for i in range(0,len(args.class_confidence_dict)):
    class_confidence_dict[i] = args.class_confidence_dict[i]

## for trimming make sure you input video in the input_path
if args.do_trimming==True:
    ##write_video_trim = args.write_video_trim
    write_frames_trim = args.write_frames_trim
    start_time = args.start_time
    duration = args.duration

    trimmed_vid_path,path_frames = trim_video_and_extract_frames(path_input,path_write_main,start_time,duration,
                                                    write_video=True,write_frames=write_frames_trim,save_ext='.jpg')

    path_input = trimmed_vid_path


print('-------------------')
print('Performing inference...')
print('-------------------')

paths_dict_all,failed_images_all,passed_images_all,all_predictions,out_vid_name = custom_yolov8_inference_video(custom_yolo,path_input,path_write_main,
                                    is_video,adjust_fraction=adjust_fraction,
                                    num_imgs=num_imgs,figsize=(6,3),box_color_dict=box_color_dict,
                                    save_txt=args.save_txt,save_original=save_FLAG,save_bbox=args.save_bbox,save_blur=args.save_blur,display_bbox=False,
                                    label_dict=None,img_quality=img_quality,
                                    class_confidence_dict=class_confidence_dict,
                                    gpu_writer=args.gpu_writer)


if is_video==True and args.skip_sound==False:
    video_name,video_ext = os.path.splitext(path_input.split('/')[-1])
    final_vid_path = os.path.join(paths_dict_all['videos'],video_name+'_final'+video_ext)
    ##add_sound_back(path_input,out_vid_name,final_vid_path)
    ##os.remove(out_vid_name)

    ffmpeg_command = ['ffmpeg', '-y', '-i', out_vid_name, '-i', path_input, '-c', 'copy', '-map', '0:0', '-map', '1:1', final_vid_path]
    subprocess.run(ffmpeg_command)
