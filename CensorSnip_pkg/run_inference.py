
import os
from .utils.display_utils import *
from .utils.misc_utils import *
from .utils.prediction_utils import *
from .utils.video_utils import *
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
import imageio.v3 as iio
from glob import glob
import subprocess
import ffmpegcv

def delete_unused_paths(paths_dict,keep_paths_dict):
    for k in keep_paths_dict.keys():
        if keep_paths_dict[k]==False:
            shutil.rmtree(paths_dict[k])

def custom_yolov8_inference_video(porn_model,input_path,path_write,is_video,box_color_dict=None,save_txt=True,save_original=True,save_bbox=True,save_blur=True,display_bbox=False,
                      adjust_fraction=1,num_imgs=4,figsize=(3,3),label_dict=None,img_quality=100,
                                  class_confidence_dict=None,video_reader='cv2',video_writer='imageio',pred_batch=1,
                                  write_encoding=None):

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
        all_images = glob(os.path.join(input_path,'*'))
        print('Total Images:',len(all_images))

        all_images = all_images[:num_imgs] if num_imgs!=None else all_images
        print('Selected Images:',len(all_images))
        total_images = len(all_images)

    elif is_video==True:
        ##vid_data = iio.immeta(input_path,exclude_applied=False)
        if video_reader=='gpu_ffmpeg':
            cap = ffmpegcv.VideoCaptureNV(input_path)
        elif video_reader=='cpu_ffmpeg':
            cap = ffmpegcv.VideoCapture(input_path)
        elif video_reader=='cv2':
            cap = cv2.VideoCapture(input_path)

        if video_reader in ['gpu_ffmpeg','cpu_ffmpeg']:
            fps = cap.fps
            frame_width = cap.width
            frame_height = cap.height
            total_images = cap.count
            write_encoding = cap.codec if write_encoding==None else write_encoding
        elif video_reader=='cv2':
            fps = cap.get(5)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            total_images = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if write_encoding==None:
                in_enc = int(cap.get(cv2.CAP_PROP_FOURCC))
                write_encoding = chr(in_enc&0xff) + chr((in_enc>>8)&0xff) + chr((in_enc>>16)&0xff) + chr((in_enc>>24)&0xff)


        print('----------------------------')
        print('Video Info:')
        print('FPS:',fps)
        print('Total Images:',total_images)
        print('Frame Width:',frame_width,'Frame Height:',frame_height)
        print('Video Codec:',write_encoding)
        print('----------------------------')

    out_vid_name = None # dummy name in case input is a directory
    passed_images = []
    failed_images = []
    all_predictions = {}  ## for debug

    print('----------------------------')
    print('Predicting...')
    print('----------------------------')
    for i in tqdm(range(0,total_images,pred_batch),mininterval=10):
        if is_video==False:
            img_path = all_images[i]
            image_name = img_path.split('/')[-1]
            image_name = os.path.splitext(image_name)[0]
            img = cv2.imread(img_path)
        else:
            if i==0:
                out_vid_name,out_vid_ext = os.path.splitext(input_path.split('/')[-1])
                out_vid_name = os.path.join(paths_dict['videos'],out_vid_name+'_filtered'+out_vid_ext)
                if video_writer=='gpu_ffmpeg':
                    vid_writer = ffmpegcv.VideoWriterNV(out_vid_name, write_encoding, fps)
                elif video_writer=='cpu_ffmpeg':
                    vid_writer = ffmpegcv.VideoWriter(out_vid_name, write_encoding, fps)
                elif video_writer=='cv2':
                    if write_encoding not in ['XVID','mp4v','mjpg']:
                        write_encoding = 'mp4v' # default encoding

                    fourcc = cv2.VideoWriter_fourcc(*write_encoding)
                    vid_writer = cv2.VideoWriter(out_vid_name, fourcc, fps, (frame_width, frame_height))
                elif video_writer=='imageio':
                    vid_data = iio.immeta(input_path,exclude_applied=False)
                    if write_encoding not in ['hevc','h264']:
                        write_encoding = vid_data['codec']

                    vid_writer = imageio.get_writer(out_vid_name, fps=fps,codec=write_encoding,quality=10,macro_block_size=1)

                print('\n----------------------------')
                print('Final codec: ',write_encoding)
                print('----------------------------')

            img_batch = []
            for j in range(0,pred_batch):

                img_path = os.path.join('/dummy_name/',str(i*pred_batch+j)+'.jpg')
                image_name = img_path.split('/')[-1]
                image_name = os.path.splitext(image_name)[0]
                ret, img = cap.read()
                if not ret:
                    break

                img_batch.append(img)

        pred_porn_all = porn_model.predict(img_batch)

        for k in range(0,len(img_batch)):
            img_idx = i*pred_batch+k

            img = img_batch[k]
            img_h,img_w = img.shape[:2]
            img_blur = cv2.blur(img.copy(),(int(img_h/2),int(img_w/2)))
            final_blur = img.copy()
            img_org = img.copy()

            pred_porn = pred_porn_all[k]
            pass_fail_dict = {}
            pass_fail_dict = pass_fail_image(pred_porn,box_color_dict,label_dict,adjust_fraction,class_confidence_dict,img_h,img_w)


            if is_video==True:
                video_img = img_blur if pass_fail_dict!={} else img_org
                if video_writer in ['gpu_ffmpeg','cpu_ffmpeg','cv2']:
                    vid_writer.write(video_img)
                elif video_writer=='imageio':
                    vid_writer.append_data(video_img[:,:,::-1])


            ######################
            ## for debug
            one_img_pred = []
            if len(pred_porn)!=0:
                for o in range(0,len(pred_porn)):
                    one_pred_img = extract_prediction_points(pred_porn,o,box_color_dict,label_dict,adjust_fraction,img_h,img_w)
                    one_img_pred.append(one_pred_img)

            all_predictions[img_idx] = one_img_pred
            ######################

            if pass_fail_dict!={}:
                failed_images.append(img_idx)
            else:
                passed_images.append(img_idx)

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

    print('----------------------------')
    print('Finished Predicting...')
    print('----------------------------')
    if is_video==True:
        try:
            vid_writer.close()
        except:
            pass

    # return paths_dict,failed_images,passed_images,all_predictions,out_vid_name
    return paths_dict,out_vid_name


def run(path_model=None,path_input=None,path_results='model_results/',class_confidence_dict=[0.5,0.5,0.5,0.5,0.5],
         num_imgs=None,adjust_fraction=1,img_quality=100,save_FLAG=False,save_bbox=False,save_txt=False,save_blur=False,
         do_trimming=False,write_frames_trim=False,start_time=None,duration=None,
         video_reader='gpu_ffmpeg',video_writer='gpu_ffmpeg',
         pred_batch=1,write_encoding=None,skip_sound=False):
    #############################################################
    ## Main
    #############################################################

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
    assert path_model!=None,'Please give appropriate model path'
    assert path_input!=None, 'Please give valid input path' 
    if os.path.isfile(path_input)==False and os.path.isdir(path_input)==False:
        print("------------------------------------------------------")
        print("ERROR!!! please make sure the file/directory exists. Exiting program")
        sys.exit(0)

    ## check if video or a directory
    is_video = True if os.path.isfile(path_input) else False

    ## working with videos
    if is_video==True:
        if do_trimming==True:
            assert is_video==True, 'make sure *path_input* is a video file as trimming can only be done on a video'
            ##assert args.write_frames_trim==True, 'when do_trimming is set to True, make sure you set the write_frames_trim'
            assert (start_time!=None and duration!=None), 'set positive integer values for the *start_time* and *duration*'
            ##assert args.save_FLAG==True, 'set save_FLAG flag == True in order to utilize do_trimming'

        if num_imgs!=None:
            assert is_video==False, 'To use num_imgs make sure the input path is a directiory containing images'

        assert (video_reader in ['gpu_ffmpeg','cpu_ffmpeg','cv2'])==True, 'please select valid video reader from [gpu_ffmpeg,cpu_ffmpeg,cv2]'
        assert (video_writer in ['gpu_ffmpeg','cpu_ffmpeg','cv2','imageio'])==True, 'please select valid video writer from [gpu_ffmpeg,cpu_ffmpeg,cv2,imageio]'

        if write_encoding!=None:
            if video_writer in ['gpu_ffmpeg','cpu_ffmpeg']:
                assert (write_encoding in ['hevc_nvenc','h264_nvenc'])==True, 'please select valid encoding from [hevc_nvenc,h264_nvenc]'
            elif video_writer=='cv2':
                assert (write_encoding in ['XVID','mp4v','mjpg'])==True, 'please select valid encoding from [XVID,mp4v,mjpg]'
            elif video_writer=='imageio':
                assert (write_encoding in ['hevc','h264'])==True, 'please select valid encoding from [hevc,h264]'


    ## working with images
    elif is_video==False:
        assert do_trimming==False, 'do_trimming can only be used with videos'
        assert skip_sound==False, 'skip_sound can only be used with videos'
        ##assert args.write_video_trim==False, 'write_video_trim can only be used with videos'
        assert write_frames_trim==False, 'write_frames_trim can only be used with videos'
        assert start_time==None, 'start_time can only be used with videos'
        assert duration==None, 'duration can only be used with videos'

    assert len(class_confidence_dict)<=len(class_confidence_dict), 'length of class_confidence_dict must be less than 5, as there are only 5 classes in the model '
    assert type(img_quality)==int, 'image quality must be a integer value between 1-100'

    ###########################
    ## load model
    ###########################
    custom_yolo = YOLO(path_model)

    ###########################
    ## read variables
    ###########################
    path_input = path_input
    path_write_main = path_results
    adjust_fraction = adjust_fraction
    num_imgs = num_imgs

    save_FLAG = save_FLAG
    img_quality = img_quality


    for i in range(0,len(class_confidence_dict)):
        class_confidence_dict[i] = class_confidence_dict[i]

    ## for trimming make sure you input video in the input_path
    if do_trimming==True:
        ##write_video_trim = args.write_video_trim
        write_frames_trim = write_frames_trim
        start_time = start_time
        duration = duration

        trimmed_vid_path,path_frames = trim_video_and_extract_frames(path_input,path_write_main,start_time,duration,
                                                        write_video=True,write_frames=write_frames_trim,save_ext='.jpg')

        path_input = trimmed_vid_path


    print('----------------------------')
    print('Performing inference...')
    print('----------------------------')

    # paths_dict_all,failed_images_all,passed_images_all,all_predictions,out_vid_name = custom_yolov8_inference_video(custom_yolo,path_input,path_write_main,
    #                                     is_video,adjust_fraction=adjust_fraction,
    #                                     num_imgs=num_imgs,figsize=(6,3),box_color_dict=box_color_dict,
    #                                     save_txt=args.save_txt,save_original=save_FLAG,save_bbox=args.save_bbox,save_blur=args.save_blur,display_bbox=False,
    #                                     label_dict=None,img_quality=img_quality,
    #                                     class_confidence_dict=class_confidence_dict,
    #                                     gpu_writer=args.gpu_writer,pred_batch=args.pred_batch)

    paths_dict_all,out_vid_name = custom_yolov8_inference_video(custom_yolo,path_input,path_write_main,
                                        is_video,adjust_fraction=adjust_fraction,
                                        num_imgs=num_imgs,figsize=(6,3),box_color_dict=box_color_dict,
                                        save_txt=save_txt,save_original=save_FLAG,save_bbox=save_bbox,save_blur=save_blur,display_bbox=False,
                                        label_dict=None,img_quality=img_quality,
                                        class_confidence_dict=class_confidence_dict,
                                        video_writer=video_writer,video_reader=video_reader,
                                        pred_batch=pred_batch,write_encoding=write_encoding)


    if is_video==True and skip_sound==False:
        print('----------------------------')
        print('Adding sound back...')
        print('----------------------------')
        video_name,video_ext = os.path.splitext(path_input.split('/')[-1])
        final_vid_path = os.path.join(paths_dict_all['videos'],video_name+'_final'+video_ext)
        ##add_sound_back(path_input,out_vid_name,final_vid_path)
        ##os.remove(out_vid_name)

        ffmpeg_command = ['ffmpeg','-loglevel','error', '-y', '-i', out_vid_name, '-i', path_input, '-c', 'copy', '-map', '0:0', '-map', '1:1', final_vid_path]
        subprocess.run(ffmpeg_command)

if __name__ == "__main__":
    run(path_model=path_model,path_input=path_input,path_results=path_results,class_confidence_dict=class_confidence_dict,
         num_imgs=num_imgs,adjust_fraction=adjust_fraction,img_quality=img_quality,
         save_FLAG=save_FLAG,save_bbox=save_bbox,save_txt=save_txt,save_blur=save_blur,
         do_trimming=do_trimming,write_frames_trim=write_frames_trim,start_time=start_time,duration=duration,
         video_reader=video_reader,video_writer=video_writer,
         pred_batch=pred_batch,write_encoding=write_encoding,skip_sound=skip_sound)
