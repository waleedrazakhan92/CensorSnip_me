import cv2
import os
import numpy as np

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from moviepy.editor import VideoFileClip, concatenate_videoclips
from ffmpegcv import VideoWriter

from tqdm import tqdm
import shutil

from utils.misc_utils import make_folders_multi

from datetime import timedelta
import subprocess

def seconds_to_sexagesimal_string(seconds):
    timedelta_object = timedelta(seconds=seconds)
    hours, remainder = divmod(timedelta_object.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


def trim_video_and_extract_frames(video_path,path_write,start_time=None,duration=None,write_video=True,write_frames=True,save_ext='.png'):
    clip = VideoFileClip(video_path) ## just to read duration

    video_name,video_ext = os.path.splitext(video_path.split('/')[-1])
    path_write_imgs = os.path.join(path_write,'frames/')
    path_write_vid = os.path.join(path_write,'videos/')
    temp_vid_path = os.path.join(path_write_vid,video_name+'_trimmed'+video_ext)

    make_folders_multi(path_write,path_write_imgs,path_write_vid)
    # Cut the video for desired duration
    if (start_time==None) and (duration==None):
        shutil.copy(video_path,temp_vid_path)
    else:
        ##ffmpeg_extract_subclip(video_path, start_time, duration, targetname=temp_vid_path)
        if (':' in start_time)==False or (':' in duration)==False:
            start_time_sexa = seconds_to_sexagesimal_string(int(start_time))
            duration_sexa = seconds_to_sexagesimal_string(int(duration))
        else:
            start_time_sexa = str(start_time)
            duration_sexa = str(duration)


        print('----------------------------')
        print('Trimming.... From {}, Duration {}'.format(start_time_sexa,duration_sexa))
        print('----------------------------')
        ## freezed frames problem
        ## https://superuser.com/questions/1167958/video-cut-with-missing-frames-in-ffmpeg/1168028#1168028
        ##trim_command = ['ffmpeg','-y', '-i', video_path, '-ss', start_time, '-to', duration, '-c:v', 'copy','-c:a', 'copy', temp_vid_path]
        ##trim_command = ['ffmpeg','-y','-ss',start_time_sexa,'-t',duration_sexa,'-i', video_path,'-c','copy','-avoid_negative_ts', 'make_zero',temp_vid_path]
        trim_command = ['ffmpeg','-loglevel','error','-y','-ss',start_time_sexa,'-t',duration_sexa,
        '-i', video_path,'-c:v','copy','-c:a','copy','-avoid_negative_ts', 'make_zero',temp_vid_path]

        print('Trim Command:')
        print(trim_command)
        subprocess.run(trim_command)

    if write_frames==True:
        # Read the cut video and write frames in a folder
        cap = cv2.VideoCapture(temp_vid_path)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imwrite(path_write_imgs + f'{frame_count}{save_ext}', frame)
            frame_count += 1

    if write_video==False:
        os.remove(temp_vid_path)

    return temp_vid_path,path_write_imgs

def get_audo_from_video(in_vid_path,out_audio_path):
    ffmpeg_extract_audio(in_vid_path,out_audio_path)

def sort_func(e):
    return int(os.path.splitext(e)[0])

def get_sorted_frames(in_dir):
    sorted_list = os.listdir(in_dir)
    sorted_list.sort(key=sort_func)
    return sorted_list

def append_complete_path(in_path,sorted_frames_list):
    sorted_paths_list = list(map(lambda x: in_path + x, sorted_frames_list))
    return sorted_paths_list

def drop_unwanted_frames(video_path,out_vid_path,frames_to_drop):
    clip = VideoFileClip(video_path)
    frame_rate = clip.fps

    # Calculate timepoints to cut
    time_to_cut = [frame / frame_rate for frame in frames_to_drop]

    # Generate subclips, excluding the frames to drop
    last_t = 0
    subclips = []
    for t in time_to_cut:
        # Ensure the timepoint is within the video duration
        if 0 < t < clip.duration:
            subclips.append(clip.subclip(last_t, t))
            last_t = t + (1 / frame_rate)

    # Add the last part of the video
    if last_t < clip.duration:
        subclips.append(clip.subclip(last_t))

    final_clip = concatenate_videoclips(subclips)
    final_clip.write_videofile(out_vid_path, codec="libx264", audio_codec="aac",fps=frame_rate,threads=4, preset='ultrafast')


def write_vid_from_frames(all_images,path_write_vid,frame_width=None,frame_height=None,fps=30):
    if frame_width==None or frame_height==None:
        frame_width,frame_height = cv2.imread(all_images[0]).shape[:2]

    # vid_writer = cv2.VideoWriter(path_write_vid,cv2.VideoWriter_fourcc(*'MJPG'),
    #                              fps, (frame_height,frame_width))
    vid_writer = VideoWriter(path_write_vid, None, fps, (frame_height,frame_width))

    for i in tqdm(range(0,len(all_images))):
        img_path = all_images[i]
        img = cv2.resize(cv2.imread(img_path),(frame_height,frame_width)).astype('uint8')
        vid_writer.write(img)

    vid_writer.release()

def add_sound_back(vid_original,vid_input,vid_final):
    audio_file = VideoFileClip(vid_original).audio
    video_file = VideoFileClip(vid_input)
    final_file = video_file.set_audio(audio_file)
    final_file.write_videofile(vid_final)
