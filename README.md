# CensorSnip
An advanced tool leveraging machine learning and Python technology. Our intelligent system analyzes and detects inappropriate scenes, seamlessly removing them to provide a curated and safe viewing experience.


# Instructions:
To control the strictness of the algorithm, set the *--class_confidence_dict* to low values like 0.2 or 0.3 etc.
Below is the index of the probability dictionary:
```
['EXPOSED_BREAST_F', 'EXPOSED_BUTTOCKS', 'EXPOSED_GENITALIA_F', 'EXPOSED_GENITALIA_M', 'KISS']
```
The first index represents exposed breasts, the second buttocks and so on.

```
Clone the repository:
git clone repository.git

Install the requirements:
pip install -r requirements.txt
```
Run the command `python3 dummy_main.py` to filter the haram content.
# Code parameters:
```
--path_model                Path of the pretrained model checkpoint.
--path_results              Path to save the results
--path_input                Path of the input video file. Or a folder of images.

## Optional flags
--class_confidence_dict     (important images/video) List of class probabilities to filter the content. Lower the number means tighter filtering. Type list. Value [0-1] 
--adjust_fraction           (optional images/video) Manually adjust the bounding box size if you want to display the results. Type fraction. Value float. default 1. bigger number means bigger bounding box.
--save_FLAG                 (optional images/video) Save the intermediate images. Bool flag
--save_bbox                 (optional images/video) Save the resultant images with bounding boxes drawn on them.
--save_blur                 (optional images/video) Save the images with explicit regions blurred in them.
--save_txt                  (optional images/video) Save the txt files with results in them. 
--do_trimming               (video) If you want to trim the input video to make a shorter version. Bool flag.
--write_frames_trim         (video) Write frames from the trimmed video on the disc if you chose to do trimming. Bool flag
--start_time                (video) Start time in seconds or sexagesimal format for video trimming. Type int/str. Value must be less than the size of video in seconds.
--duration                  (video) Duration in seconds or sexagesimal format for how long you want to trim the video.
--skip_sound                (video) Skip the final merging of the audio back to the video for faster inference. Bool flag.
--num_imgs                  (images) If you want to manually select the number if images you want to process. Type int. Value less than number of frames in a video.
--pred_batch                (optional images/video) If you want to do batch predictions. Type int. Depends on the size of GPU/RAM.
--video_reader              (optional video) The reader with which to read the frames from video.
--video_writer              (optional video) The writer with which to write the final video.
--write_encoding            (optional video) The encoding in which to write video frames in the final video.
```

# Sample commands
Note that there are certain parameters that you need to set if you're working with videos and others if you're working with audios.
## Best command (with trimming)
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.7 0.7 0.7 0.7 0.7 \  ## list of 5 probabilities i.e 5 classes in the model
    --do_trimming \
    --start_time '0:0:0' \
    --duration '0:5:0' \
    --pred_batch 256 \
    --video_reader 'gpu_ffmpeg' \
    --video_writer 'gpu_ffmpeg' \
    --write_encoding 'hevc_nvenc'
```
## Best command (without trimming)
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.2 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --pred_batch 256 \
    --video_reader 'gpu_ffmpeg' \
    --video_writer 'gpu_ffmpeg' \
    --write_encoding 'hevc_nvenc'
```
## Sample Command 1 (complete video):
In case you want to process the whole video you need to set the flags like these:
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.7 0.7 0.7 0.7 0.7 \  ## list of 5 probabilities i.e 5 classes in the model
```
## Sample Command 2 (trimmed video):
In case you want to run the inference on only a certain time frame of a video i.e trim a video, you MUST set the following flags to their values:
You can either set the values in seconds or sexagesimal format. The sexagesimal format is hh:mm:ss. So the below commant is cropping the video from 0hours:0minutes:0seconds to 0hours:5minutes:0seconds 

--do_trimming            
--start_time '0:0:0' 
--duration '0:5:0' 
OR
--do_trimming            
--start_time 0
--duration 300 


The do_trimming flag will tell the model to first perform trimming on the video. The inference will be run on that trimmed video.

```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.7 0.7 0.7 0.7 0.7 \  ## list of 5 probabilities i.e 5 classes in the model
    --do_trimming \
    --start_time '0:0:0' 
    --duration '0:5:0' 
```
## Sample Command 3 (video without sound):
For very long videos the model takes very long time to drop explicit frames and recompile the video with synchronized sound. So if the sound is not important to you then
simply recompile the video from the saved frames to skip sound. For that you need so set the following flags accordingly:
--skip_sound
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.7 0.7 0.7 0.7 0.7 \  ## list of 5 probabilities i.e 5 classes in the model
    --skip_sound
```

## Sample Command 4 (all important variables)
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.2 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --do_trimming \
    --start_time '0:0:0' \
    --duration '0:5:0' \
    --pred_batch 256 \
    --video_reader 'gpu_ffmpeg' \
    --video_writer 'gpu_ffmpeg' \
    --write_encoding 'hevc_nvenc'
```
## Sample Command (images folder):
If you want to run inference on a directory of images then the following command will run inference on all the images and store them in a desired directory.
*Make sure to have atleast one of the four flags raised to visualize the results.*
--save_FLAG     Will save explicit images in one directory and the non-explicit in the other
--save_bbox     Will save explicit images(with drawn bounding boxes) in one directory and the non-explicit in the other 
--save_blur     Will save explicit images(with blurred explicit regions) in one directory and the non-explicit in the other 
--save_txt      Will save the text files with results for explicit images in one directory and the non-explicit in the other
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'folder/with/explicit_images/' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.7 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --adjust_fraction 1 \
    --save_FLAG \
    --save_bbox \
    --save_blur \
    --save_txt 
```
You can limit the number of images on which you want to run inference on by adding the num_imgs flag. The resultant command will look like this:
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'folder/with/explicit_images/' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.7 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --adjust_fraction 1 \
    --save_FLAG \
    --save_bbox \
    --save_blur \
    --save_txt \
    --num_imgs 100
```
