# CensorSnip
An advanced tool leveraging machine learning and Python technology. Our intelligent system analyzes and detects inappropriate scenes, seamlessly removing them to provide a curated and safe viewing experience.


# Instructions:
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
--save_FLAG                 (images/video) Save the intermediate blurred and bounding box images along with the text labels. Bool flag
--do_trimming               (video) If you want to trim the input video to make a shorter version. Bool flag.
--write_frames_trim         (video) Write frames from the trimmed video on the disc if you chose to do trimming. Bool flag
--start_time                (video) Start time in seconds for video trimming. Type int. Value must be less than the size of video in seconds.
--end_time                  (video) End time in seconds for video trimming. Type int. Value must be less than the size of video in seconds.
--skip_sound                (video) Skip the final merging of the audio back to the video for faster inference. Bool flag.
--num_imgs                  (images) If you want to manually select the number if images you want to process. Type int. Value less than number of frames in a video.
```

# Sample commands
Note that there are certain parameters that you need to set if you're working with videos and others if you're working with audios.
## Sample Command 1 (complete video):
In case you want to process the whole video you need to set the flags like these:
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.2 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
```
## Sample Command 2 (trimmed video):
In case you want to run the inference on only a certain time frame of a video i.e trim a video, you MUST set the following flags to their values:
--do_trimming            
--start_time 120 
--end_time 420 
The do_trimming flag will tell the model to first perform trimming on the video. The inference will be run on that trimmed video.

```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.2 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --do_trimming \
    --start_time 20 \
    --end_time 40 
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
    --class_confidence_dict 0.2 0.2 0.2 0.2 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --skip_sound
```

## Sample Command 4 (all variables)
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.2 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --do_trimming \
    --start_time 120 \
    --end_time 420 \
    --skip_sound 
```
## Sample Command (images folder):
If you want to run inference on a directory of images then the following command will run inference on all the images and store them in a desired directory:
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'folder/with/explicit_images/' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.7 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --adjust_fraction 1 
```
You can limit the number of images on which you want to run inference on by adding the num_imgs flag. The resultant command will look like this:
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'folder/with/explicit_images/' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.7 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --adjust_fraction 1 \
    --num_imgs 100
```
