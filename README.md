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
--write_video_trim          (video) Write the trim video on the disc if you chose to do trimming. Bool flag.
--write_frames_trim         (video) Write frames from the trimmed video on the disc if you chose to do trimming. Bool flag
--start_time                (video) Start time in seconds for video trimming. Type int. Value must be less than the size of video in seconds.
--end_time                  (video) End time in seconds for video trimming. Type int. Value must be less than the size of video in seconds.
--skip_sound                (video) Skip the final merging of the audio back to the video for faster inference. Bool flag.
--num_imgs                  (images/video) If you want to manually select the number if images you want to process. Type int. Value less than number of frames in a video.
```

## Sample Command (video):
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'explicit_video.mp4' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.2 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --adjust_fraction 1 \
    --save_FLAG \
    --do_trimming \
    --write_video_trim \
    --write_frames_trim \
    --start_time 120 \
    --end_time 420 \
    --skip_sound \
    # --num_imgs 50 \
```
## Sample Command (images folder):
```
python3 dummy_main.py \
    --path_model '/pretrained_models/best_full_v0_640_aug_v2.pt' \
    --path_input 'folder/with/explicit_images/' \
    --path_results '../test_inference/' \
    --class_confidence_dict 0.2 0.2 0.2 0.7 0.2 \  ## list of 5 probabilities i.e 5 classes in the model
    --adjust_fraction 1 \
    --save_FLAG \
    --num_imgs 300
```
