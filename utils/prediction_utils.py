############################
## Prediction utils
############################
import numpy as np
import os

class one_pred_class:
    def __init__(self, b_conf,bbox_class,b_color,c_lab,b_txt,start_point,end_point,
                    x_c_norm,y_c_norm,b_w_norm,b_h_norm):
        self.b_conf = b_conf
        self.bbox_class = bbox_class
        self.b_color = b_color
        self.c_lab = c_lab
        self.b_txt = b_txt

        self.start_point = start_point
        self.end_point = end_point
        self.x_c_norm = x_c_norm
        self.y_c_norm = y_c_norm
        self.b_w_norm = b_w_norm
        self.b_h_norm = b_h_norm

def extract_prediction_points(pred_porn,p,box_color_dict,label_dict,adjust_fraction,img_h,img_w):
    bbox = np.squeeze(pred_porn[p].boxes.xywh.cpu().numpy())
    b_conf = float(np.squeeze(pred_porn[p].boxes.conf.cpu().numpy()))
    bbox_class = int(np.squeeze(pred_porn[p].boxes.cls.cpu().numpy()))

    pred_x,pred_y = bbox[0],bbox[1]
    pred_w,pred_h = bbox[2],bbox[3]

    pred_w = pred_w*adjust_fraction
    pred_h = pred_h*adjust_fraction

    x_start = pred_x-(pred_w/2)
    y_start = pred_y-(pred_h/2)

    start_point = (x_start,y_start)
    end_point = (x_start+pred_w,
                y_start+pred_h)

    start_point = tuple(map(int, start_point))
    end_point = tuple(map(int, end_point))

    if type(box_color_dict)==dict:
        b_color = box_color_dict[bbox_class]
    else:
        b_color = (0,255,0)

    b_txt = str(round(b_conf,3))

    if type(label_dict)==dict:
        c_lab = label_dict[bbox_class]
    else:
        c_lab = bbox_class#label_dict[bbox_class]

    x_c_norm = pred_x/img_w
    y_c_norm = pred_y/img_h
    b_w_norm = pred_w/img_w
    b_h_norm = pred_h/img_h

    one_pred = one_pred_class(b_conf,bbox_class,b_color,c_lab,b_txt,start_point,end_point,
                    x_c_norm,y_c_norm,b_w_norm,b_h_norm)
    return one_pred

def pass_fail_image(pred_porn,box_color_dict,label_dict,adjust_fraction,class_confidence_dict,img_h,img_w):

    pass_fail_dict = {}
    for p in range(0,len(pred_porn)):
        one_pred = extract_prediction_points(pred_porn,p,box_color_dict,label_dict,adjust_fraction,img_h,img_w)
        if (one_pred.bbox_class not in pass_fail_dict) and (one_pred.b_conf>=class_confidence_dict[one_pred.bbox_class]):
            pass_fail_dict[one_pred.bbox_class] = True

    return pass_fail_dict

def redo_pass_fail(all_predictions,class_confidence_dict):

    failed_images = []
    passed_images = []
    for p in range(0,len(all_predictions)):
        pass_fail_dict = {}
        for q in range(0,len(all_predictions[p])):
            one_pred = all_predictions[p][q]
            if (one_pred.bbox_class not in pass_fail_dict) and (one_pred.b_conf>=class_confidence_dict[one_pred.bbox_class]):
                pass_fail_dict[one_pred.bbox_class] = True

        if pass_fail_dict=={}:
            passed_images.append(p)
        else:
            failed_images.append(p)

    return failed_images,passed_images

def calculate_pass_fail(all_predictions,class_confidence_dict):

    failed_images = []
    passed_images = []
    all_frame_preds = {}
    for p in range(0,len(all_predictions)):
        pass_fail_dict = {}

        one_frame_pred = []
        for q in range(0,len(all_predictions[p])):
            one_pred = all_predictions[p][q]
            one_frame_dict = {'bbox_class':one_pred.bbox_class, 'b_conf':one_pred.b_conf}
            one_frame_pred.append(one_frame_dict)

            if (one_pred.bbox_class not in pass_fail_dict) and (one_pred.b_conf>=class_confidence_dict[one_pred.bbox_class]):
                pass_fail_dict[one_pred.bbox_class] = True
            
        all_frame_preds[p] = one_frame_pred

        if pass_fail_dict=={}:
            passed_images.append(p)
        else:
            failed_images.append(p)

    return failed_images,passed_images,all_frame_preds

def write_preds_txt_file(txt_file_path,one_txt_line):
    if not os.path.isfile(txt_file_path):
        with open(txt_file_path, 'a') as fp:
            fp.write("%s\n" % one_txt_line)
    else:
        with open(txt_file_path, 'r') as fp:
            existing_content = fp.read()
            new_content = existing_content + one_txt_line
        with open(txt_file_path, 'w') as fp:
            fp.write("%s\n" % new_content)

