# IP_CCTV Acess Libaries
import http.client
from base64 import b64encode
# Torch/Detectron 2 Libraries
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#Import JSON for Data Transfer
import json
import csv
import time
import threading
import os
import socket
#Lists to store object detections and keypoints
detections_list=[]
keypoints_list=[]

#CSV File Paths
csv_keypoint_path="./csv/keypoints.csv"
csv_detections_path="./csv/detections.csv"

#=============================================
#   Reading from ONVIF compatible IP Camera
#=============================================
def connect_cctv(username,password,ip_addr,get_request):
    cctv_credentials = b64encode(str.encode("{}:{}".format(username,password))).decode("ascii")
    _headers = { 'Authorization' : 'Basic %s' %  cctv_credentials }
    conn=http.client.HTTPConnection(ip_addr,80)
    conn.request("GET",get_request,headers=_headers)
    resp=conn.getresponse()
    conn.close()
    return resp.read()

def Get_CurrentFrame_CCTV(username,password,ip_addr):
    raw_img=connect_cctv(username,password,ip_addr,"/tmpfs/auto.jpg")
    raw_img=np.frombuffer(raw_img,dtype='int8')
    return cv2.imdecode(raw_img,1)

#===============================
#   Socket Operations
#================================
def Send_JPEG_Frame(ip_addr,port,frame):
    #Init the socket
    s=socket.socket()
    s.connect((ip_addr,port))
    
    #Encode OpenCv frame in jpeg
    frame=cv2.resize(frame,(640,480))
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY), 25]
    jpeg_frame = cv2.imencode('.JPEG', frame,encode_param)[1]
    jpeg_string=jpeg_frame.tostring()
    print("Out Image Size: {} bytes".format(len(jpeg_string)))
    s.sendall(jpeg_string)
    s.close()
    
#=================
#   Read Labels
#=================
def Read_COCO_Names():
    with open('coco.names','r') as cn:
        coco_names=cn.readlines()
    return coco_names

#=====================================
#   Convert to JSON
#====================================
def To_JSON(input_dict,collective_name):
    ret_json_string="{"+"\"{}\":[".format(collective_name)
    #Parse each item into json
    count=0
    for item in input_dict:
        count+=1
        if count!=len(input_dict):
            ret_json_string+=json.dumps(item,indent=2)+","
        else:
            ret_json_string+=json.dumps(item,indent=2)
    ret_json_string+="]}"
    return ret_json_string

#==============================================
#   File I/O Opperations
#==============================================

detections_write_flag=False
keypoints_write_flag=False

def Write_Detections_ToFile():
    print("nDetections_List: {}".format(len(detections_list)))
    if len(detections_list)>0:
        with open('{}'.format(csv_detections_path),"a") as file:
            w=csv.DictWriter(file,detections_list[0].keys())
            w.writerows(detections_list)
            detections_list.clear()
    threading.Timer(120,Write_Detections_ToFile).start()

def Write_Keypoints_ToFile():
    print("nKeypoints_List: {}".format(len(keypoints_list)))
    if len(keypoints_list)>0:
        with open('{}'.format(csv_keypoint_path),'a') as file:
            w=csv.DictWriter(file,keypoints_list[0].keys())
            w.writerows(keypoints_list)
            keypoints_list.clear()
    threading.Timer(120,Write_Detections_ToFile).start()

#=============================
#   Check Social Distancing
#=============================
def Is_Socially_Distanced(local_keypoints_list,x_dist_perc,frame_width):
    for p1 in local_keypoints_list:
        for p2 in local_keypoints_list:
            if p1 != p2:
                shoulder_rl=abs(float(p1["right_shoulder"]['x'])-float(p2["left_shoulder"]['x']))
                shoulder_lr=abs(float(p1["left_shoulder"]['x'])-float(p2["right_shoulder"]['x']))

                shld_rl_perc=(shoulder_rl/frame_width)*100
                shld_lr_perc=(shoulder_lr/frame_width)*100
                print("Shoulder R-L Distance (px): {} | {},{}".format(shld_rl_perc,p1['nose']['x'],p2['nose']['x']))
                print("Shoulder L-R Distance (px): {} | {},{}".format(shld_lr_perc,p1['nose']['x'],p2['nose']['x']))

                if shld_rl_perc<x_dist_perc:
                    return False
                if shld_lr_perc<x_dist_perc:
                    return False
    return True

#======================================
#   Setting up the detector
#======================================

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.DEVICE='cpu' 
predictor = DefaultPredictor(cfg)
#-------------------------------------------------------------------------------------------------------------------
coco_names=Read_COCO_Names()
print(coco_names)
while True:
    #Grab Frames
    #Get_CurrentFrame_CCTV("admin","admin","192.168.1.12")
    im=Get_CurrentFrame_CCTV("admin","admin","192.168.1.12")
    im=cv2.resize(im,(854,480))
    #Get outputs from predictor
    outputs = predictor(im)
    h,w,c=im.shape
    print("H:{} W:{} C:{}".format(h,w,c))
    print('='*100)
    local_keypoints_list=[]
    local_detections_list=[]
  
    #======================================
    #   Etract Predictions and Keypoints
    #======================================
    #-----------------------------------------------------------
    print("Detected {} object(s).".format(len(outputs["instances"].pred_boxes)))

    
    for pred_idx in range(len(outputs["instances"].pred_boxes)):
        detection={
            'date_time':time.ctime(),
            'class':str(coco_names[outputs["instances"].pred_classes[pred_idx]]).replace('\n',''),
             'x': str(outputs["instances"].pred_boxes[pred_idx][0].tensor.cpu().numpy()[0][0]),
             'y': str(outputs["instances"].pred_boxes[pred_idx][0].tensor.cpu().numpy()[0][1]),
             'w': str(outputs["instances"].pred_boxes[pred_idx][0].tensor.cpu().numpy()[0][2]),
             'h': str(outputs["instances"].pred_boxes[pred_idx][0].tensor.cpu().numpy()[0][3])
        }
        detections_list.append(detection)
        local_detections_list.append(detection)

   
    #---------------------------------------------------------
   
    try:
        for det_keypoints in outputs["instances"].pred_keypoints:
            person_keypoint={
                "date_time":time.ctime(),
                "nose":{'x':str(det_keypoints.cpu().numpy()[0][0]),'y': str(det_keypoints.cpu().numpy()[0][1]), 'conf': str(det_keypoints.cpu().numpy()[0][2])},
                "left_eye":{'x':str(det_keypoints.cpu().numpy()[1][0]),'y': str(det_keypoints.cpu().numpy()[1][1]), 'conf': str(det_keypoints.cpu().numpy()[1][2])},
                "right_eye":{'x':str(det_keypoints.cpu().numpy()[2][0]),'y': str(det_keypoints.cpu().numpy()[2][1]), 'conf': str(det_keypoints.cpu().numpy()[2][2])},
                "left_ear":{'x':str(det_keypoints.cpu().numpy()[3][0]),'y': str(det_keypoints.cpu().numpy()[3][1]), 'conf': str(det_keypoints.cpu().numpy()[3][2])},
                "right_ear":{'x':str(det_keypoints.cpu().numpy()[4][0]),'y': str(det_keypoints.cpu().numpy()[4][1]), 'conf': str(det_keypoints.cpu().numpy()[4][2])},
                "left_shoulder":{'x':str(det_keypoints.cpu().numpy()[5][0]),'y': str(det_keypoints.cpu().numpy()[5][1]), 'conf': str(det_keypoints.cpu().numpy()[5][2])},
                "right_shoulder":{'x':str(det_keypoints.cpu().numpy()[6][0]),'y': str(det_keypoints.cpu().numpy()[6][1]), 'conf': str(det_keypoints.cpu().numpy()[6][2])},
                "left_elbow":{'x':str(det_keypoints.cpu().numpy()[7][0]),'y': str(det_keypoints.cpu().numpy()[7][1]), 'conf': str(det_keypoints.cpu().numpy()[7][2])},
                "right_elbow":{'x':str(det_keypoints.cpu().numpy()[8][0]),'y': str(det_keypoints.cpu().numpy()[8][1]), 'conf': str(det_keypoints.cpu().numpy()[8][2])},
                "left_wrist":{'x':str(det_keypoints.cpu().numpy()[9][0]),'y': str(det_keypoints.cpu().numpy()[9][1]), 'conf': str(det_keypoints.cpu().numpy()[9][2])},
                "right_wrist":{'x':str(det_keypoints.cpu().numpy()[10][0]),'y': str(det_keypoints.cpu().numpy()[10][1]), 'conf': str(det_keypoints.cpu().numpy()[10][2])},
                "left_hip":{'x':str(det_keypoints.cpu().numpy()[11][0]),'y': str(det_keypoints.cpu().numpy()[11][1]), 'conf': str(det_keypoints.cpu().numpy()[11][2])},
                "right_hip":{'x':str(det_keypoints.cpu().numpy()[12][0]),'y': str(det_keypoints.cpu().numpy()[12][1]), 'conf': str(det_keypoints.cpu().numpy()[12][2])},
                "left_knee":{'x':str(det_keypoints.cpu().numpy()[13][0]),'y': str(det_keypoints.cpu().numpy()[13][1]), 'conf': str(det_keypoints.cpu().numpy()[13][2])},
                "right_knee":{'x':str(det_keypoints.cpu().numpy()[14][0]),'y': str(det_keypoints.cpu().numpy()[14][1]), 'conf': str(det_keypoints.cpu().numpy()[14][2])},
                "left_ankle":{'x':str(det_keypoints.cpu().numpy()[15][0]),'y': str(det_keypoints.cpu().numpy()[15][1]), 'conf': str(det_keypoints.cpu().numpy()[15][2])},
                "right_ankle":{'x':str(det_keypoints.cpu().numpy()[16][0]),'y': str(det_keypoints.cpu().numpy()[16][1]), 'conf': str(det_keypoints.cpu().numpy()[16][2])}
            }
            keypoints_list.append(person_keypoint)
            local_keypoints_list.append(person_keypoint)
            
    except:
        print("Error: Could not extract Keypoint data from the frame.")

    #----------------------------------------------------------------------
    print("Detected Objects: ")

    for det_obj in local_detections_list:
        print("Class: {} x:{} y:{}".format(det_obj['class'],det_obj['x'],det_obj['y']))

    print("Detected Keypoints: ")
    for det_kpnt in local_keypoints_list:
        print("Nose:{} Right Eye:{} Left Elbow:{}".format(det_kpnt['nose'],det_kpnt['right_eye'],det_kpnt['left_elbow']))


    print("Is Frame Socially Distanced? {}".format(Is_Socially_Distanced(local_keypoints_list,int(10),w)))

    if detections_write_flag==False and len(detections_list)>0:
        #Write headers-only to file
        with open('{}'.format(csv_detections_path),'a') as file:
            w=csv.DictWriter(file,detections_list[0].keys())
            w.writeheader()
        Write_Detections_ToFile()
        detections_write_flag=True

    if keypoints_write_flag==False and len(keypoints_list)>0:
        #Write headers-only to file
        with open('{}'.format(csv_keypoint_path),'a') as file:
            w=csv.DictWriter(file,keypoints_list[0].keys())
            w.writeheader()
        Write_Keypoints_ToFile()
        keypoints_write_flag=True

    #Output to image
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #Output image to gui
    out_img=out.get_image()[:, :, ::-1]
    Send_JPEG_Frame("192.168.1.4",1024,out_img)
