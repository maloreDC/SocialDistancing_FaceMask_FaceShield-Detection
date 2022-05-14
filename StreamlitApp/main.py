import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

import os

from utils import Sound, TimeForSoundChecker, play_alarm, has_violations


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer

import time
import pandas as pd
import math
from itertools import combinations

#change = ClientSettings - RTCConfiguration

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


st.set_page_config(page_title="YoloV4 App", page_icon=":nazar_amulet:")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun4.l.google.com:19302"]}]},
)

# WEBRTC_CLIENT_SETTINGS = ClientSettings(
#     rtc_configuration={"iceServers": [
#         {"urls": ["stun:stun.l.google.com:19302"]}]},
#     media_stream_constraints={
#         "video": True,
#     },
# )


def main():

    st.title("Lite Real time Object Detection WebApp")
    st.subheader("Using YOLOv4 tiny")

    with st.spinner('Wait for the Weights and Configuration files to load'):
        time.sleep(1)
    st.success('Done!')

    st.info("Please wait for 30-40 seconds for the webcam to load with the dependencies")

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown("Social Distancing Violations")
        kpi1_text = st.markdown('0')
    with kpi2:
        st.markdown("Face Mask Violations")
        kpi2_text = st.markdown('0')
    with kpi3:
        st.markdown("Face Shield Violations")
        kpi3_text = st.markdown('0')

    app_object_detection(kpi1_text,kpi2_text,kpi3_text)

    st.error('Please allow access to camera and microphone in order for this to work')
    st.warning(
        'The object detection model might varies due to the server speed and internet speed')


    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


# Threshold Values
Conf_threshold = 0.25
NMS_threshold = 0.25

Conf_threshold2 = 0.30
NMS_threshold2 = 0.30

MIN_DISTANCE = 90

# Colours
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# empty list
class_name = []

#Coco - Server
COCO = "models/coco.names"
OBJ = "models/obj.names"

#Coco - Local
#COCO = "models\\coco.names"


# for reading all the datasets from the coco.names file into the array
with open(COCO, 'rt') as f:
    class_name = f.read().rstrip('\n').split('\n')

with open(OBJ, 'rt') as f:
    class_name2 = f.read().rstrip('\n').split('\n')

# configration and weights file location - Server
model_config_file = "models/yolov4-tiny.cfg"
model_weight = "models/yolov4-tiny.weights"

model_config_file2 = "models/yolov4-tiny-3l-obj.cfg"
model_weight2 = "models/yolov4-tiny-3l-obj_best.weights"

# configration and weights file location - Local
#model_config_file = "models\\yolov4-tiny.cfg"
#model_weight = "models\\yolov4-tiny.weights"


# darknet files
net = cv2.dnn.readNetFromDarknet(model_config_file, model_weight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

net2 = cv2.dnn.readNetFromDarknet(model_config_file2, model_weight2)
net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load Model
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

model2 = cv2.dnn_DetectionModel(net2)
model2.setInputParams(size=(608,608), scale=1/255, swapRB=True)


def is_close(p1, p2):
    """
    #================================================================
    # 1. Purpose : Calculate Euclidean Distance between two points
    #================================================================    
    :param:
    p1, p2 = two points for calculating Euclidean Distance
    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1**2 + p2**2)
    #=================================================================#
    return dst 


def convertBack(x, y, w, h): 
    #================================================================
    # 2.Purpose : Converts center coordinates to rectangle coordinates
    #================================================================  
    """
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x))
    xmax = int(round(x+w))
    ymin = int(round(y))
    ymax = int(round(y+h))

    return xmin, ymin, xmax, ymax


def app_object_detection(kpi1_text,kpi2_text,kpi3_text):

    checker = TimeForSoundChecker()

    class Video(VideoProcessorBase):

        def __init__(self):
            self.scViolators = 0
            self.fmViolators = 0
            self.fsViolators = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")

            classes, scores, boxes = model.detect(
                image, Conf_threshold2, NMS_threshold2)

            classes2, scores2, boxes2 = model2.detect(
                image, Conf_threshold, NMS_threshold)

            if checker.has_been_a_second():
                if has_violations(classes2):
                    play_alarm()
            
            centroid_dict = dict() 
            objectId = 0
            red_zone_list = []
            red_line_list = []
            no_face_mask =[]
            no_face_shield = []

            for i , (classid, score, box) in enumerate (zip(classes, scores, boxes)):
                if classid == 0:
                    centerCoord = (int(box[0]+(box[2]/2)), int(box[1]+(box[3]/2)))
                    cv2.circle(image, centerCoord, 5, (255, 0, 0), 1) 
                    x, y, w, h= box
                    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                    centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax,centerCoord) 
                    objectId += 1

            for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): 
                dx, dy = p1[0] - p2[0], p1[1] - p2[1]   	
                distance = is_close(dx, dy) 			
                if distance < MIN_DISTANCE:						
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)       
                        red_line_list.append(p1[6]) 
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)	
                        red_line_list.append(p2[6])

            for idx, box in centroid_dict.items():
                if idx in red_zone_list:  
                    cv2.rectangle(image, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
                else:
                    cv2.rectangle(image, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)
                self.scViolators = len(red_zone_list)
                
            for check in range(0, len(red_line_list)-1):					
                start_point = red_line_list[check] 
                end_point = red_line_list[check+1]
                check_line_x = abs(end_point[0] - start_point[0])   		
                check_line_y = abs(end_point[1] - start_point[1])	
                if (check_line_x < MIN_DISTANCE) and (check_line_y < 25):			
                    cv2.line(image, start_point, end_point, (255, 0, 0), 2) 

            for (classid, score, box) in zip(classes2, scores2, boxes2):
                if classid != 4:
                    
                    color = COLORS[int(classid) % len(COLORS)]

                    label = "%s : %f" % (class_name2[classid[0]], score)

                    cv2.rectangle(image, box, color, 1)
                    cv2.putText(image, label, (box[0], box[1]-10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                    if classid == 3:
                        no_face_mask.append(score)
                    if classid == 1:
                        no_face_shield.append(score)
                self.fmViolators = len(no_face_mask)
                self.fsViolators = len(no_face_shield)


            return av.VideoFrame.from_ndarray(image, format="bgr24")

    # webrtc_ctx = webrtc_streamer(
    #     key="object-detection",
    #     mode=WebRtcMode.SENDRECV,
    #     client_settings=WEBRTC_CLIENT_SETTINGS,
    #     video_processor_factory=Video,
    #     async_processing=True,
    # )

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
        },
        video_processor_factory=Video,
        async_processing=True,
    )

    while webrtc_ctx.video_processor:
        if webrtc_ctx.video_processor:
            kpi1_text.write(str(webrtc_ctx.video_processor.scViolators))
            kpi2_text.write(str(webrtc_ctx.video_processor.fmViolators))
            kpi3_text.write(str(webrtc_ctx.video_processor.fsViolators))

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in [
        "false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
