import streamlit as st
from detection import detect_people,detect_people2
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import datetime
import wget
import time


st.title("Social Distancing, Face Mask, Face Shield Detector")
st.subheader('A Social Distancing, Face Mask, Face Shield Monitoring System Using Yolov4 Algorithm')

st.subheader('Test Demo Video Or Try Live Detection')
option = st.selectbox('Choose your option',
                    ('Demo1', 'Demo2', 'Try Live Detection Using Webcam'))


MIN_CONF = 0.0
NMS_THRESH = 0.25


USE_GPU = bool(True)


MIN_DISTANCE = 90

# file_url = 'https://pjreddie.com/media/files/yolov3.weights'
# file_name = wget.download(file_url)

labelsPath = "/app/socialdistancing_facemask_faceshield-detection/StreamlitApp/model/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")



#weightsPath = "yolo-coco/yolov3.weights"
weightsPath = "/app/socialdistancing_facemask_faceshield-detection/StreamlitApp/model/yolov4-tiny.weights"
configPath = "/app/socialdistancing_facemask_faceshield-detection/StreamlitApp/model/yolov4-tiny.cfg"


labelsPath2 = "/app/socialdistancing_facemask_faceshield-detection/StreamlitApp/model/obj.names"
LABELS2 = open(labelsPath2).read().strip().split("\n")

weightsPath2 = "/app/socialdistancing_facemask_faceshield-detection/StreamlitApp/model/yolov4-tiny-3l-obj_best.weights"
configPath2 = "/app/socialdistancing_facemask_faceshield-detection/StreamlitApp/model/yolov4-tiny-3l-obj.cfg"


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

net2 = cv2.dnn.readNetFromDarknet(configPath2, weightsPath2)


if USE_GPU:

    st.info("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

ln2 = net2.getLayerNames()
ln2 = [ln2[i[0] - 1] for i in net2.getUnconnectedOutLayers()]

if st.button('Start'):

    st.info("[INFO] loading YOLO from disk...")
    st.info("[INFO] accessing video stream...")
    if option == "Demo1":
        vs = cv2.VideoCapture("/app/socialdistancing_facemask_faceshield-detection/StreamlitApp/test3.mp4")
    elif option == "Demo2":
        vs = cv2.VideoCapture("/app/socialdistancing_facemask_faceshield-detection/StreamlitApp/rizalpark.mp4")
    else:
        vs = cv2.VideoCapture(0)
    writer = None

    image_placeholder = st.empty()

    while True:

        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)

        prevTime = 0
        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime

        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
        results2 = detect_people2(frame, net2, ln2, personIdx=LABELS2.index("person"))

        violate = set()

        if len(results) >= 2:

            centroids = np.array([r[2] for r in results])

            
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):

                    if D[i, j] < MIN_DISTANCE:

                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):

            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)
                

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)
        
        for (i, (prob, bbox, names)) in enumerate(results2):

            (startX, startY, endX, endY) = bbox
            color = (0, 255,247)
            
            if names == 0:
                label = 'face_shield'
            elif names == 1:
                label = 'no face_shield'
            elif names == 2:
                label = 'face_mask'
            elif names == 3:
                label = 'no face_mask'
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        datet = str(datetime.datetime.now())
        text1 = "FPS: {}".format(str(fps)[0:1])
        frame = cv2.putText(frame, text1, (0, 35), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

        display = 1
        if display > 0:
            image_placeholder.image(frame, caption='Live Social Distancing Monitor Running..!', channels="BGR")

        if writer is not None:
            writer.write(frame)

st.success("Video Done")