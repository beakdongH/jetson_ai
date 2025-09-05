import cv2 as cv
import numpy as np

# Load TF frozen graph
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

def gstreamer_pipeline(
    capture_width=1280, capture_height=720,
    display_width=960, display_height=540,
    framerate=30, flip_method=0
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        % (capture_width, capture_height, framerate,
           flip_method, display_width, display_height)
    )

cap = cv.VideoCapture(gstreamer_pipeline(), cv.CAP_GSTREAMER)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    blob = cv.dnn.blobFromImage(frame, 1.0, (368,368),
                                (127.5,127.5,127.5),
                                swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
