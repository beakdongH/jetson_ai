import cv2
import numpy as np

# OpenPose COCO model parameters
PROTO_FILE = "pose_deploy_linevec.prototxt"
WEIGHTS_FILE = "pose_iter_440000.caffemodel"
N_POINTS = 18  # COCO 모델: 총 18개의 관절 포인트
POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def skeleton_detect():
    window_title = "Skeleton Detect"
    net = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)

    # Jetson Orin Nano에서 CUDA 사용 (OpenCV DNN CUDA 지원 시)
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except:
        print("CUDA 가속을 사용할 수 없어 CPU 모드로 실행합니다.")

    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                frame_height, frame_width = frame.shape[:2]
                in_width = 368
                in_height = 368
                inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height),
                                                 (0, 0, 0), swapRB=False, crop=False)
                net.setInput(inp_blob)
                output = net.forward()

                H = output.shape[2]
                W = output.shape[3]

                # 각 관절 포인트 검출
                points = []
                for i in range(N_POINTS):
                    prob_map = output[0, i, :, :]
                    min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

                    x = int((frame_width * point[0]) / W)
                    y = int((frame_height * point[1]) / H)

                    if prob > 0.1:
                        cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                        cv2.putText(frame, "{}".format(i), (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                        points.append((x, y))
                    else:
                        points.append(None)

                # 관절 연결
                for pair in POSE_PAIRS:
                    part_a = pair[0]
                    part_b = pair[1]

                    if points[part_a] and points[part_b]:
                        cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 2)

                # 창 닫기 여부 확인
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break

                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):  # ESC 또는 q 키 종료
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == "__main__":
    skeleton_detect()
