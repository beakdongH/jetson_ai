#!/usr/bin/env python3
# Jetson Orin/Nano action recognition skeleton
# Layered design: Capture (GStreamer) -> Pose (TensorRT) -> Action (state machine)
# Author: PK용 스켈레톤

import time
import math
import cv2
import numpy as np
from collections import deque

# ------------------------------------------------------------
# 1) GStreamer capture for Jetson CSI camera (nvarguscamerasrc)
# ------------------------------------------------------------
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
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True sync=false"
        % (
            capture_width, capture_height, framerate,
            flip_method,
            display_width, display_height,
        )
    )

class JetsonCamera:
    def __init__(self, pipeline: str):
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open CSI camera via GStreamer")
    def read(self):
        return self.cap.read()
    def release(self):
        self.cap.release()

# ------------------------------------------------------------
# 2) Pose estimator interface (TensorRT)
#    Replace TODO parts with real trt_pose inference & decoding
# ------------------------------------------------------------
class PoseEstimatorTRT:
    def __init__(self, engine_path: str, input_size=(224, 224)):
        self.engine_path = engine_path
        self.input_w, self.input_h = input_size
        # TODO: load TensorRT engine, allocate bindings, CUDA streams, etc.
        # self.context = ...
        # self.bindings = ...
        # self.stream = ...
        # self.topology = ...  # for decoder
        pass

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Resize + normalize to model input; adjust as your model requires
        inp = cv2.resize(frame_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        inp = inp[:, :, ::-1].astype(np.float32)  # BGR -> RGB
        # Example normalization (change to your model’s scheme)
        inp = (inp / 255.0 - 0.5) / 0.5
        inp = np.transpose(inp, (2, 0, 1))  # CHW
        return np.ascontiguousarray(inp[None, ...])  # NCHW

    def infer(self, input_tensor: np.ndarray):
        # TODO: do TensorRT execution and return raw outputs for decoder
        # Example: heatmaps, pafs = self.context.execute_v2(...)
        # return heatmaps, pafs
        return None

    def decode(self, raw_outputs, orig_frame_shape):
        # TODO: use trt_pose or your decoder to convert raw outputs into keypoints
        # Return format: list of persons; each person is dict { "kps": [(x,y,conf), ...] }
        # Keypoint order should be consistent (e.g., COCO 17-keypoints)
        h, w = orig_frame_shape[:2]
        persons = []
        # Dummy example: no detection
        return persons

    def predict(self, frame_bgr: np.ndarray):
        inp = self.preprocess(frame_bgr)
        raw = self.infer(inp)
        persons = self.decode(raw, frame_bgr.shape)
        return persons

# ------------------------------------------------------------
# 3) Simple action recognizer from keypoints
#    Example actions: Hand Raise, Squat, Head Turn (angle/ratio based)
# ------------------------------------------------------------
def angle_3pts(a, b, c):
    # angle at point b for triangle a-b-c
    ax, ay = a; bx, by = b; cx, cy = c
    ab = np.array([ax - bx, ay - by], dtype=np.float32)
    cb = np.array([cx - bx, cy - by], dtype=np.float32)
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    cosang = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

class ActionStateMachine:
    def __init__(self, history=10):
        self.history = deque(maxlen=history)
        self.current = "Idle"
        self.last_change = time.time()

    def update(self, kps):
        """
        kps: list of (x, y, conf) with fixed ordering (e.g., COCO)
        We assume indices:
          5: left shoulder, 6: right shoulder, 7: left elbow, 8: right elbow,
          9: left wrist, 10: right wrist, 11: left hip, 12: right hip,
          13: left knee, 14: right knee, 15: left ankle, 16: right ankle
        Adjust indices to your model’s spec.
        """
        if not kps or len(kps) < 17:
            self.history.append(("NoPerson", 0.0))
            self.current = "Idle"
            return self.current

        conf_thr = 0.3
        def pick(i): 
            return (kps[i][0], kps[i][1]) if kps[i][2] >= conf_thr else None

        ls = pick(5); rs = pick(6)
        le = pick(7); re = pick(8)
        lw = pick(9); rw = pick(10)
        lh = pick(11); rh = pick(12)
        lk = pick(13); rk = pick(14)

        # Feature 1: hand raise if wrist higher (smaller y) than shoulder by margin
        hand_raise = False
        margin = 0.08  # relative to image height (tune)
        if ls and lw:
            hand_raise |= (lw[1] + margin * 1.0 < ls[1])
        if rs and rw:
            hand_raise |= (rw[1] + margin * 1.0 < rs[1])

        # Feature 2: squat if hip-knee vertical distance small and hips lowered
        squat = False
        if lh and rh and lk and rk:
            hip_y = (lh[1] + rh[1]) / 2.0
            knee_y = (lk[1] + rk[1]) / 2.0
            squat |= (knee_y - hip_y) < 0.12  # tune threshold
            # extra: overall hip y compared to shoulder y (body lowered)
            if ls and rs:
                shoulder_y = (ls[1] + rs[1]) / 2.0
                squat &= (hip_y - shoulder_y) < 0.22

        # Feature 3: elbow angle to detect arm bend
        arm_bend = False
        if ls and le and lw:
            ang_l = angle_3pts(ls, le, lw)
            arm_bend |= ang_l < 60.0
        if rs and re and rw:
            ang_r = angle_3pts(rs, re, rw)
            arm_bend |= ang_r < 60.0

        # Simple priority
        if hand_raise:
            state = "HandRaise"
        elif squat:
            state = "Squat"
        elif arm_bend:
            state = "ArmBend"
        else:
            state = "Idle"

        # hysteresis via short history
        self.history.append((state, time.time()))
        states = [s for s, _ in self.history]
        # pick majority in history to stabilize
        stable = max(set(states), key=states.count)
        if stable != self.current:
            self.current = stable
            self.last_change = time.time()
        return self.current

# ------------------------------------------------------------
# 4) Visualization helpers
# ------------------------------------------------------------
def draw_keypoints_and_skeleton(frame, persons, color=(0, 255, 0)):
    for p in persons:
        kps = p.get("kps", [])
        for (x, y, c) in kps:
            if c > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        # TODO: draw bones per your topology if available

def put_hud(frame, fps, state):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f"Action: {state}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)

# ------------------------------------------------------------
# 5) Main loop
# ------------------------------------------------------------
def main():
    window = "Action Recognition"
    cam = JetsonCamera(gstreamer_pipeline(
        capture_width=1920, capture_height=1080,
        display_width=960, display_height=540,
        framerate=30, flip_method=0
    ))

    # Replace engine_path with your TensorRT engine
    pose = PoseEstimatorTRT(engine_path="resnet18_trt_pose.engine", input_size=(224, 224))
    sm = ActionStateMachine(history=8)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    t0 = time.time()
    frame_cnt = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                continue

            persons = pose.predict(frame)  # list of { "kps": [(x,y,conf), ...] }
            # Select primary person (e.g., highest average conf or largest bbox)
            if persons:
                # Example heuristic: use first
                primary = persons[0].get("kps", [])
                state = sm.update(primary)
            else:
                state = sm.update([])

            # Draw
            draw_keypoints_and_skeleton(frame, persons)
            frame_cnt += 1
            if frame_cnt % 10 == 0:
                t1 = time.time()
                fps = 10.0 / (t1 - t0 + 1e-9)
                t0 = t1
            put_hud(frame, fps, state)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
