import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def run_skeleton_camera():
    cap = cv2.VideoCapture(0)  

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("카메라에서 영상을 가져올 수 없습니다.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                )

            cv2.imshow('Skeleton Tracking', image)

            if cv2.waitKey(1) & 0xFF == 27: 
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_skeleton_camera()
