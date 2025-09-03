#!/usr/bin/env python3
"""
Human Skeleton Detection using OpenCV and MediaPipe
Optimized for Jetson Orin Nano

Requirements:
pip install opencv-python mediapipe numpy

For Jetson Orin Nano, you might need to install OpenCV with CUDA support:
sudo apt update
sudo apt install python3-opencv
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class SkeletonDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the skeleton detector
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection with optimized settings for Jetson
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Light, 1=Full, 2=Heavy (use 1 for Jetson)
            smooth_landmarks=True,
            enable_segmentation=False,  # Disable for better performance
            smooth_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define skeleton connections for custom drawing
        self.skeleton_connections = [
            # Head and neck
            (0, 1), (1, 2), (2, 3), (3, 7),  # Face outline
            (0, 4), (4, 5), (5, 6), (6, 8),  # Face outline
            (9, 10),  # Mouth
            
            # Torso
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Shoulder to hip
            (23, 24),  # Hips
            
            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            
            # Right arm  
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            
            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            
            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]
    
    def draw_skeleton(self, image, landmarks):
        """
        Draw skeleton connections on the image
        
        Args:
            image: Input image
            landmarks: MediaPipe landmarks
        """
        h, w, _ = image.shape
        
        # Convert landmarks to pixel coordinates
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        # Draw connections
        for connection in self.skeleton_connections:
            start_idx, end_idx = connection
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                
                # Check if landmarks are visible
                start_visibility = landmarks.landmark[start_idx].visibility
                end_visibility = landmarks.landmark[end_idx].visibility
                
                if start_visibility > 0.5 and end_visibility > 0.5:
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw joints
        for i, point in enumerate(points):
            visibility = landmarks.landmark[i].visibility
            if visibility > 0.5:
                cv2.circle(image, point, 4, (0, 0, 255), -1)
                cv2.circle(image, point, 6, (255, 255, 255), 2)
    
    def process_frame(self, frame):
        """
        Process a single frame and detect skeleton
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with skeleton overlay
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Draw skeleton if pose is detected
        if results.pose_landmarks:
            # Option 1: Use MediaPipe's built-in drawing (simpler)
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Option 2: Use custom skeleton drawing (uncomment to use)
            # self.draw_skeleton(frame, results.pose_landmarks)
        
        return frame, results
    
    def get_landmark_coordinates(self, results, frame_shape):
        """
        Extract landmark coordinates in pixel space
        
        Args:
            results: MediaPipe pose results
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            List of (x, y) coordinates for each landmark
        """
        if not results.pose_landmarks:
            return None
        
        h, w = frame_shape[:2]
        landmarks = []
        
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            visibility = landmark.visibility
            landmarks.append((x, y, visibility))
        
        return landmarks

def main():
    """
    Main function to run skeleton detection
    """
    # Initialize skeleton detector
    detector = SkeletonDetector()
    
    # Open camera (0 for default camera, adjust if needed)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution (optimize for Jetson performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting skeleton detection...")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Performance monitoring
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        processed_frame, results = detector.process_frame(frame)
        
        # Add FPS counter
        fps_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = fps_counter / elapsed_time
            fps_counter = 0
            start_time = time.time()
            
            # Display FPS on frame
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add detection status
        if results.pose_landmarks:
            cv2.putText(processed_frame, "Skeleton Detected", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Optional: Get landmark coordinates for further processing
            landmarks = detector.get_landmark_coordinates(results, frame.shape)
            if landmarks:
                # Example: Print nose position (landmark 0)
                nose_x, nose_y, visibility = landmarks[0]
                if visibility > 0.5:
                    cv2.putText(processed_frame, f"Nose: ({nose_x}, {nose_y})", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(processed_frame, "No Skeleton Detected", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Skeleton Detection', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            filename = f"skeleton_detection_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.pose.close()
    print("Skeleton detection stopped")

class SkeletonAnalyzer:
    """
    Additional class for analyzing skeleton data
    """
    
    @staticmethod
    def calculate_angle(p1, p2, p3):
        """
        Calculate angle between three points
        
        Args:
            p1, p2, p3: Points as (x, y) tuples
            
        Returns:
            Angle in degrees
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    @staticmethod
    def detect_pose(landmarks):
        """
        Detect specific poses based on landmark positions
        
        Args:
            landmarks: List of landmark coordinates
            
        Returns:
            String describing detected pose
        """
        if not landmarks or len(landmarks) < 33:
            return "Unknown"
        
        # Extract key points (only use if visibility > 0.5)
        left_shoulder = landmarks[11] if landmarks[11][2] > 0.5 else None
        right_shoulder = landmarks[12] if landmarks[12][2] > 0.5 else None
        left_elbow = landmarks[13] if landmarks[13][2] > 0.5 else None
        right_elbow = landmarks[14] if landmarks[14][2] > 0.5 else None
        left_wrist = landmarks[15] if landmarks[15][2] > 0.5 else None
        right_wrist = landmarks[16] if landmarks[16][2] > 0.5 else None
        
        # Simple pose detection examples
        if (left_wrist and left_shoulder and right_wrist and right_shoulder):
            # Check if both hands are raised
            if (left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]):
                return "Hands Up"
        
        if (left_elbow and left_shoulder and left_wrist):
            # Check for left arm bent
            angle = SkeletonAnalyzer.calculate_angle(
                (left_shoulder[0], left_shoulder[1]),
                (left_elbow[0], left_elbow[1]),
                (left_wrist[0], left_wrist[1])
            )
            if 30 < angle < 120:
                return "Left Arm Bent"
        
        return "Standing"

# Advanced usage example
def advanced_skeleton_detection():
    """
    Advanced skeleton detection with pose analysis
    """
    detector = SkeletonDetector()
    analyzer = SkeletonAnalyzer()
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, results = detector.process_frame(frame)
        
        if results.pose_landmarks:
            # Get landmark coordinates
            landmarks = detector.get_landmark_coordinates(results, frame.shape)
            
            # Analyze pose
            pose_type = analyzer.detect_pose(landmarks)
            
            # Display pose type
            cv2.putText(processed_frame, f"Pose: {pose_type}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.imshow('Advanced Skeleton Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.pose.close()

if __name__ == "__main__":
    # Run basic skeleton detection
    main()
    
    # Uncomment to run advanced version with pose analysis
    # advanced_skeleton_detection()
