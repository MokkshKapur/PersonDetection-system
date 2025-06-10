import cv2
from PIL import Image
from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

def process_video(video_path, output_dir, new_directory, target_fps=5):
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the original frames per second (FPS) of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval for skipping frames
    frame_interval = int(original_fps / target_fps)

    frame_num = 0
    processed_frame_count = 0
    person_frame_count = 0

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(new_directory, exist_ok=True)

    while True:
        ret, frame = cap.read()

        # Break the loop if the video is over
        if not ret:
            break

        frame_num += 1

        # Only process and save every nth frame based on frame_interval
        if frame_num % frame_interval == 0:
            # Save the frame to the original output directory
            frame_filename = os.path.join(output_dir, f"frame_{processed_frame_count + 1:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            processed_frame_count += 1

            # Run YOLOv8 inference on the frame
            results = model(frame)  # Perform inference on the frame

            # Process detections (bounding boxes, confidence, and class)
            detections = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes else []
            class_ids = results[0].boxes.cls.cpu().numpy() if results and results[0].boxes else []
            confidences = results[0].boxes.conf.cpu().numpy() if results and results[0].boxes else []

            # Check if a person (class ID 0) is detected
            person_detected = any(class_id == 0 for class_id in class_ids)

            if person_detected:
                # Draw bounding boxes on the frame
                for det, class_id, conf in zip(detections, class_ids, confidences):
                    x1, y1, x2, y2 = det
                    label = f"Person: {conf:.2f}" if class_id == 0 else f"Class {int(class_id)}: {conf:.2f}"
                    color = (0, 255, 0)  # Green color for bounding boxes
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Increment the counter before saving the annotated frame
                person_frame_count += 1

                # Save the annotated frame to the detected frames directory
                annotated_frame_filename = os.path.join(new_directory, f"annotated_frame_{person_frame_count:04d}.jpg")
                cv2.imwrite(annotated_frame_filename, frame)

    cap.release()
    return processed_frame_count, person_frame_count

# Paths for the input video and output directories
video_path = r"yt1z.net - CCTV footages lay bare the horrors of chain snatching robbers in Chennai (360p).mp4"
output_directory = r"frames4"  # All frames at 5 FPS
new_directory = r"detected_frames_cctv1"  # Frames with person detection

# Call the function to process the video
all_frame_count, person_frame_count = process_video(video_path, output_directory, new_directory)
print(f"Processed {all_frame_count} frames at 5 FPS and saved them to {output_directory}")
print(f"Saved {person_frame_count} annotated frames with person detection to {new_directory}")
