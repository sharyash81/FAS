import os
import cv2
import numpy as np
import pandas as pd

def extract_frames(video_path, label, output_dir, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames = []
    labels = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_rate == 0:
            frame_name = f"{video_name}_frame_{frame_count}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            labels.append(label)
            frame_count += 1
    cap.release()
    return frames, labels

def preprocess_dataset(data_dir, output_dir, label_file):
    all_frame_paths = []
    all_labels = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for video in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video)
                if video.endswith('.avi'):
                    if video in ['1.avi', '2.avi', 'HR_1.avi', 'HR_4.avi']:
                        label = 1  # live
                    else:
                        label = 0  # spoofed
                    frames, labels = extract_frames(video_path, label, output_dir)
                    all_frame_paths.extend(frames)
                    all_labels.extend(labels)
    
    # Save labels to a CSV file
    label_df = pd.DataFrame({
        'frame_path': all_frame_paths,
        'label': all_labels
    })
    label_df.to_csv(label_file, index=False)

data_dir = '/content/drive/My Drive/CASIA_faceAntisp/train_release'
output_dir = '/content/drive/My Drive/CASIA_faceAntisp/frames' 
label_file = '/content/drive/My Drive/CASIA_faceAntisp/labels.csv'  

preprocess_dataset(data_dir, output_dir, label_file)
print("Preprocessing complete. Frames and labels are saved.")
