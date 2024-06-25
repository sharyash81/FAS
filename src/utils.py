import cv2 
import os
def extract_main_frame(video_path, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the index of the main frame (middle frame)
    main_frame_index = total_frames // 2
    
    # Set the video position to the main frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, main_frame_index)
    
    # Read the main frame
    ret, main_frame = cap.read()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the main frame
    main_frame_filename = f'{video_path}_main_frame.jpg'
    main_frame_path = os.path.join(output_dir, main_frame_filename)
    cv2.imwrite(main_frame_path, main_frame)
    
    cap.release()
    print(f"Extracted main frame from {total_frames} total frames.")