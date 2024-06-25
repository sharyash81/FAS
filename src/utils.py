import cv2 
import face_recognition
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
    
    
def recognize_faces(image_name, output_dir):
    image_path = f'/Images/main_frame/{image_name}.jpg'
    # Load the image from file
    frame = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, face_location in enumerate(face_locations):
        # Extract the face location
        top, right, bottom, left = face_location

        # Add a margin to the face location
        margin = 30
        top = max(0, top - margin)
        right = min(frame.shape[1], right + margin)
        bottom = min(frame.shape[0], bottom + margin)
        left = max(0, left - margin)

        # Crop the face from the frame
        face_image = frame[top:bottom, left:right]

        # Save the cropped face image
        face_filename = os.path.join(output_dir, f"{image_name}_face{i+1}.jpg")
        cv2.imwrite(face_filename, face_image)
