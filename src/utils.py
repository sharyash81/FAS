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
    
    
def recognize_faces(image_path):
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

    for face_location in face_locations:
        # Draw a box around each face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Show the output image with recognized faces
    cv2.imshow(frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
