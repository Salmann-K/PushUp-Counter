# Import OpenCV for video processing
# Import MediaPipe for pose estimation

import cv2  
import mediapipe as md  

# Initialize drawing utilities from MediaPipe for visualizing pose landmarks

md_drawing = md.solutions.drawing_utils
md_drawing_styles = md.solutions.drawing_styles
md_pose = md.solutions.pose



count = 0  
position = None  



# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture("Storage address")

# Initialize the MediaPipe Pose model with specified detection and tracking confidences

with md_pose.Pose(
    min_detection_confidence=0.7,  # Minimum confidence value for the detection to be considered successful
    min_tracking_confidence=0.7  # Minimum confidence value for the tracking to be considered successful
) as pose:
    while cap.isOpened():                                                                                                  # Loop while the video capture is open
        success, image = cap.read()                                                                                        # Read a frame from the video capture
        if not success:                                                                                                    # Check if the frame was read successfully
            print("Empty Camera")
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                                                                     # Convert the image from BGR to RGB
        result = pose.process(image)                                                                                       # Process the image to detect pose landmarks

        imlist = []                                                                                                        # List to store the landmark positions

        if result.pose_landmarks:                                                                                          # Check if pose landmarks are detected
            # Draw pose landmarks on the image
            md_drawing.draw_landmarks(
                image, result.pose_landmarks, md_pose.POSE_CONNECTIONS
            )
            for id, im in enumerate(result.pose_landmarks.landmark):
                h, w, _ = image.shape                                                                                      # Get the dimensions of the image
                X, Y = int(im.x * w), int(im.y * h)                                                                        # Convert normalized landmark coordinates to pixel values
                imlist.append([id, X, Y])                                                                                  # Append landmark id and its pixel coordinates to the list

        if len(imlist) != 0:  # Check if landmark list is not empty

            if ((imlist[12][2] - imlist[14][2]) >= 15 and (imlist[11][2] - imlist[13][2]) >= 15):
                position = "down"

            if ((imlist[12][2] - imlist[14][2]) <= 5 and (imlist[11][2] - imlist[13][2]) <= 5) and position == "down":
                position = "up"
                count += 1  # Increment the push-up counter
                print(count)


        cv2.putText(image, f"Push-ups: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Push-up counter", image)  # Show the image with pose landmarks and counter
        key = cv2.waitKey(1)  # Wait for 1 ms for a key press
        if key == ord('q'):  # Exit the loop if 'q' key is pressed
            break

cap.release()  
cv2.destroyAllWindows()  
