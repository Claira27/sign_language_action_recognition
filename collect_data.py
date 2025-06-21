import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

# from mediapipe.python.solutions.pose import PoseLandmark

# for landmark in PoseLandmark:
#     print(landmark.name, landmark.value)

mp_holistic = mp.solutions.holistic #Holistic model for pose, face, and hand landmarks
mp_drawing = mp.solutions.drawing_utils # Drawing utilities for visualizing landmarks
mp_face_mesh = mp.solutions.face_mesh
#print(mp_holistic.POSE_CONNECTIONS) # Print the pose connections for reference

# Function to draw landmarks on the image
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the image to RGB
    image.flags.writeable = False # Make the image non-writeable for performance
    results = model.process(image) # make predictions
    image.flags.writeable = True #the image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert the image back to BGR
    return image, results 

# Function to draw landmarks on the image
def draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_face_mesh.FACEMESH_TESSELATION,
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

def extract_keypoints(results):
    # Extract keypoints from the results
    pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4) # 33 landmarks for pose
    face = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3) # 468 landmarks for face
    lh =  np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3) # 21 landmarks for left hand
    rh =  np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten()  if results.right_hand_landmarks else np.zeros(21*3) # 21 landmarks for right hand
    return np.concatenate([lh, rh]) # Concatenate left-hand & right-hand keypoints into a single array

# Actions to recognize
#actions = np.array(['hello', 'thanks', 'iloveyou'])
#actions = np.array(['yes', 'no', 'please', 'sorry','goodbye'])
#actions = np.array(['stop', 'wait', 'go', 'come'])
#actions = np.array(['love', 'F', 'help', 'like', 'dislike', 'happy', 'sad'])
#actions = np.array(['I', 'you', 'it'])
actions = np.array(['who', 'what', 'where', 'when', 'why'])
                    #'tired', 'bored', 'excited', 'scared', 'curious', 'proud', 'embarrassed',
                    # 'this', 'that', 'here', 'there',
                    # 'all', 'some', 'none', 'every']) 
no_sequences = 100 # Number of videos per action
sequence_length = 30 # Length of each video sequence: 30 frames
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Sign_language_action_recognition/data')
#create folder for each action
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

#Keypoints using mediapipe holistic
cap = cv2.VideoCapture(0)
#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through actions
    for action in actions:
        # Hiển thị thông báo BẮT ĐẦU ACTION
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        cv2.putText(image, f'STARTING ACTION: {action.upper()}', 
                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
        cv2.imshow('frame', image)
        cv2.waitKey(2000)  # Chờ 2 giây trước khi bắt đầu action

        for sequence in range(no_sequences):
            # Hiển thị thông báo BẮT ĐẦU VIDEO
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            cv2.putText(image, f'STARTING VIDEO {sequence+1} for {action.upper()}', 
                        (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
            cv2.imshow('frame', image)
            cv2.waitKey(500)  # Chờ 2 giây

            # Loop qua từng frame
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                cv2.putText(image, f'{action.upper()} | Video {sequence+1}/{no_sequences} | Frame {frame_num+1}/{sequence_length}', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

                # Trích xuất và lưu keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Hiển thị ảnh
                cv2.imshow('frame', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
cap.release()
cv2.destroyAllWindows()