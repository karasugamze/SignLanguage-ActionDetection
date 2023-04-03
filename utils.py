import os

import cv2
import mediapipe as mp
import numpy as np

# actions (moves) that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Thirty videos worth of data
no_sequences = 30

# videos are going to be 30 frames in length
sequence_length = 30

# path for exported keyframe Dada (.npy)
DATAPATH = os.path.join('MP_Data')

mp_holistic = mp.solutions.holistic  # holistic model for detection
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities for drawing detection


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Detection using mediapipe
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color conversion RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)  # draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # draw left hand Connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # draw right hand connections


def draw_styled_landmarks(image, results):
    # draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              # color landmarks
                              mp_drawing.DrawingSpec(color=(80, 110, 121), thickness=1, circle_radius=1)
                              # color connections
                              )  # draw
    # draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=4),
                              # color landmarks
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=2)
                              # color connections
                              )  # draw
    # draw left hand Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 76), thickness=2, circle_radius=4),
                              # color landmarks
                              mp_drawing.DrawingSpec(color=(80, 44, 250), thickness=2, circle_radius=2)
                              # color connections
                              )  # draw
    # draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 117, 66), thickness=2, circle_radius=4),
                              # color landmarks
                              mp_drawing.DrawingSpec(color=(80, 66, 230), thickness=2, circle_radius=2)
                              # color connections
                              )  # draw


def extract_keypoint_values(results):
    # if pose is in picture, extract the values, else set empty array (pose shape is (132,))
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)  # (33 landmarks * x/y/z/visibility)

    # if face is in picture, extract the values, else set empty array (face shape is (1404,))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)  # (468 Landmarks * x/y/z)

    # if left hand is in picture, extract the values, else set empty array (hand shape is (63,))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)  # (21 Landmarks * x/y/z)

    # if right hand is in picture, extract the values, else set empty array (hand shape is (63,))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)  # (21 Landmarks * x/y/z)

    return np.concatenate([pose, face, lh, rh])
