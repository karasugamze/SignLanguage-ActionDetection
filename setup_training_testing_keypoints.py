import os

import cv2
import mediapipe as mp

# actions (moves) that we try to detect
import numpy as np

import utils
from utils import actions, mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoint_values, DATAPATH

# read Feed
cap = cv2.VideoCapture(0)

# set mediapipe model
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as holistic:
    # loop through sequences of actions
    for action in actions:
        # loop through the sequences (= videos)
        for sequence in range(utils.no_sequences):
            # loop through video (sequence) langth
            for frame_num in range(utils.sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Apply wait Logic
                if frame_num == 0:
                    cv2.putText(image, "Starting Collection", (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, "Collecting frames for {} Video Number {}".format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # show frame to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1500)
                else:
                    cv2.putText(image, "Collecting frames for {} Video Number {}".format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # show frame to screen
                    cv2.imshow('OpenCV Feed', image)

                # Export captured frame-keypoints
                keypoints = extract_keypoint_values(results)
                npy_path = os.path.join(DATAPATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
