import cv2
from utils import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoint_values

if __name__ == '__main__':
    # read Feed
    cap = cv2.VideoCapture(0)
    # set mediapipe model
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            extraction = extract_keypoint_values(results)

            # show frame to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
