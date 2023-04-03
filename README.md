# Real-time Sign Language Detection
This project is written in Python and designed to detect sign language 
gestures in real-time through a webcam video stream. It uses a pre-trained 
keras model to identify three gestures: "hello", "I love you", and "thanks". 
The detected gestures are displayed on the screen, allowing the user to build 
sentences with them.

## Packages
The program uses the following packages:

* **OpenCV**: OpenCV is a popular computer vision library that is used for real-time computer vision applications. It is used in this project for webcam feed and information display.

* **MediaPipe**: MediaPipe is an open-source framework that provides cross-platform, customizable ML solutions for live and streaming media. It is used in this project for gesture keypoints detection.

* **Keras**: Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It is used in this project for machine learning.

## Usage
There are four main scripts in the program:

* `make_predictions.py`: Predicts the trained gestures and builds sentences.
* `folder_setup.py`: Builds the folder structure.
* `training_testing.py`: Trains the model with the "MP_Data" Keypoints and evaluates it.
* `setup_training_testing_keypoints.py`: Starts a training program where the user can input the gestures for the actions. The keypoints are then saved and can be used for training and evaluation.

## Getting started
To get started with the program, follow these steps:

1. Install the required packages using the following command:<br>
```pip install opencv-python mediapipe keras```

2. Clone the repository and navigate to the project directory.<br>
`git clone https://github.com/Marc-Kruiss/SignLanguage-ActionDetection`<br>
```cd SignLanguage-ActionDetection```

3. Run the `folder_setup` script to build the required folder structure.<br>
```python folder_setup.py```

4. Run the `setup_training_testing_keypoints` script to input the gestures for the actions and save the keypoints.<br>
```python setup_training_testing_keypoints.py```

5. Train the model using the saved keypoints by running the `training_testing.py` script.<br>
```python training_testing.py```

6. Use the `make_predictions` script to detect the gestures in real-time and build sentences.<br>
```python make_predictions.py```

## References
MediaPipe: https://mediapipe.dev/ <br>
Keras: https://keras.io/ <br>
OpenCV: https://opencv.org/ <br>

## Acknowledgements
This project was inspired by the work of Ahmed Hassanien and his article "Real-Time American Sign Language Recognition using Deep Learning Neural Networks"