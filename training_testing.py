import os

import numpy as np
from sklearn.model_selection import train_test_split

from keras import Sequential
from keras.utils import to_categorical
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from folder_setup import DATAPATH
from utils import actions, no_sequences, sequence_length

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


def get_train_test_data():
    # {'hello':1, 'thanks':2, ...}
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            # stores all frames of the sequence
            window = []
            for frame_num in range(sequence_length):
                # load npy file data
                res = np.load(os.path.join(DATAPATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    return X_train, X_test, y_train, y_test


def setup_model():
    model = Sequential()  # easy to build up a model
    # add LSTM layers
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    # add dense layers (fully connected layers)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))  # softmax => 0-1

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def train_model(model, X_train, y_train):
    # setup logging
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model.fit(X_train, y_train, epochs=3000, callbacks=[tb_callback])


def saveModel(model):
    model.save('action.h5')


def evaluateModel(model, X_test, y_test):
    model.load_weights('action.h5')
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    multilabel_confusion_matrix(ytrue, yhat)
    print(accuracy_score(ytrue, yhat))
    model.summary()


if __name__ == '__main__':
    retrain = False
    X_train, X_test, y_train, y_test = get_train_test_data()
    model = setup_model()
    if retrain:
        train_model(model, X_train, y_train)
        saveModel(model)

    evaluateModel(model, X_test, y_test)
