import os
import numpy as np

# path for exported data
DATAPATH = os.path.join('MP_Data')

# actions (moves) that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyour'])

# Thirty videos worth of data
no_sequences = 30

# videos are going to be 30 frames in length
sequence_length = 30

# Generate folders for each video of each action (e.x. MP_Data/hello/1)
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATAPATH, action, str(sequence)))
        except:  # when already exists
            pass
