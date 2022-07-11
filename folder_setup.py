import os
from utils import actions, no_sequences, DATAPATH

# Generate folders for each video of each action (e.x. MP_Data/hello/1)
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATAPATH, action, str(sequence)))
        except:  # when already exists
            pass
