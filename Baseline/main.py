from pre_processing_draft import *
from baseline import Baseline
import random
import torch
import torch.nn as nn


MIDI_DIR = './TEST_OK/'

#------ pre-processing ------#

#Take the song specified in TEST_PATH, converts to chunks, then trains the lstm
#using these chunks as input data.

midis = get_midi_paths(MIDI_DIR)
data = []

for raw_midi in midis:
    print(raw_midi)
    midi = load_midi(raw_midi)
    vectorized_song = notes_to_vectors(midi)
    if vectorized_song is not None:
        song_data = split_song(vectorized_song, 5, 1)
        data += song_data
        print("Preprocess ok for {0}".format(raw_midi))
    else:
        print("Import failed")
    print(" ")

time_steps = data[0][0].shape[1]
pred_len = data[0][1].shape[0]

print("x-data has " + str(time_steps) + " columns.")
print("---------")
print("y-data has " + str(pred_len) + " rows.")
print("---------")
print("In total {0} training examples".format(str(len(data))))

# ----- model definitions -------#
model = Baseline(time_steps, pred_len)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#------- Training -------#

#Right now, training is only done on one song, with 20 chunks. No training/test
#set defined yet. If everything is set up as intended, the network is trained
#over again with these 20 x,y pairs.
#It is not very tested, but loss does decrease when increasing epochs.

random.shuffle(data)

epochs = 5

for i in range(epochs):
    for x, y in data:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.prediction_length), torch.zeros(1, 1, model.prediction_length))

        y_pred = model(x)

        single_loss = loss_function(y_pred, y)
        single_loss.backward()
        optimizer.step()
    print(single_loss)


#----- Predictions ------#

#Maing predictions should be as easy as running below. It is going to be a bad
#prediction, but still, seems to be on the correct format.
#
# bad_prediction = model(data[0][0])
# print(bad_prediction)
