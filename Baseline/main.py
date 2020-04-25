from pre_processing_draft import *
from baseline import Baseline
import torch
import torch.nn as nn


TEST_PATH = '/home/ralleking/Downloads/clean_midi/ABBA/Honey Honey.mid'


#------ pre-processing ------#

#Take the song specified in TEST_PATH, converts to chunks, then trains the lstm
#using these chunks as input data. We can create more training data by looping
#through all songs and stacking them on the data tuple-list. Shuffling and
#splitting this list yields a training/test set when incorporating more songs.

#Note that the input dimensions to split_song defines our problem. I have
#arbitrarily selected 20 as the timesteps (e.g. how many columns/input steps
#do we base our prediction on) and 5 (number of time steps we try to predict)

midi = load_midi(TEST_PATH)
vectorized_song = notes_to_vectors(midi)
data = split_song(vectorized_song, 20, 5)

time_steps = data[0][0].shape[1]
pred_len = data[0][1].shape[1]

print("x-data has " + str(time_steps) + " columns.")
print("---------")
print("y-data has " + str(pred_len) + " columns.")
print("---------")


# ----- model definitions -------#
model = Baseline(time_steps, pred_len)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#------- Training -------#

#Right now, training is only done on one song, with 20 chunks. No training/test
#set defined yet. If everything is set up as intended, the network is trained
#over again with these 20 x,y pairs.
#It is not very tested, but loss does decrease when increasing epochs.

epochs = 50

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

bad_prediction = model(data[0][0])
print(bad_prediction)
