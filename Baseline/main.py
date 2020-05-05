from old_preprocess import *
from tx1_pre_process import *
from baseline import Baseline
import random
import torch
import os
import torch.nn as nn

BASELINE_DIR = os.getcwd() + '/Baseline'
DATA_PATH = BASELINE_DIR  + '/TEST_TXT_SONGS/'
VOCAB_PATH = BASELINE_DIR  + '/resources/vocab.txt'

#------ pre-processing ------#

#Take the song specified in TEST_PATH, converts to chunks, then trains the lstm
#using these chunks as input data.

vocab, vocab_vectors = import_vocab('/home/ralleking/Code/Python/Project/dev/DeepDIVA/Baseline/resources/vocab.txt')

txt_songs = get_midi_paths(DATA_PATH)
vectorized_songs = []
training_data = []
data = []

#vectorize txt song
for txt_song in txt_songs:
    vectorized_song = tx1_to_vectors(txt_song, vocab, vocab_vectors)
    vectorized_songs.append(vectorized_song)

#create training examples
for song in vectorized_songs:
    data.append(sample_song(song))

#number of cols in x and y
time_steps = data[0][0].shape[1]
pred_len = 1
num_classes = data[1][0].shape[0]

print("---------")
print("x-data has " + str(time_steps) + " columns.")
print("---------")
print("y-data has " + str(pred_len) + " columns.")
print("---------")
print("In total {0} training examples".format(str(len(data))))
print("---------")
print("There are also " + str(num_classes) + " classes")
print("---------")

# ----- model definitions -------#
model = Baseline(time_steps, 64, num_classes)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)

#------- Training -------#

random.shuffle(data)

epochs = 10

for i in range(epochs):
    for x, y in data:
        optimizer.zero_grad()
        y_pred = model(x)
        single_loss = loss_function(y_pred, y)
        single_loss.backward()
        optimizer.step()
    print(single_loss)

#----- Predictions ------#
