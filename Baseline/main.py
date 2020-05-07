from old_preprocess import *
from tx1_pre_process import *
from baseline import Baseline
import random
import torch
import os
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE_DIR = os.getcwd() + '/Baseline'
DATA_PATH = BASELINE_DIR  + '/TEST_TXT_SONGS/'
VOCAB_PATH = BASELINE_DIR  + '/resources/vocab.txt'

TRAIN_PATH = '/home/ralleking/Downloads/nesmdb_tx1/train/'
VAL_PATH = '/home/ralleking/Downloads/nesmdb_tx1/val/'
TEST_PATH = '/home/ralleking/Downloads/nesmdb_tx1/test/'

#------ pre-processing ------#

#Take the song specified in TEST_PATH, converts to chunks, then trains the lstm
#using these chunks as input data.

vocab, vocab_vectors = import_vocab('/home/ralleking/Code/Python/Project/dev/DeepDIVA/Baseline/resources/vocab.txt')

txt_songs = get_midi_paths(TEST_PATH)
vectorized_songs = []
training_data = []
data = []

#vectorize txt song
for txt_song in txt_songs:
    vectorized_song = tx1_to_vectors(txt_song, vocab, vocab_vectors)
    vectorized_songs.append(vectorized_song)

#create training examples
for song in vectorized_songs:
    if len(song) >= 30:
        song_sample = sample_song(song)
        data.append(song_sample)

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
model = Baseline(time_steps, 12, num_classes)

#loss and things
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)

#------- Training -------#
random.shuffle(data)

epochs = 10

for i in range(epochs):
    for x, y in data:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        y_pred = y_pred.to(device)
        single_loss = loss_function(y_pred, y)
        single_loss.backward()
        optimizer.step()
    print(single_loss)


#------- Prediction ------------#
seq = predict_sequence('Baseline/TEST_TXT_SONGS/013_AlterEgo_00_01Title.tx1.txt', vocab, vocab_vectors, model)
vectors_to_tx1('out.txt', seq, vocab, vocab_vectors)
