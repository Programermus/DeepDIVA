import numpy as np
import random
import torch
import os

def quantize_wait(wait):
    wait = min(wait, 100000)

    if wait > 10000:
        wait = 1000 * int(round(float(wait) / 1000) + 1e-4)
    elif wait > 1000:
        wait = 100 * int(round(float(wait) / 100) + 1e-4)
    elif wait > 100:
        wait = 10 * int(round(float(wait) / 10) + 1e-4)
    return wait

def import_vocab(vocab_txt):
    vocab = []
    with open(vocab_txt, 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))
        f.close()
    return vocab, np.eye(len(vocab))

def word_to_idx(word,vocab):
    idx = vocab.index(word)
    return idx

def idx_to_word(idx,vocab):
    word = vocab[idx]
    return word

def tx1_to_vectors(file, vocab, vocab_vectors):
    events = []
    with open(file, 'r') as f:
        for event in f:
            event = event.strip('\n')
            split_event = event.split('_')
            if split_event[0] == 'WT':
                split_event[1] = str(quantize_wait(int(split_event[1])))
                event = split_event[0] + '_' + split_event[1]
            idx = word_to_idx(event,vocab)
            vector = vocab_vectors[:, idx]
            events.append(vector)
    return events

def vectors_to_tx1(filename, vector_set,vocab,vocab_vectors):
    with open(filename,'w') as f:
        for vector in vector_set:
            idx = np.argwhere(vector==1)[0][0]
            word = idx_to_word(idx,vocab)
            f.write(word + '\n')
        f.close()

#Takes n random samples of specified chunk size from a song.
#Returns all samples as a list of pytorch tensors [X, Y]. This list
#we then concatenate to create a sample data set of the same form.
#---
#Sample size is a tuple describing length of input data (number of time_steps)
#and how many notes in adavance we want to predict, default = (5,1)
def sample_song(vectorized_song, n=1, sample_size=(5,1)):

    #draw a random segment from list, dependent on sample size
    sample_end = random.randint(sample_size[0] + sample_size[1], len(vectorized_song))
    sample_start = sample_end-sample_size[0]-1
    sample = vectorized_song[sample_start:(sample_end)]

    #split into x and y
    x, y = sample[:sample_size[0]], sample[sample_size[0]:]

    #collapse lists
    x, y = np.stack(x, axis=1), np.stack(y, axis=1)

    #torch want class index as input, not one-hot
    y = np.array([np.argmax(y)])

    #make into pytorch tensors
    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()

    #enable cuda if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x.to(device)
    y.to(device)

    return x, y

#----- Predictions ------#
def predict_sequence(song_txt, vocab, vocab_vectors, model, prediction_length=30, prediction_start=10):
    vectorized_song = tx1_to_vectors(song_txt, vocab, vocab_vectors)
    song_start = np.stack(vectorized_song[prediction_start-5:prediction_start], axis=1)
    x = torch.from_numpy(song_start).float()
    predicted_sequence = vectorized_song[0:prediction_start]
    for k in range(1, prediction_length+prediction_start):
        y = model(x)
        y = y.view(-1)
        y = np.asarray(y.data)
        y[np.where(y==np.max(y))] = 1
        y[np.where(y!=1)] = 0
        predicted_sequence.append(y)
        y = torch.from_numpy(y).float()
        y = y.view(1,len(y))
        x = torch.cat((x,y.t()), 1)
        x = x[:,-5:]
    return predicted_sequence
