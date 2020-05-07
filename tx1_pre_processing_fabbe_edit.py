import numpy as np
import os
import torch
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

import random    




def random_XY_pair_from_song(filename, A , vocab, vocab_vectors):
    '''
    file: any file in the directory 'nesmdb_tx1/train/' , example: files = os.listdir('nesmdb_tx1/train/')
    A : length of X , eg A = 20 ==> size(X) = (20,630)             filename = files[0]
    '''
    events = tx1_to_vectors('nesmdb_tx1/train/'+ filename ,vocab,vocab_vectors) # convert txt to file 'duh!!'
    if len(events) > A: # check that current file is appropriate size 
        R = random.randint(0,len(events)-A-1) # chooses a random point to start extracting dat
        X = []
        for i in range(0,A):# append the X vectors starting from the random number
            X.append(events[R+i]) # highest index here will be R+19 
        Y = events[R+A] # (index is R +20) Y vector is the vector coming after the last vector in X , this will be the label
        data_pair = (X,Y) # tuple of the data pair
        return data_pair #return the tuple
    else:
        return



def generate_trainset_data():
    files = os.listdir('nesmdb_tx1/train/')
    DATA = []
    for file in files:
        data_pair = random_XY_pair_from_song(file, 20 ,vocab,vocab_vectors) # load data pair
        DATA.append(data_pair) # append each pair to a new list
    return DATA

def split_DATA(DATA):
    X = []
    Y = []
    for i in range(0,len(DATA)):
        x,y = DATA[i] #append the X and Y to separate list
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    X_tensor = torch.from_numpy(X)
    Y_tensor = torch.from_numpy(Y)
    return X_tensor,Y_tensor



vocab,vocab_vectors = import_vocab('nesmdb_tx1/vocab.txt') # import vocabulary 

DATA = generate_trainset_data()# WARNING: THIS CONSUMES TIME!

print(len(DATA),len(DATA[-1])) 
X_tensor,Y_tensor = split_DATA(DATA) # this dont work :(


        
