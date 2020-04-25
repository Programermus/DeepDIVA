import mido
import os
import torch
import numpy as np

def list_files(startpath): #Prototype function for listing the directory
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def load_midi(path): #Load midi file
    mid = mido.MidiFile(path)
    return mid

def bpm_to_tempo(bpm): #Converts tempo in beats/minute to microseconds/beat
    return (60000000/bpm)

def tempo_to_bpm(tempo): #Converts tempo in microseconds/beat to beats/minute
    return 60000000/tempo

def ticks_to_seconds(ticks,tempo,tpb):
    return (tempo/1000000)*ticks/tpb

def quantize(midi ,ticks_per): #ticks_per denotes quantization base. 12 for 32nds, 24 for 16ths and 48 for 8ths
    for track in midi.tracks:
        for msg in track:
            msg.time = round(msg.time/ticks_per)*ticks_per
    return midi

def generate_pitch_classes(): #Returns a list where the first index (0-11) corresponds to each musical note starting from A0
    notes = [[],[],[],[],[],[],[],[],[],[],[],[]]
    A0 = 21
    for i in range(8):
        for j in range(12):
            note = A0 +i*12 + j
            if note <= 108:
                notes[j].append(note)
    return notes

def note_to_pitchclass(midi_note,pitch_classes): #Returns a vector with a 1 for the entered note
    pitch_class = np.zeros((12,1))
    for i in range(12):
        if midi_note in pitch_classes[i]:
            pitch_class[i][0] = 1
            return pitch_class

def notes_to_vectors(mid):  #Returns vector set of played notes in the format [note_played; note_velocity; delta_t; pitch_class]
    time_passed = 0
    i = 0
    pitch_classes = generate_pitch_classes()
    for msg in mid.tracks[1]:
        time_passed += msg.time
        if msg.type == 'note_on':
            pitch_class = note_to_pitchclass(msg.note,pitch_classes)
            vector = np.vstack((msg.note,msg.velocity, msg.time,pitch_class))
            if i == 0:
                input_seq = vector
            else:
                input_seq = np.hstack((input_seq,vector))
            i += 1
    return input_seq

def split_song(vectorized_song, time_steps=20, prediction_length=5):
    #Should return a list of tuple where the tuple elements are pytorch tensors.
    #Each song is chunked up into time_steps and prediction_length, where
    #time_steps is the predictors (default to a matrix of size 15,20) and
    #prediction_length specifies how many note-steps we try to predict.

    data = []

    #pad song to be splittable into chunks of size x_data.shape = (time_steps, rows)
    #and y_data.shape=(prediction_length, rows).
    rows, cols = vectorized_song.shape[0], vectorized_song.shape[1]
    divisor = 0
    while (cols+divisor) % (time_steps + prediction_length) != 0:
        divisor += 1
    padding = np.zeros(shape=(rows, divisor))
    padded_sequence = np.hstack((vectorized_song, padding))

    #split song and store in list of tuples (where x and y are torch tensors)
    chunk_size = time_steps+prediction_length
    for i in range(0, padded_sequence.shape[1], chunk_size):
        song_chunk = padded_sequence[:, i:i+chunk_size]
        x = song_chunk[:,0:time_steps]
        y = song_chunk[:,time_steps:]
        x_data = torch.from_numpy(x).float()
        y_data = torch.from_numpy(y).float()
        data.append((x_data, y_data))
    return data
