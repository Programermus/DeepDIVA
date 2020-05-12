import os
import mido
import numpy as np

def predict_next_note(data, model):
    #prediction
    pred = model(data)
    
    #remove first entry in data
    data = np.delete(data, 0, axis=1)
    
    #concatenate prediction to data
    data = np.hstack(data, pred)
    
    return data



def note_to_pitchclass(midi_note,pitch_classes): #Returns a vector with a 1 for the entered note
    pitch_class = [0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(12):
        if midi_note in pitch_classes[i]:
            pitch_class[i] = 1
    return pitch_class

def generate_pitch_classes(): #Returns a list where the first index (0-11) corresponds to each musical note starting from A0
    notes = [[],[],[],[],[],[],[],[],[],[],[],[]]
    A0 = 21
    for i in range(8):
        for j in range(12):
            note = A0 +i*12 + j
            if note <= 108:
                notes[j].append(note)
    return notes

def read_txt_files(data_folder):
    folds = []
    pitch_classes = generate_pitch_classes()
    for file in sorted(os.listdir(data_folder)):
        if not file.split('.')[1] == 'txt': #Read only txt files
            continue
        with open(data_folder + '/' + file,'r') as f: #Read groups one at a time
            groups = []
            for line in f:
                group = []
                for event in line.split(','):
                    event = event.split('_')
                    note = event[0]
                    vel = event[1]
                    time = event[2]
                    pitch_class = note_to_pitchclass(int(note),pitch_classes)
                    attr = [int(note), int(vel), int(time)]
                    attr.extend(pitch_class)
                    group.append(attr)
                groups.append(group)
        folds.append(groups)
    return folds

def array_to_midi(data,root,midi_name):
    os.chdir(root)
    if not os.path.exists('Output'): #Create output directory
        os.makedirs('Output')
    os.chdir(root + '/Output')

    mid = mido.MidiFile()
    mid.ticks_per_beat = 384
    track0 = mido.MidiTrack()
    track1 = mido.MidiTrack()

    track0.append(mido.MetaMessage('set_tempo', tempo = 500000, time = 0)) # Meta information like tempo and time signature assigned to track 0
    track0.append(mido.MetaMessage('time_signature', numerator = 4, denominator = 4, clocks_per_click = 24, notated_32nd_notes_per_beat = 8, time = 0))
    track0.append(mido.MetaMessage('end_of_track', time = 1))
    mid.tracks.append(track0)
    track1.append(mido.Message('program_change', channel = 0, program = 0, time = 0))

    for group in data: #Write notes to track 1
        for i, row in enumerate(group):
            note = row[0]
            vel = row[1]
            time = row[2]
            track1.append(mido.Message('note_on', channel = 0, note = note, velocity = vel, time = time))
    track1.append(mido.MetaMessage('end_of_track', time = 1))

    mid.tracks.append(track1)

    mid.save(midi_name)
    os.chdir(root)

