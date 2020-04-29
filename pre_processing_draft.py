import mido
import os
import numpy as np

example_midi = 'nesmdb_midi/test/002_1943_TheBattleofMidway_00_01Title.mid'

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
    tempo = (60000000/bpm)
    return tempo

def tempo_to_bpm(tempo): #Converts tempo in microseconds/beat to beats/minute
    bpm = 60000000/tempo
    return bpm

def ticks_to_seconds(ticks,tempo,tpb):
    return tempo/1000000*ticks/tpb
    
print()

def quantize_track(track ,ticks_per): #ticks_per denotes quantization base. 12 for 32nds, 24 for 16ths and 48 for 8ths
    for msg in track:
        msg.time = round(msg.time/ticks_per)*ticks_per
    return track


def print_verbose(mid):
    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            print(msg)
            
def print_track_verbose(midi_track):
    for msg in midi_track:
        print(msg)
        
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
         
def group_by_time(track, group_size, ignore_tail = True, force_even = False): #Group note messages by time
    grouped_messages = []
    current_group = []
    time_passed = 0
    for i,msg in enumerate(track):
        if msg.type == 'note_on':
            
            if time_passed+msg.time > group_size: #If note exceeds time-frame
                surplus_time = time_passed+msg.time-group_size
                msg.time = group_size-time_passed
                current_group.append(msg)
                grouped_messages.append(current_group)
                current_group = [mido.Message('note_on', note=0, velocity = 0, channel=msg.channel, time = surplus_time)]
                time_passed = surplus_time
            else:
                time_passed += msg.time
                current_group.append(msg)
                
        if not ignore_tail: #Keep/drop last group if not full
            if i == len(track): #Check if end of track is reached, and if it is appends the current group to the total group 
                grouped_messages.append(current_group)
                
    if force_even:
        if not len(grouped_messages) % 2 == 0:
            del grouped_messages[-1]

    return grouped_messages

def group_by_nr_notes(track, group_size,ignore_tail = True, force_even = False): #Group messages by number of notes
    current_group = []
    grouped_messages = []
    current_size = 0
    for i, msg in enumerate(track):
        
        if msg.type == 'note_on':
            if not current_size == group_size: #Append msg to group
                current_group.append(msg)
                current_size += 1
            else: #If end of group is reached, start new group
                grouped_messages.append(current_group)
                current_group = [msg]
                current_size = 1
                
        if not ignore_tail:#Keep/drop last group if not full
            if i == len(track)-1: #Check if end of track is reached, and if it is appends the current group to the total group 
                grouped_messages.append(current_group)
                
    if force_even:
        if not len(grouped_messages) % 2 == 0:
            del grouped_messages[-1]
    return grouped_messages

def NES_midi_to_txt(root, group_size, group_type = 1, ignore_tail = True, force_even = False, expression = True):
    for split in os.listdir(root):
        path = root + '/' + split
        with open(split + '.txt',"w") as f:
            for mid in os.listdir(path):
                midi_file = load_midi(path+'/'+ mid)
                for i,track in enumerate(midi_file.tracks):
                    if group_type == 1:
                        grouped_messages = group_by_time(track, group_size, ignore_tail, force_even)
                    elif group_type == 2:
                        grouped_messages = group_by_nr_notes(track, group_size, ignore_tail, force_even)
                    else:
                        print('Error: Invalid group type! Must be either 1 or 2')
                    for group in grouped_messages:  
                        for j,msg in enumerate(group):
                            if expression:
                                f.write(str(i) + '_' + str(msg.note) + '_' + str(msg.velocity) + '_' + str(msg.time))
                            else:
                                f.write(str(i) + '_' + str(msg.note) + '_' + str(msg.time))
                            if not j == len(group)-1:
                                f.write(',')
                            else:
                                f.write(';')
                f.write('\n')
                    
