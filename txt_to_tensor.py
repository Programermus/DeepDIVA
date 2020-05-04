import os
import numpy as np
import torch
def isolate_group(group_num):
    group_vector=[]
    i = 1 
    with open('midi_data.txt', 'r') as data: # append each character in the file to a list
        for st in data.readline():
            group_vector.append(st)
            if st ==';':
                A = group_vector
                group_vector = []
                if i == group_num: # checks wich group we wish to extract  
                    break
                i += 1

        group_str = "".join(A) #join the whole list into  one big string
        
    group_ls = group_str.split(',')   # separate the string on the  ',' character   
  
    return group_ls # return a list of strings eg. group_ls = [[1_0_40_50],[1_0_75_80],... ] 



def group_to_tensor(group):
    # separate the _ char from the list
    GROUP =[]
    for msg in group:
        GROUP.append(msg.split('_'))
    #_------------------------------

    # removing the ; char from the last list entry, (code is not pretty but it does the job)
    A = GROUP[-1][-1].split(';') 
    A.remove('') #
    A = "".join(A)#
    GROUP[-1][-1] = A #
    #-------------------------

    # convert the list of strings into list of integers
    new_GROUP =[] 
    for msg in GROUP:
        for i in msg:
            new_GROUP.append(int(i))
    #---------------------------
    #turn list of integers into numpy array
    data = np.array(new_GROUP)
    #------------------
    #turn numpy array into pytorch tensor
    torch_data = torch.from_numpy(data)
    #-------------
    return torch_data



group=isolate_group(4)
data = group_to_tensor(group)
print(data)