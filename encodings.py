'''
Scores have to be encoded into matrices in order to be read by the neural networks.
Different encodings exist and this file contains the function that convert scores
from one encoding to another. Encodings have specific numerical IDs.
0: default encoding
1: rests treated as notes, all-zero means the previous note is hold
2: rests treated as notes, long notes are treated as many short ones
'''

import numpy as np

import music21 as m21

MAX_ID = 2

def change_encoding(dict, old, new):
    if old > MAX_ID or new > MAX_ID:
        raise ValueError('Encoding %d does not exist: encodings have IDs between 0 and %d')

    if old == new:
        return dict

    new_dict = {}

    for l in dict:
        voice = dict[l]
        new_voice = []

        for matrix in voice:
            height, length = np.shape(matrix)
            new_matrix = np.copy(matrix)

            if old == 0:
                if new == 1: # 0 to 1 OK
                    new_matrix[-1,:] = 1 - matrix[-1,:]
                    for j in range(length):
                        for i in range(height-1):
                            if matrix[i,j]:
                                new_matrix[-1,j] = 0
                                break

                else: # 0 to 2 OK
                    for j in range(1, length):
                        if matrix[-1,j]: # there is a hold symbol
                            for i in range(height-1):
                                if new_matrix[i,j-1]:
                                    new_matrix[i,j] = 1

                    new_matrix = new_matrix[:-1,:]

            elif old == 1:
                if new == 0: # 1 to 0 OK
                    new_matrix[-1,:] = 1 - matrix[-1,:]
                    for j in range(length):
                        for i in range(height-1):
                            if matrix[i,j]:
                                new_matrix[-1,j] = 0
                                break

                else: # 1 to 2
                    for j in range(1, length):
                        is_note = False
                        for i in range(height-1):
                            if matrix[i,j]:
                                is_note = True
                                break
                        if not is_note:
                            for i in range(height-1):
                                if new_matrix[i,j-1]:
                                    new_matrix[i,j] = 1

                    new_matrix = new_matrix[:-1,:]

            else:
                new_matrix = np.zeros((height+1, length), dtype=int)
                new_matrix[:-1,:] = matrix
                if new == 0: # 2 to 0 OK
                    for j in range(1,length):
                        for i in range(height):
                            if matrix[i,j] and matrix[i,j-1]:
                                new_matrix[i,j] = 0
                                new_matrix[-1,j] = 1
                else:
                    if np.sum(matrix[:,0]) == 0:
                        new_matrix[-1,0] = 1
                    for j in range(1, length):
                        is_note = False
                        for i in range(height):
                            if matrix[i,j]:
                                is_note = True
                                if matrix[i,j-1]:
                                    new_matrix[i,j] = 0
                        if not is_note:
                            new_matrix[-1,j] = 1

            new_voice.append(new_matrix)

        new_dict[l] = new_voice

    return new_dict

def debug():
    M = np.array(
        [[0,1,0,0,0,1],
        [0,0,1,1,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,1,0]]
    )
    dict = {'a': [M]}

    print('dict:\n', dict['a'][0])
    dict01 = change_encoding(dict, 0, 1)
    print('dict01:\n', dict01['a'][0])
    dict02 = change_encoding(dict, 0, 2)
    print('dict02:\n', dict02['a'][0])
    dict10 = change_encoding(dict01, 1, 0)
    print('dict10:\n', dict10['a'][0])
    dict12 = change_encoding(dict01, 1, 2)
    print('dict12:\n', dict12['a'][0])
    dict20 = change_encoding(dict02, 2, 0)
    print('dict20:\n', dict20['a'][0])
    dict21 = change_encoding(dict02, 2, 1)
    print('dict21:\n', dict21['a'][0])




if __name__ == '__main__':
    debug()
