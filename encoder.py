import numpy as np

import music21 as m21

NUMBER_OF_PITCHES = 128
TIME_DIVISION = 12

def add_note_to_matrix(matrix, pitch, offset, duration):
    '''
    add the note ([pitch], [offset], [duration]) to the coding [matrix]
    '''
    matrix[pitch, offset] = 1
    for i in range(1, duration):
        matrix[-1, offset+i] = 1


def pattern_to_matrix(pattern, dim):
    '''
    compute the matrix which represents the code of [pattern]
    '''
    matrix = np.zeros(dim, dtype=bool)
    print('voice')
    pattern.flat.notes.show('text')
    for note in pattern.flat.notes:
        # extract data
        offset = int(note.offset * TIME_DIVISION)
        duration = int(note.duration.quarterLength * TIME_DIVISION)
        print(note.pitches, note.offset, note.duration.quarterLength)
        for pitch in note.pitches:
            add_note_to_matrix(matrix, pitch.midi, offset, duration)
    # print('debug')
    # for i in range(128):
    #     for j in range(108):
    #         if M[i,j] != matrix[i,j]:
    #             print(i,j)
    print(matrix[-1])
    return matrix

def separate_matrix(matrix, time_signature):
    '''
    divides the [matrix] into a list of matrices, each representing a measure
    '''
    matrix_list = []
    measure_size = TIME_DIVISION * time_signature
    last_measure = np.shape(matrix)[1]//measure_size
    for i in range(last_measure):
        matrix_list.append(matrix[:,measure_size*i:measure_size*(i+1)])
    return matrix_list

def file_to_dictionary(file):
    '''
    compute the dictionary encoding the file
    '''
    score = m21.converter.parse(file) if not isinstance(file, m21.stream.Stream) else file
    time_signature = 3 # TODO:
    length = int(score.highestTime * TIME_DIVISION)
    height = NUMBER_OF_PITCHES + 1
    matrix_dict = {}
    voice_number = 1
    for voice in score.recurse().getElementsByClass(m21.stream.Part):
        matrix = pattern_to_matrix(voice, (height, length))
        matrix_list = separate_matrix(matrix, time_signature)
        matrix_dict['Voice %d' % voice_number] = matrix_list # TODO: add instruments
        voice_number += 1
    return matrix_dict

def dictionary_to_midi(matrix_dict):
    '''
    decode the dictionary encoding a pattern and return the pattern as a score
    '''
    score = m21.stream.Score()
    for l in matrix_dict:
        voice = m21.stream.Part()
        for matrix in matrix_dict[l]:
            height, length = np.shape(matrix)
            rest = 0
            for j in range(length):
                chord_pitches = []
                for i in range(height-1):
                    if matrix[i,j]:
                        if rest > 0:
                            voice.append(m21.note.Rest(quarterLength=rest/TIME_DIVISION))
                            rest = 0
                        if chord_pitches == []:
                            offset = j/TIME_DIVISION
                            qL = 1
                            d = j+1
                            while d < length and matrix[-1,d]:
                                qL += 1
                                d += 1
                        chord_pitches.append(i)
                if len(chord_pitches) == 1: # note
                    voice.append(m21.note.Note(chord_pitches[0], offset=offset, quarterLength=qL/TIME_DIVISION))
                elif len(chord_pitches) > 1: # chord
                    voice.append(m21.chord.Chord(chord_pitches, offset=offset, quarterLength=qL/TIME_DIVISION))
                else: # rest
                    rest += 1
        score.append(voice)
    return score

def print_mat(mat):
    n, d = np.shape(mat)
    for i in range(n):
        for j in range(d):
            v = 1 if mat[i,j] else 0
            print(v, end=',')
        print('')
    print('\n')


def debug_test():
    score = m21.converter.parse('sonata05-1.krn').measures(1,3)
    dict = file_to_dictionary(score)
    # print(len(dict), len(dict['Voice 1']), np.shape(dict['Voice 1'][0]))
    # print(dict['Voice 2'][0][:,0])
    score.flat.notes.show('midi')
    score_rec = dictionary_to_midi(dict)
    score_rec.show('midi')

# TODO: time signatures, instrument, keys

if __name__ == '__main__':
    debug_test()
