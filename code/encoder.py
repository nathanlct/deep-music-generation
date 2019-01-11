import numpy as np

import music21 as m21

NUMBER_OF_NOTES = 128
TIME_DIVISION = 12

def add_note_to_matrix(matrix, pitch, offset, duration):
    # filling matrix
    matrix[pitch, offset] = 1
    for i in range(1, duration):
        matrix[-1, offset+i] = 1

def compute_matrix(measure, dim):
    matrix = np.zeros(dim, dtype=bool)
    for note in measure.recurse().notes:
        # extract data
        offset = note.offset * TIME_DIVISION
        int_offset = int(offset)
        duration = note.duration.quarterLength * TIME_DIVISION
        int_duration = int(duration)
        # check that offset and duration are actually ints
        if abs(offset-int_offset) > 0.01:
            print('Improper offset:', note.offset)

        if abs(duration - int_duration) > 0.01:
            print('Improper duration:', duration)

        if isinstance(note, m21.note.Note):
            pitch = note.pitch.midi
            add_note_to_matrix(matrix, pitch, int_offset, int_duration)

        elif isinstance(note, m21.chord.Chord):
            for pitch in note.pitches:
                add_note_to_matrix(matrix, pitch.midi, int_offset, int_duration)
        else:
            raise TypeError('Something which is not a chord/note has been detected')
    return matrix



def midi_to_matrix(path):
    '''
    Computes the array which corresponds to the midi file located in 'file'
    '''
    score = m21.converter.parse(path)
    length = score.highestTime * TIME_DIVISION
    int_length = int(length)
    if abs(length - int_length) > 0.01:
        print('Improper end')
    height = NUMBER_OF_NOTES + 1
    matrix_dict = {}
    voice_number = 1
    for voice in score.recurse().voices:
        # voice.show('text')
        # break
        # matrix_list = []
        # for measure in voice.getElementsByClass('Measure'):
        #     matrix = compute_measure(measure, (length, width))
        #     matrix_list.append(matrix)
        matrix = compute_matrix(voice, (height, int_length))
        matrix_dict['Voice %d' % voice_number] = matrix # TODO: add instruments
        voice_number += 1
    return matrix_dict


if __name__ == '__main__':
    matrix_dict = midi_to_matrix('bwv963.mid')
    print(matrix_dict)
