import numpy as np

import music21 as m21

from encoder import file_to_dictionary, dictionary_to_midi, NUMBER_OF_PITCHES

NUMBER_OF_PITCH_CLASSES = 12
PITCHES_WEIGHTS = np.array([0,16,1,9,1,1,16,1,16,0,4,4])

def pitches_histogram(dict):
    '''
    compute the table hist of the score encoded by [dict], where hist[i] denotes
    the number of notes whose pitch class is i in the score
    '''
    pitch_classes = np.zeros(NUMBER_OF_PITCH_CLASSES)
    pitches = np.zeros(NUMBER_OF_PITCHES)

    for l in dict:
        voice = dict[l]
        height, length = np.shape(voice[0])
        for matrix in voice:
            pitches += np.sum(matrix[:-1,:], axis=1)

    for i in range(NUMBER_OF_PITCHES):
        pitch_classes[i%NUMBER_OF_PITCH_CLASSES] += pitches[i]

    return pitch_classes

def best_tonality(hist, weights=PITCHES_WEIGHTS, print_results=False):
    '''
    find statistically what is the tonality of the score whose histogram is [hist]
    '''
    c = NUMBER_OF_PITCH_CLASSES
    extended_hist = np.zeros(2*c)
    extended_hist[:c] = hist
    extended_hist[c:] = hist
    scores = np.zeros(c)

    for p in range(c):
        scores[p] = np.sum(extended_hist[p:c+p] * weights)

    if print_results:
        for p in range(c):
            pitch = m21.pitch.Pitch(p)
            print('%s: %d' % (pitch.name, scores[p]))

    return np.argmin(scores)

def transpose(dict, tonality):
    '''
    transpose the score encoded by [dict] to C major/A minor
    CAUTION: doesn't work for encoding number 3!
    '''
    new_dict = {}
    for l in dict:
        voice = dict[l]
        new_voice = []
        for matrix in voice:
            new_matrix = np.zeros(np.shape(matrix))
            new_matrix[-1] = matrix[-1]
            if tonality < 6:
                new_matrix[:-1-tonality,:] = matrix[tonality:-1,:]
            else:
                new_matrix[tonality:-1,:] = matrix[:-1-tonality,:]
            new_voice.append(new_matrix)
        new_dict[l] = new_voice
    return new_dict

def main(file):
    score = m21.converter.parse(file)
    dict = file_to_dictionary(score)
    hist = pitches_histogram(dict)
    t = best_tonality(hist, print_results=True)
    transposed = transpose(dict, t)
    hist2 = pitches_histogram(transposed)
    t2 = best_tonality(hist2, print_results=True)
    score.show('midi')
    print(len(transposed["Voice 1"]))
    print(transposed["Voice 1"][0].shape)
    score_rec = dictionary_to_midi(transposed)
    score_rec.show('midi')

if __name__ == '__main__':
    main('data/Brahms/4.mid')
