""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    pitch_occurrences = dict()
    for file in glob.glob("minecraft_5_songs/*.mid"):
        midi = converter.parse(file)
        
        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            print("s2: ", s2)
            print(len(s2.parts[0]))
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                # print("element pitch: ", str(element.pitch))
                if(str(element.pitch) not in pitch_occurrences):
                    pitch_occurrences[str(element.pitch)] = 1
                else:
                    pitch_occurrences[str(element.pitch) ] += 1
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                s = '.'.join(str(n) for n in element.normalOrder)
                # print("chord: ", s)
                notes.append('.'.join(str(n) for n in element.normalOrder))
            else: 
                try: 
                    print("element neither chord or note: ", element)
                except:
                    print("yike")
                    pass
        

    

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    c = 0
    for key in pitch_occurrences:
        # print("label: ", key)
        # print("plotting at index c: ", c)
        # print("num occurrences: ", pitch_occurrences[key])
        if(pitch_occurrences[key] > 20):
            plt.bar(c, int(pitch_occurrences[key]) , width = 0.9, label = str(key), color = numpy.random.rand(3,))
        # plt.subplots_adjust(bottom= 0.2, top = 0.98)
        
        c += 1
    fontP = FontProperties()
    fontP.set_size('small')
    # plt.legend([plot1], "title", prop=fontP)
    plt.legend(prop=fontP)
    plt.show()

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
