import pickle
import numpy
import glob
from music21 import converter, instrument, note, chord
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# f = open("minecraft_5_songs/test_output.mid", "r")
notes = []
midi = None
for file in glob.glob("midi_songs/*.mid"):
    if midi is None:
        midi = converter.parseFile(file)
    else:
        midi += converter.parseFile(file)


pitch_occurrences = dict()

notes_to_parse = None

try: # file has instrument parts
    s2 = instrument.partitionByInstrument(midi)
    # print("s2: ", s2)
    # print(len(s2.parts[0]))
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

c = 0
for key in pitch_occurrences:
    # print("label: ", key)
    # print("plotting at index c: ", c)
    # print("num occurrences: ", pitch_occurrences[key])
    if(pitch_occurrences[key] > 150):
        plt.bar(c, int(pitch_occurrences[key]) , width = 0.2, label = str(key) )
    # plt.subplots_adjust(bottom= 0.2, top = 0.98)
    
    c += 1
fontP = FontProperties()
fontP.set_size('small')
# plt.legend([plot1], "title", prop=fontP)
plt.legend(prop=fontP)
plt.show()