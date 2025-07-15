from music21 import stream, note, chord, midi
import random
import re
import os
import copy  # ✅ new import

prompt = os.getenv("PROMPT", "happy melody in C major").lower()

def extract_key(prompt):
    match = re.search(r'([a-g])\s*major', prompt)
    if match:
        return match.group(1).upper() + '4'
    return 'C4'

base_note = extract_key(prompt)
scale = [note.Note(base_note)]
for interval in [2, 2, 1, 2, 2, 2, 1]:  # major scale intervals
    scale.append(scale[-1].transpose(interval))

melody = stream.Stream()
for _ in range(32):
    original = random.choice(scale)
    n = copy.deepcopy(original)  # ✅ proper note duplication
    n.quarterLength = random.choice([0.25, 0.5, 1.0])
    melody.append(n)

mf = midi.translate.streamToMidiFile(melody)
mf.open("output.mid", 'wb')
mf.write()
mf.close()
