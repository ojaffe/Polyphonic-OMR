import os
import csv

import numpy as np

def strip_series(input: str, max_chord_stack: int) -> (list, list):
    """
    Given some input string of sequential notes, parse it to produce
    two lists, separating the pitch from rhythm
    """
    output = ['<SOS>']

    sequence = input.split(' + ')  # Each element is a series of simultaneous tokens
    for elem in sequence:
        notes = elem.split(' ')
        if '' in notes:
            notes.remove('')

         # If super massive chord, then take random amount of them up to threshold
        if len(notes) >= max_chord_stack:
            np.random.shuffle(notes)
            notes = notes[:max_chord_stack]

        if len(notes) >= 2:
            output.append('<CHORD START>')

        for c in notes[:max_chord_stack]:
            c_sub = c.replace('.', '')

            if "note" in c_sub:
                c_pitch, c_rhythm = c_sub.split("_")
                output.append(c_pitch)
                output.append(c_rhythm)
            elif "rest" in c_sub:
                c_pitch, c_rhythm = c_sub.split("-")
                output.append(c_pitch)
                output.append(c_rhythm)
            else:
                output.append(c_sub)
            
            if "." in c:
                output.append("<DOT>")

        if len(notes) >= 2:
            output.append('<CHORD END>')

    output.append('<EOS>')
    return output


def make_csv(data_cfg: dict) -> None:
    """
    Given an input directory and output path, for each .png and .semantic pair
    within the directory, write a row in an output csv of the .png path, and
    note pitch+rhythm of the .semantic file
    """
    data_dir = data_cfg["data_dir"]
    csv_out = data_cfg["csv_out"]
    max_chord_stack = data_cfg["max_chord_stack"]

    # Get lists of valid tokens
    with open(data_cfg["vocab_path"], 'r') as f:
        vocab = f.read().split()

    with open(csv_out, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['img_path', 'tokens'])

        data_imgs = [x for x in os.listdir(data_dir) if '.png' in x]
        for img in data_imgs:

            # Get respective .semantic file
            id = img.split('-')[0]
            num = img.split('-')[1].split('.')[0].lstrip('0')
            sem_file = id + '-' + num + '.semantic'
            if not os.path.isfile(os.path.join(data_dir, sem_file)):
                print(sem_file)
                break

            with open(os.path.join(data_dir, sem_file), 'r') as f_sem:
                contents = f_sem.read()

            tokens = strip_series(contents, max_chord_stack)

            writer.writerow([img, tokens])
