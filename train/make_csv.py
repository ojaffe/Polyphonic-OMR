import os
import csv

import numpy as np
from tqdm import tqdm
import pandas as pd


def strip_series(input: str, max_chord_stack: int) -> (list, list):
    """
    Given some input string of sequential notes, parse it to produce a tokenized version. Additionally
    calculate note density, i.e., how many symbols (notes+rests) per input series
    """
    density = 0
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
                c_pitch = c_sub.split("_")[0]
                c_rhythm = "_".join(c_sub.split("_")[1:])
                output.append(c_pitch)
                output.append(c_rhythm)

                density += 1
            elif "rest" in c_sub:
                c_pitch, c_rhythm = c_sub.split("-")
                output.append(c_pitch)
                output.append(c_rhythm)

                density += 1
            else:
                output.append(c_sub)
            
            if "." in c:
                output.append("<DOT>")

        if len(notes) >= 2:
            output.append('<CHORD END>')

    output.append('<EOS>')
    return output, density


def make_csv(data_cfg: dict) -> None:
    """
    Given an input directory and output path, for each .png and .semantic pair
    within the directory, write a row in an output csv of the .png path, and
    note pitch+rhythm of the .semantic file
    """
    data_dir = data_cfg["data_dir"]
    csv_out = data_cfg["csv_out"]
    hard_csv_out = data_cfg["hard_csv_out"]
    max_chord_stack = data_cfg["max_chord_stack"]

    # Get lists of valid tokens
    with open(data_cfg["vocab_path"], 'r') as f:
        vocab = f.read().split()

    data = {"img_path": [], "tokens": []}
    all_densities = []
    data_imgs = [x for x in os.listdir(data_dir) if '.png' in x]
    for img_path in tqdm(data_imgs):
        # Get respective .semantic file
        id = img_path.split('-')[0]
        num = img_path.split('-')[1].split('.')[0].lstrip('0')
        sem_file = id + '-' + num + '.semantic'
        if not os.path.isfile(os.path.join(data_dir, sem_file)):
            print(sem_file)
            break

        with open(os.path.join(data_dir, sem_file), 'r') as f_sem:
            contents = f_sem.read()

        tokens, density = strip_series(contents, max_chord_stack)
        all_densities.append(density)

        data["img_path"].append(img_path)
        data["tokens"].append(tokens)

    all_densities_sorted = list(reversed(np.argsort(all_densities)))

    df = pd.DataFrame.from_dict(data)

    # Genetate separate "hard" dataset
    if data_cfg["gen_hard_dataset"]:
        hard_idxs = all_densities_sorted[:1000]
        df_easy = df.drop(hard_idxs)
        df_hard = df.loc[hard_idxs]

        df_easy.to_csv(csv_out, header=True, index=False)
        df_hard.to_csv(hard_csv_out, header=True, index=False)
    else:
        df.to_csv(header=True, index=False)
