import sys
import os
import argparse
from musicxml import MusicXML

if __name__ == '__main__':

    """
    Command line args:

    -input <input directory with MusicXMLS>
    -output_seq <output directory to write sequences to>
    -output_vocab <output directory to write vocab to>
    """

    # Parse command line arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', dest='input', type=str, required='-c' not in sys.argv, help='Path to the input directory with MusicXMLs.')
    parser.add_argument('-output_seq', dest='output_seq', type=str, required=True, help='Path to the output directory to write sequences.')
    parser.add_argument('-output_vocab', dest='output_vocab', type=str, required=True, help='Path to the output directory to write vocab.')
    parser.add_argument('--gen_annotations', dest='gen_annotations', action='store_true')
    args = parser.parse_args()

    tokens = list()

    # For tracking number of MusicXML files read
    file_num = 0

    # Go through all inputs generating output sequences
    for i, file_name in enumerate(os.listdir(args.input)):

        # Ignore non .musicxml files
        if not file_name.endswith('.musicxml'):
            continue

        # Create a MusicXML object for generating sequences
        input_path = os.path.join(args.input, file_name)
        output_seq_path = os.path.join(args.output_seq, ''.join(file_name.split('.')[:-1]) + '.semantic')
        musicxml_obj = MusicXML(input_file=input_path, output_file=output_seq_path, gen_annotations=args.gen_annotations)

        try:
            # Write sequence
            sequences = musicxml_obj.get_sequences()
            musicxml_obj.write_sequences(sequences)
            file_num += 1

            # Add new unique tokens to vocab
            sequences = [x for x in sequences if x != '']
            for seq in sequences:
                s = seq.split(' + ')  # Each element is a series of simultaneous tokens
                for elem in s:
                    notes = elem.split(' ')

                    for n in notes:
                        if 'note' in n:
                            n_sub = n.replace('.', '')
                            n_pitch, n_rhythm = n_sub.split("_")

                            if n_pitch not in tokens:
                                tokens.append(n_pitch)
                            if n_rhythm not in tokens:
                                tokens.append(n_rhythm)
                        elif "rest" in n:
                            n_sub = n.replace('.', '')
                            n_pitch, n_rhythm = n_sub.split("-")

                            if n_pitch not in tokens:
                                tokens.append(n_pitch)
                            if n_rhythm not in tokens:
                                tokens.append(n_rhythm)
                        elif n != '\n':
                            if n not in tokens:
                                tokens.append(n)
        except UnicodeDecodeError: # Ignore bad MusicXML
            pass  # TODO nuke bad files

    # Sort vocab for readability
    tokens.remove('')
    tokens.sort()

    # Write vocab
    os.makedirs(os.path.dirname(args.output_vocab), exist_ok=True)

    with open(os.path.join(args.output_vocab, "vocab.txt"), 'w') as f:
        f.write("\n".join(map(str, tokens)))

    print('Num MusicXML files read:', file_num)
    print("Unique Tokens: {:4d}".format(len(tokens)))
