import sys
import os
import argparse

from tqdm import tqdm
from lxml import etree


def remove_credits(input_dir):
    for file_name in tqdm(os.listdir(input_dir), desc="Removing credits"):
        if not file_name.endswith('.musicxml'):
            continue

        input_file = os.path.join(input_dir, file_name)

        try:
            doc=etree.parse(input_file)
        except:
            os.remove(input_file)
            continue

        # Remove credits/rights text that could interfere with music 
        for elem in doc.xpath('//credit'):
            parent = elem.getparent()
            parent.remove(elem)
        for elem in doc.xpath('//rights'):
            parent = elem.getparent()
            parent.remove(elem)

        # Write to same file with next XML
        f = open(input_file, 'wb')
        f.write(etree.tostring(doc))
        f.close()

    print('Successfully removed credits!')


def remove_no_label(input_dir):
    num_missing = 0
    num_total = 0

    for file_name in tqdm(os.listdir(input_dir), desc="Removing examples without label"):
        if not file_name.endswith('.png'):
            continue

        sample_id = file_name.split('-')[0]
        num = file_name.split('-')[1].split('.')[0]

        sem_name = sample_id + '-' + num + '.semantic'

        # Try find respective semantic file
        num_total += 1
        try:
            sem_file = open(os.path.join(input_dir, sem_name), 'r')
            sem_file.close()
        except FileNotFoundError:
            num_missing += 1
            os.remove(os.path.join(input_dir, file_name))
                
    print('Num missing: {:}\t Num total: {:}\t Examples Left: {:}'.format(num_missing, num_total, num_total-num_missing))


def remove_sparse_samples(input_dir):
    sparse_count = 0

    for file_name in tqdm(os.listdir(input_dir), desc="Removing sparse examples"):
        if not file_name.endswith('.semantic'):
            continue

        # Try different possible names for file depending on leading 0s
        sem_name1 = file_name
        sample_id = file_name.split('-')[0]
        num = file_name.split('-')[1].split('.')[0]
        
        sem_file = open(os.path.join(input_dir, sem_name1), 'r')
        seq = sem_file.read()

        # Check label for sparse sample
        if 'note' not in seq:
            sem_file.close()
            sparse_count += 1
            img_filename = str(sample_id) + '-' + str(num) + '.png'

            os.remove(os.path.join(input_dir, img_filename))
            os.remove(os.path.join(input_dir, file_name))

    print('Number of sparse files: ', sparse_count)


def remove_title_imgs(input_dir):
    for file_name in tqdm(os.listdir(input_dir), desc="Removing title images"):
        if file_name.endswith('-1.png'):
            os.remove(os.path.join(input_dir, file_name))


def remove_wrong_res(input_dir):
    counter = 0
    total_len = len(img_paths)

    for file_name in tqdm(os.listdir(input_dir), desc="Removing wrong res"):
        if not file_name.endswith('.png'):
            continue

        im = cv2.imread(os.path.join(input_dir, i))
        
        if im.shape != (432, 2977, 3):
            os.remove(os.path.join(input_dir, i))

            img_id = i.split('-')[0]
            num = i.split('-')[1].split('.')[0].lstrip('0')
            sem_file = img_id + '-' + num + '.semantic'
            os.remove(os.path.join(input_dir, sem_file))

            counter += 1

    print("{:5d} images with incorrect resolution, {:.2f}% of total".format(
        counter,
        counter / total_len
    ))


def main():

    """
    Main method
    """

    num_files = 0

    # Parse command line arguments for input directory
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', dest='input', type=str, required='-c' not in sys.argv, help='Path to the input directory with MusicXMLs.')
    parser.add_argument('--remove-credits', dest='remove_credits', action='store_true', help='Remove credits from musicxml files')
    parser.add_argument('--remove-no-label', dest='remove_no_label', action='store_true', help='Removes images that do not have respective label')
    parser.add_argument('--remove-sparse-samples', dest='remove_sparse_samples', action='store_true', help='Remove examples that have no notes')
    parser.add_argument('--remove-title-imgs', dest='remove_title_imgs', action='store_true', help='Remove examples that contain the title of the score')
    parser.add_argument('--remove-wrong-res', dest='remove_wrong_res', action='store_true', help='Remove examples when their image is not a specific resolution')
    args = parser.parse_args()

    # Convert .png file names so they match resepctive .semantic
    for file_name in tqdm(os.listdir(args.input), desc='Converting image file names'):
        if not file_name.endswith('.png'):
            continue

        new_name = file_name.split('-')[0] + '-' + str(int(file_name.split('-')[1].split('.')[0])) + '.png'
        os.rename(os.path.join(args.input, file_name), os.path.join(args.input, new_name))

    if args.remove_credits:
        remove_credits(args.input)
    if args.remove_no_label:
        remove_no_label(args.input)
    if args.remove_sparse_samples:
        remove_sparse_samples(args.input)
    if args.remove_title_imgs:
        remove_title_imgs(args.input)
    if args.remove_wrong_res:
        remove_wrong_res(args.input)


if __name__ == "__main__":
    main()