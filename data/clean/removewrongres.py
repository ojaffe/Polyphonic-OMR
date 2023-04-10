import os
import argparse

import cv2
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-imgs', dest='imgs', type=str, required=True, help='Path to the directory with imgs.')
    parser.add_argument('-labels', dest='labels', type=str, required=True, help='Path to the directory with labels.')
    args = parser.parse_args()

    img_paths = [x for x in os.listdir(args.imgs) if '.png' in x]

    counter = 0
    total_len = len(img_paths)
    for i in tqdm(img_paths):
        im = cv2.imread(os.path.join(args.imgs, i))
        
        if im.shape != (432, 2977, 3):
            os.remove(os.path.join(args.imgs, i))

            img_id = i.split('-')[0]
            num = i.split('-')[1].split('.')[0].lstrip('0')
            sem_file = img_id + '-' + num + '.semantic'
            os.remove(os.path.join(args.labels, sem_file))


            counter += 1

    print("{:5d} images with incorrect resolution, {:.2f}% of total".format(
        counter,
        counter / total_len
    ))


if __name__ == "__main__":
    main()