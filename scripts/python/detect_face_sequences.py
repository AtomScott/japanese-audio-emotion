import argparse

from pathlib import Path

from JAVER.track_tools import *


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument()

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    video_paths = list(Path(args.in_path).glob())
    out_path = list(Path(args.out_path).glob())
    ref_paths = list(Path(args.ref_path).glob())

    batch_size = args.batch_size
    step_large = args.step_large
    step_small = args.step_small
    image_size = args.image_size

