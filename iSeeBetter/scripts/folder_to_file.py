import os
import argparse
import random
import sys

SOCCERNET = "SoccerNet"
VEO = "Veo"
VIDEO = "video"


def main():
    main_parser = argparse.ArgumentParser(
        description="parser for turning folder names to file",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    main_parser.add_argument(
        "--path",
        type=str,
        required=False,
        default="/work3/s213685/iSeeBetter-pytorch/vimeo_septuplet/HR",
        help="Path to root dataset\n"
        "├─ SoccerNet/\n"
        "│  ├─ 0001/\n"
        "│  ├─ 0002/\n"
        "│  ├─ .../\n"
        "├─ Veo/\n"
        "│  ├─ video1/\n"
        "│  │   ├─ 01/\n"
        "│  │   ├─ 02/\n"
        "│  │   ├─ .../\n"
        "│  ├─ video2/\n"
        "│  │   ├─ 01/\n"
        "│  │   ├─ 02/\n"
        "│  │   ├─ .../\n"
        "│  ├─ .../\n",
    )

    main_parser.add_argument(
        "--val_split",
        type=int,
        required=False,
        default=5,
        help="Proportion in percentage of the initial dataset to be used as validation",
    )

    args = main_parser.parse_args()

    wlk = os.walk(os.path.join(args.path, SOCCERNET))

    filenames = []
    for i, (root, dirname, files) in enumerate(sorted(wlk)):
        if i == 0:
            continue
        filenames.append(os.path.join(SOCCERNET, os.path.basename(root)))

    wlk = os.walk(os.path.join(args.path, VEO))
    for i, (root, _, _) in enumerate(wlk):
        if i == 0 or VIDEO in os.path.basename(root):
            continue

        # Take only the path starting inside the `VEO` folder
        # Fastest way is to use regex.
        root_split = root.split(VEO)
        without_prefix = os.path.relpath(root, os.path.join(root_split[0], VEO))

        # Take the path starting from the parent folder ot `VEO`
        filenames.append(os.path.join(VEO, without_prefix))

    random.shuffle(filenames)

    split_idx = int(0.01 * (100 - args.val_split) * len(filenames))

    # Training file list
    with open(os.path.join(args.path, "sep_trainlist.txt"), "w") as f:
        [f.write(name + "\n") for name in filenames[: split_idx - 1]]

    with open(os.path.join(args.path, "sep_val_list.txt"), "w") as f:
        [f.write(name + "\n") for name in filenames[split_idx:]]


if __name__ == "__main__":
    main()
