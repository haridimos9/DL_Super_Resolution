from pathlib import Path
import re
import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="",
                    help="data directory path, contains folders.txt")

parser.add_argument("--train", type=float, default=0.85,
                    help="train percentage. scale 0 to 1")

parser.add_argument("--val", type=float, default=0.15,
                    help="validation percentage. scale 0 to 1")

parser.add_argument("--seed", type=int, default=1,
                    help="Seed for random shuffle. If 1 no seed is used")

opt = parser.parse_args()
print(opt)

if opt.train > 1 or opt.train < 0:
    print('Train ratio  should be between 0 and 1')
    sys.exit()
if opt.val > 1 or opt.val < 0:
    print('Validation ratio  should be between 0 and 1')
    sys.exit()
if (opt.train + opt.val) > 1:
    print('Train + Validation ratios  should be lower than 1')
    sys.exit()


data_path = Path.cwd() / opt.data_dir

try:
    with open((data_path / 'folders.txt'), 'r') as f:
        lines = f.readlines()
except:
    print('folders.txt not found.')
    sys.exit()

train = opt.train
val = opt.val
test = 1-train-val

test_flag = False
if test > 0:
    test_flag = True

data_len = len(lines)
data_idx = np.arange(data_len).astype(int)

if opt.seed != 1:
    np.random.seed(opt.seed)
np.random.shuffle(data_idx)


train_size = int(np.floor(train*data_len))
val_size = int(np.floor(val*data_len))

train_idx = data_idx[:train_size]

val_idx = data_idx[train_size:train_size+val_size]

if test_flag:
    test_idx = data_idx[train_size+val_size:]

for i in train_idx:
    with open((data_path/'train.txt'), 'a') as f:
        f.write(lines[i])


for i in val_idx:
    with open((data_path/'val.txt'), 'a') as f:
        f.write(lines[i])

if test_flag:
    for i in test_idx:
        with open((data_path/'test.txt'), 'a') as f:
            f.write(lines[i])
