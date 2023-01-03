from pathlib import Path
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--look_in_dir", type=str, default="",
                    help="starting folder name to look inside")

opt = parser.parse_args()
print(opt)

pattern = opt.look_in_dir + r'\/([\w0-9\/]+)\/im'

start_dir = Path.cwd() / opt.look_in_dir

for dir_nm in start_dir.glob('**/*7.png'):
    if dir_nm.is_file():
        match = re.findall(pattern, dir_nm.as_posix())
        if match:
            with open(start_dir/'folders.txt', 'a') as f:
                f.write(match[0]+'\n')
