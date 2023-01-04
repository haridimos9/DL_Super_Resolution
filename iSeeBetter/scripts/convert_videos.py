import os 
import subprocess
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


def job(video, input, output):
        command = [
            'ffmpeg', 
            '-i',
            os.path.join(input, video),
            '-c:v',
            'libx264',
            os.path.join(output, video.replace('.ts', '.mp4'))
            ]

        subprocess.call(' '.join(command), shell=True)

def main():
    main_parser = argparse.ArgumentParser(
        description="parser for converting ts videos to mp4",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    main_parser.add_argument(
        "--input_path",
        type=str,
        required=False,
        default="/work3/s213685/Datasets/Veo/ts/",
        help="Path to folder containing videos",
    )
    main_parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="/work3/s213685/Datasets/Veo/mp4",
        help="Path to output folder",
    )

    main_parser.add_argument(
        "--jobs",
        type=int,
        required=False,
        default=1,
        help="Number of parallel jobs",
    )

    args = main_parser.parse_args()
    Parallel(n_jobs=args.jobs)(delayed(job)(video, args.input_path, args.output_path) for video in tqdm(os.listdir(args.input_path)))
    

if __name__ == '__main__':
    main()

