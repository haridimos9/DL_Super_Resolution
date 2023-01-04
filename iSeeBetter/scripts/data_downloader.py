import os
import argparse
import logger
import pathlib 
from SoccerNet.Downloader import SoccerNetDownloader

# Create a folder Data if it does not exist
def isFolder(path: pathlib.Path) -> None: 
    if not path.is_dir():
        os.mkdir(path)
        logger.info('Created Data folder')

def main():
    main_parser = argparse.ArgumentParser(
        description="parser for downloading SoccerNet Videos",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    main_parser.add_argument(
        "--path",
        type=pathlib.Path,
        required=False,
        default="/work3/s213685/Datasets/SoccerNet",
        help="Save folder",
    )
    main_parser.add_argument(
        "--password",
        type=str,
        required=False,
        default=os.environ["SOCCERNET_PASS"],
        help="Password",
    )
    args = main_parser.parse_args()

    isFolder(args.path)

    # Download SoccerNet videos
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=args.path)

    # Provide password
    mySoccerNetDownloader.password = args.password

    ########################################################################
    # Run in terminal inside the folder to download the data.
    # Ctrl + C to stop a download. Take care to stop after the same number of matches
    # when you download the low quality data

    # Select which videos to download
    try:
        mySoccerNetDownloader.downloadGames(
            files=["1_720p.mkv"], split=["train", "valid", "test"])
    except KeyboardInterrupt as e:
        logger.info('Stopped')
        pass


if __name__ == "__main__":
    main()