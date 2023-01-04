import cv2
import os 
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

# Settings
parser = argparse.ArgumentParser(description='Vid2Clip for veo data')
parser.add_argument('-i', '--input', help="Input folder path", required=True)
parser.add_argument('-o', '--output', help="Output folder path", required=True)

def job(video_path, input_path, output_path):
    # read video
    vs = cv2.VideoCapture(os.path.join(input_path, video_path))
    total_frames = int(vs.get(7))
    ret, frame = vs.read()

    # clips list
    clips = []

    # initialize object detector
    object_detector = cv2.createBackgroundSubtractorKNN(history=1000, dist2Threshold=2000, detectShadows=False)

    # flag that we write frames
    frame_counter = 1
    frame_interval = 1
    check_every = 60
    for i in tqdm(range(int(total_frames/check_every))):
        # iterate frame_counter
        frame_counter += check_every

        # read next frame
        vs.set(1, frame_counter)
        ret, frame = vs.read() 

        if ret == True:
            # change frame to get bottom right corner    
            frame = frame[int(frame.shape[0]/2):, int(frame.shape[1]/2):]
            search_frame = frame[:, int(frame.shape[1]/2)-20:int(frame.shape[1]/2)+20]

            # Create mask
            mask = object_detector.apply(search_frame)

            # Erode it and dilate it to avoid small points
            kernel = np.ones((1,1), np.uint8) 
            mask = cv2.erode(mask, kernel, iterations=1) 
            mask = cv2.dilate(mask, kernel, iterations=1)
                
            # Find contours in image
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # Calculate the area of the specific contour
                area = cv2.contourArea(cnt)
                
                # Only look at contours with an area of over 2k pixels
                if area > 2000:
                    clip = []
                    for j in range(1,8):
                        vs.set(1, frame_counter + j*frame_interval)
                        ret, frame = vs.read()
                        clip.append(frame[int(frame.shape[0]/2):, int(frame.shape[1]/2):])

                    clips.append(clip)
                    # skip_frames = 180
                    # frame = vs.set(1, frame_counter + skip_frames)
                    # frame_counter += skip_frames
                    # i += int(skip_frames/check_every)
                    break


    output_folder = os.path.join(output_path, video_path.split('.')[0])

    if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

    clip_counter = 0
    for clip in clips:
        clip_counter += 1
        if not os.path.isdir(os.path.join(output_folder, str(clip_counter))):
            os.mkdir(os.path.join(output_folder, str(clip_counter)))
        counter = 1
        for frame in clip:
            cv2.imwrite(os.path.join(output_folder, str(clip_counter), f'im{counter}.png'), frame)
            counter += 1
    

if __name__=="__main__":
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.isdir(os.path.join(os.getcwd(), output_path)):
        os.mkdir(os.path.join(os.getcwd(), output_path))

    Parallel(n_jobs=1)(delayed(job)(video, input_path, output_path) for video in tqdm(os.listdir(input_path)))

