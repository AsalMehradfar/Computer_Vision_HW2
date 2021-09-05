import video_processing
import utils
import tqdm
import numpy as np
import cv2
import read_config
from scipy import ndimage as nd

path = read_config.path_config()
res_path = path['res_path']
low_resolution_frames_path = path['low_resolution_frames_path']
background_frames_path = path['background_frames_path']

params = read_config.param_config('param_config_q1')
NUM_FRAMES = params['num_frames']


def p6():
    """
    part six
    first I load the original image and the background,
    then I make a difference matrix same as the previous parts using norm 2,
    here some individual points have big differences so I used a uniform filter
    to solve this problem,
    after that I normalize the difference matrix and use a threshold for emphasizing
    the points which are the foregrounds.
    now we should make the explained points red by adding 100 to their R and clipping
    them to 255.
    at last write the frames in batches on the video in a loop.

    Inputs:
    --> Nothing
    Outputs:
    ==> Nothing, just saving the video
    """
    img = utils.get_img(low_resolution_frames_path + str(1).zfill(3) + '.jpg')
    vid_size = (img.shape[1], img.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(res_path + 'res08-foreground-video.mp4', fourcc, 30, vid_size)
    frames = []

    for i in tqdm.tqdm(range(NUM_FRAMES)):
        img1 = utils.get_img(low_resolution_frames_path + str(i + 1).zfill(3) + '.jpg').astype(np.float16)
        img2 = utils.get_img(background_frames_path + str(i + 1).zfill(3) + '.jpg').astype(np.float16)
        diff = np.sum(((img1 - img2) / 100) * ((img1 - img2) / 100), axis=2)
        diff = nd.uniform_filter(diff.astype(np.uint8), size=15, mode='constant')
        diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
        diff = diff > 0.01
        frame = img1.copy()
        frame[:, :, 0] = np.clip(frame[:, :, 0] + 100 * diff, 0, 255)
        frames.append(frame.astype(np.uint8))
        if (i + 1) % 30 == 0:
            for j in range(30):
                video.write(cv2.cvtColor(frames[j], cv2.COLOR_RGB2BGR))
            frames = []

    video.release()


video_processing.capture_frames(res_path + 'res07-background-video.mp4', background_frames_path)
p6()
