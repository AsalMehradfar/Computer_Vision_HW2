import utils
import tqdm
import numpy as np
import cv2
import panorama
import read_config

path = read_config.path_config()
res_path = path['res_path']
low_resolution_frames_path = path['low_resolution_frames_path']

params = read_config.param_config('param_config_q1')
NUM_FRAMES = params['num_frames']


def p5():
    """
    part five

    I applied the inverse of the offset matrix computed before multiplied to H
    with the warp perspective on the background image for different frames
    I used the frame size of the extracted video for the size of the current video

    Inputs:
    --> Nothing
    Outputs:
    ==> Nothing, just saving the background video
    """
    H, x_min, y_min, x_max, y_max = panorama.video_corners_homography()
    [x_offset, y_offset] = [-x_min, -y_min]
    img = utils.get_img(low_resolution_frames_path + str(1).zfill(3) + '.jpg')
    vid_size = (img.shape[1], img.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(res_path + 'res07-background-video.mp4', fourcc, 30, vid_size)
    frames = []

    for i in tqdm.tqdm(range(NUM_FRAMES)):
        img = utils.get_img(res_path + 'res06-background-panorama.jpg')
        offset = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
        frame = cv2.warpPerspective(img, np.linalg.inv(np.dot(offset, H[i, :, :])), vid_size)
        frames.append(frame)
        if (i + 1) % 30 == 0:
            for j in range(30):
                video.write(cv2.cvtColor(frames[j], cv2.COLOR_RGB2BGR))
            frames = []

    video.release()


p5()
