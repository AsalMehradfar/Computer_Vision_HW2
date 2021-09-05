import cv2
import numpy as np
import utils
import tqdm
import panorama
import read_config

path = read_config.path_config()
res_path = path['res_path']
low_resolution_frames_path = path['low_resolution_frames_path']

params = read_config.param_config('param_config_q1')
NUM_FRAMES = params['num_frames']


def p3():
    """
    part three
    here I easily apply the H matrix to each frame using a same offset
    computed before and write the frames in batches on the video.
    pay attention that open-cv consider images in BGR not RGB, so before writing
    we should consider this fact.

    Inputs:
    --> Nothing
    Outputs:
    ==> Nothing, just saving the panorama video
    """
    H, x_min, y_min, x_max, y_max = panorama.video_corners_homography()
    [x_offset, y_offset] = [-x_min, -y_min]
    size = (x_max - x_min, y_max - y_min)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(res_path + 'res05-reference-plane.mp4', fourcc, 30, size)
    frames = []

    for i in tqdm.tqdm(range(NUM_FRAMES)):
        img = utils.get_img(low_resolution_frames_path + str(i + 1).zfill(3) + '.jpg')
        offset = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
        frame = cv2.warpPerspective(img, np.dot(offset, H[i, :, :]), size)
        frames.append(frame)
        if (i + 1) % 30 == 0:
            for j in range(30):
                video.write(cv2.cvtColor(frames[j], cv2.COLOR_RGB2BGR))
            frames = []

    video.release()


p3()
