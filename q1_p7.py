import cv2
import utils
import numpy as np
import panorama
import tqdm
import read_config

path = read_config.path_config()
res_path = path['res_path']
low_resolution_frames_path = path['low_resolution_frames_path']


def p7():
    """
    part seven
    I increased the video size by multiplying the x size of the image to 1.5
    in the warp perspective.
    pay attention that when we make the video wider, in the last frames of the 30sec video
    we will see a part of the video black, because we do not have any data in the frames
    for that part
    so I removed the frames after 630 that did not give us any useful data
    our final video now is around 21secs.

    Inputs:
    --> Nothing
    Outputs:
    ==> Nothing, just saving the wide background video
    """
    H, x_min, y_min, x_max, y_max = panorama.video_corners_homography()
    [x_offset, y_offset] = [-x_min, -y_min]
    img = utils.get_img(low_resolution_frames_path + str(1).zfill(3) + '.jpg')
    vid_size = (int(img.shape[1] * 1.5), img.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(res_path + 'res09-background-video-wider.mp4', fourcc, 30, vid_size)
    frames = []

    for i in tqdm.tqdm(range(630)):
        img = utils.get_img(res_path + 'res06-background-panorama.jpg')
        offset = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
        frame = cv2.warpPerspective(img, np.linalg.inv(np.dot(offset, H[i, :, :])), vid_size)
        frames.append(frame)
        if (i + 1) % 30 == 0:
            for j in range(30):
                video.write(cv2.cvtColor(frames[j], cv2.COLOR_RGB2BGR))
            frames = []

    video.release()


p7()
