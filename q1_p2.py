import utils
import numpy as np
import cv2
import panorama
import read_config

path = read_config.path_config()
res_path = path['res_path']


def p2(flag=False):
    """
    part two
    I explained my approach in the functions used below,
    just as a short review, I moved all the 5 key frames to the reference frame
    which was 450, and then used a dp algorithm to merge them.

    Inputs:
    --> flag: if true saving results, else plotting
    Outputs:
    ==> Nothing, just saving or plotting results
    """
    key_frames, H, x_min, y_min, x_max, y_max = panorama.key_frames_corners_homography()
    [x_offset, y_offset] = [-x_min, -y_min]
    (x_size, y_size) = (x_max - x_min, y_max - y_min)
    offset = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
    homography_frames = [cv2.warpPerspective(key_frames[i], np.dot(offset, H[i, :, :]), (x_size, y_size))
                         for i in range(5)]

    full_image = homography_frames[0]
    for i in range(1, 5):
        full_image = panorama.panorama_key_frames(full_image.copy(), homography_frames[i].copy())

    if flag:
        utils.save_img(full_image, res_path + 'res04-key-frames-panorama.jpg', True)
    else:
        utils.plot_img(full_image)


p2(True)
