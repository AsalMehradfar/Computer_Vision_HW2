import utils
import sift
import numpy as np
import cv2
import video_processing
import read_config

path = read_config.path_config()
video_resource_path = path['video_resource_path']
res_path = path['res_path']
frames_path = path['frames_path']
low_resolution_frames_path = path['low_resolution_frames_path']


def p1(flag=False):
    """
    part one
    drawing a red rectangle and then applying the H inverse,
    after that making a bigger matrix and first put the 900 frame and then 450
    on that.

    Inputs:
    --> flag: if true saving results, else plotting
    Outputs:
    ==> Nothing, just saving or plotting results
    """
    img1 = utils.get_img(frames_path + '450.jpg')
    img2 = utils.get_img(frames_path + '270.jpg')
    H = sift.find_homography(img1, img2)
    H_inv = np.linalg.inv(H)

    points = [(500, 500), (1000, 500), (1000, 1000), (500, 1000)]
    out_points = []
    color = (255, 0, 0)
    thickness = 5

    img1_new = img1.copy()
    img2_new = img2.copy()

    img1_new = cv2.rectangle(img1_new, points[0], points[2], color, thickness)

    for i in range(len(points)):
        p = np.array([[points[i][0], points[i][1], 1]]).transpose()
        m = np.matmul(H_inv, p)
        out_points.append((int(m[0] / m[2]), int(m[1] / m[2])))

    for i in range(len(out_points)):
        img2_new = cv2.line(img2_new, out_points[i], out_points[(i + 1) % 4], color, thickness)

    img = sift.panorama_two_images(img1, img2, H)

    if flag:
        utils.save_img(img1_new, res_path + 'res01-450-rect.jpg', True)
        utils.save_img(img2_new, res_path + 'res02-270-rect.jpg', True)
        utils.save_img(img, res_path + 'res03-270-450-panorama.jpg', True)
    else:
        utils.plot_img(img1_new)
        utils.plot_img(img2_new)
        utils.plot_img(img)


video_processing.capture_frames(video_resource_path, frames_path)
video_processing.capture_frames_low_res(video_resource_path, low_resolution_frames_path)
p1(True)
