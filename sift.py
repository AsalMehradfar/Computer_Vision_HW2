import cv2
import numpy as np
import read_config

params = read_config.param_config('param_config_q1')
R = params['r']
MAX_ITERATIONS = params['max_iterations']


def find_homography(img1, img2):
    """
    Finding match points between two images by RANSAC Algorithm

    Inputs:
    --> img1: the first desired image
    --> img2: the second desired image
    Outputs:
    ==> H: the 3*3 homography matrix
    """

    sift = cv2.SIFT_create()

    # find all key points
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # find all matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < R * n.distance:
            good_matches.append([m])

    kp1_match = [kp1[m[0].queryIdx] for m in good_matches]
    kp2_match = [kp2[m[0].trainIdx] for m in good_matches]

    # apply RANSAC
    src_pts = np.float32([kp.pt for kp in kp1_match]).reshape(-1, 1, 2)
    des_pts = np.float32([kp.pt for kp in kp2_match]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, des_pts, cv2.RANSAC, maxIters=MAX_ITERATIONS)

    return np.linalg.inv(H)


def perspective(img, H):
    """
    computing the homography of the image with matrix H

    Inputs:
    --> img: the desired image
    --> H: the homography matrix
    Outputs:
    ==> img_homography: the output image
    """
    x_min, y_min, x_max, y_max = corners_homography(img, H)
    [x_offset, y_offset] = [-x_min, -y_min]
    [x_size, y_size] = [x_max - x_min, y_max - y_min]
    offset = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
    img_homography = cv2.warpPerspective(img, np.dot(offset, H), (x_size, y_size))
    return img_homography


def corners_homography(img, H):
    """
    computing the min and max of the corners of an image
    after applying the homography matrix without offset
    we use these points to find offset and size of the homography

    Inputs:
    --> img: the desired image
    --> H: the homography matrix
    Outputs:
    ==> x_min: minimum x after homography
    ==> y_min: minimum y after homography
    ==> x_max: maximum x after homography
    ==> y_max: maximum y after homography
    """
    height, width, _ = img.shape
    corners = [np.array([[0, 0, 1]]).transpose(),
               np.array([[width - 1, 0, 1]]).transpose(),
               np.array([[0, height - 1, 1]]).transpose(),
               np.array([[width - 1, height - 1, 1]]).transpose()]
    [x_min, y_min, x_max, y_max] = [-1 for _ in range(4)]
    for c in corners:
        m = np.matmul(H, c)
        if x_min == -1:
            [x_min, y_min, x_max, y_max] = [int(m[0] / m[2]),
                                            int(m[1] / m[2]),
                                            int(m[0] / m[2]),
                                            int(m[1] / m[2])]
        else:
            [x_min, y_min, x_max, y_max] = [min(x_min, int(m[0] / m[2])),
                                            min(y_min, int(m[1] / m[2])),
                                            max(x_max, int(m[0] / m[2])),
                                            max(y_max, int(m[1] / m[2]))]
    return x_min, y_min, x_max, y_max


def panorama_two_images(img1, img2, H):
    """
    computing the panorama of two input images with matrix H on the second image
    the first image will be up to the second one in the union

    Inputs:
    --> img1: the first desired image
    --> img2: the second desired image
    --> H: the homography matrix
    Outputs:
    ==> img: the output panorama image
    """
    x_min, y_min, x_max, y_max = corners_homography(img2, H)
    [x_offset, y_offset] = [-x_min, -y_min]
    [x_size, y_size] = [max(img1.shape[1] + max(0, x_offset), x_max - x_min),
                        max(img1.shape[0] + max(0, y_offset), y_max - y_min)]
    offset = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
    img = cv2.warpPerspective(img2, np.dot(offset, H), (x_size, y_size))
    img[y_offset: img1.shape[0] + y_offset, x_offset: img1.shape[1] + x_offset, :] = img1
    return img
