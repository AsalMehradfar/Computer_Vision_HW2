import numpy as np
import tqdm
import utils
import sift
import read_config

params = read_config.param_config('param_config_q1')
NUM_FRAMES = params['num_frames']


def compute_H_to_450():
    """
    computing the homography matrix for all the images to the reference image:
    moving images less than 90 to 90 and then to 270 and at last 450,
    moving images less than 270 and more than 90 to 270 and then 450,
    moving images less than 630 and more than 270 directly to 450,
    moving images less than 810 and more than 630 to 630 and then 450,
    moving image more than 810 to 810 and then to 630 and at last 450.

    Inputs:
    --> Nothing
    Outputs:
    ==> H: a 900*3*3 matrix, each row is a homography matrix
        for one of the frames to the reference image
    """
    H = np.zeros((NUM_FRAMES, 3, 3))
    ref_images, H[89, :, :], H[269, :, :], H[449, :, :], H[629, :, :], H[809, :, :] = compute_key_frames_H()

    for i in tqdm.tqdm(range(NUM_FRAMES)):
        if i == 89 or i == 269 or i == 449 or i == 629 or i == 809:
            continue
        elif i < 89:
            H[i, :, :] = np.matmul(H[89, :, :], sift.find_homography(ref_images[0],
                                                                     utils.get_img(
                                                                         'VideoFramesLowRes/' + str(i + 1).zfill(
                                                                             3) + '.jpg')))
        elif i < 269:
            H[i, :, :] = np.matmul(H[269, :, :], sift.find_homography(ref_images[1],
                                                                      utils.get_img(
                                                                          'VideoFramesLowRes/' + str(i + 1).zfill(
                                                                              3) + '.jpg')))
        elif i < 629:
            H[i, :, :] = sift.find_homography(ref_images[2],
                                              utils.get_img('VideoFramesLowRes/' + str(i + 1).zfill(3) + '.jpg'))
        elif i < 809:
            H[i, :, :] = np.matmul(H[629, :, :], sift.find_homography(ref_images[3],
                                                                      utils.get_img(
                                                                          'VideoFramesLowRes/' + str(i + 1).zfill(
                                                                              3) + '.jpg')))
        else:
            H[i, :, :] = np.matmul(H[809, :, :], sift.find_homography(ref_images[4],
                                                                      utils.get_img(
                                                                          'VideoFramesLowRes/' + str(i + 1).zfill(
                                                                              3) + '.jpg')))

    return H


def compute_key_frames_H():
    """
    computing key frames and homography of them

    Inputs:
    --> Nothing
    Outputs:
    ==> key_images: the list of 5 key frames
    ==> H90: the homography of 90 to 450
    ==> H180: the homography of 180 to 450
    ==> H450: the homography of 450 to 450
    ==> H630: the homography of 630 to 450
    ==> H810: the homography of 810 to 450
    """
    direc = 'VideoFramesLowRes/'
    key_images = [
        utils.get_img(direc + '090.jpg'),
        utils.get_img(direc + '270.jpg'),
        utils.get_img(direc + '450.jpg'),
        utils.get_img(direc + '630.jpg'),
        utils.get_img(direc + '810.jpg')
    ]
    H90 = np.matmul(sift.find_homography(key_images[2], key_images[1]), sift.find_homography(key_images[1], key_images[0]))
    H270 = sift.find_homography(key_images[2], key_images[1])
    H450 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    H630 = sift.find_homography(key_images[2], key_images[3])
    H810 = np.matmul(sift.find_homography(key_images[2], key_images[3]), sift.find_homography(key_images[3], key_images[4]))

    return key_images, H90, H270, H450, H630, H810


def video_corners_homography():
    """
    computing the minimum and maximum coordinates for all the frames after homography to
    the reference image 450

    Inputs:
    --> Nothing
    Outputs:
    ==> H: a 900*3*3 matrix, each row is a homography matrix
        for one of the frames to the reference image
    ==> x_min: minimum x after homography for all of the frames
    ==> y_min: minimum y after homography for all of the frames
    ==> x_max: maximum x after homography for all of the frames
    ==> y_max: maximum y after homography for all of the frames
    """
    H = compute_H_to_450()
    [x_min, y_min, x_max, y_max] = sift.corners_homography(utils.get_img('VideoFramesLowRes/' + str(1).zfill(3) + '.jpg'),
                                                      H[0, :, :])
    for i in range(1, NUM_FRAMES):
        a = sift.corners_homography(utils.get_img('VideoFramesLowRes/' + str(i + 1).zfill(3) + '.jpg'),
                               H[i, :, :])
        x_min = min(x_min, a[0])
        y_min = min(y_min, a[1])
        x_max = max(x_max, a[2])
        y_max = max(y_max, a[3])

    return H, x_min, y_min, x_max, y_max


def key_frames_corners_homography():
    """
    computing corers and homography for 5 key frames
    computing the minimum and maximum coordinates for all of the key frames
    after homography to the reference image 450

    Inputs:
    --> Nothing
    Outputs:
    ==> key_frames: the list of 5 key frames
    ==> H: the 5*3*3 array of homography
    ==> x_min: minimum of x
    ==> y_min: minimum of y
    ==> x_max: maximum of x
    ==> y_max: maximum of y
    """
    H = np.zeros((5, 3, 3))
    key_frames, H[0, :, :], H[1, :, :], H[2, :, :], H[3, :, :], H[4, :, :] = compute_key_frames_H()
    [x_min, y_min, x_max, y_max] = sift.corners_homography(key_frames[0], H[0, :, :])

    for i in range(1, 5):
        a = sift.corners_homography(key_frames[i], H[i, :, :])
        x_min = min(x_min, a[0])
        y_min = min(y_min, a[1])
        x_max = max(x_max, a[2])
        y_max = max(y_max, a[3])

    return key_frames, H, x_min, y_min, x_max, y_max


def panorama_key_frames(img1, img2):
    """
    computing the panorama of two images
    we use a squared difference matrix of two images for computing a value as cost
    first we find the union of two images, we crop the pixels of the union part and
    make a cost matrix of that size
    for this purpose I initialized the first line of the union then used dp algorithm
    for computing the cost matrix
    by this approach, I found the path with the lowest cost between two images
    and used the pixels of the first image for the right side of the panorama and
    the pixels of the second image for the left side

    Inputs:
    --> img1: the first desired image (base one)
    --> img2: the second desired image
    Outputs:
    ==> out_img: the panorama of two images
    """

    [fr1, fr2] = [img1.copy(), img2.copy()]
    mask = np.zeros(img1.shape)
    diff = np.sum(((img1 - img2) / 100) * ((img1 - img2) / 100), axis=2)

    # making the union matrix
    img1[np.any(img1 != [0, 0, 0], axis=-1)] = [1, 1, 1]
    img2[np.any(img2 != [0, 0, 0], axis=-1)] = [1, 1, 1]

    union = img1 * img2

    # computing the cost matrix for the union part
    (x, y, _) = np.nonzero(union)
    [x_min, y_min, x_max, y_max] = [min(x), min(y), max(x), max(y)]
    cropped_img = np.ones((x_max - x_min, y_max - y_min)) * np.inf

    # initialization
    for i in range(cropped_img.shape[1]):
        for j in range(cropped_img.shape[0]):
            if union[x_min + j, y_min + i, 0] == 1:
                if j < cropped_img.shape[0] / 4:
                    cropped_img[j, i] = diff[x_min + j, y_min + i]
                break
    # computing cost
    for i in range(1, cropped_img.shape[0]):
        for j in range(1, cropped_img.shape[1] - 1):
            m = min(cropped_img[i - 1, j - 1], cropped_img[i - 1, j], cropped_img[i - 1, j + 1])
            if m != np.inf:
                cropped_img[i, j] = diff[x_min + i, y_min + j] + m
    # omitting the last rows of the matrix
    for i in range(cropped_img.shape[1] - 1, 0, -1):
        for j in range(cropped_img.shape[0] - 1, 0, -1):
            if union[x_min + j, y_min + i, 0] == 1:
                if j > 3 * cropped_img.shape[0] / 4:
                    cropped_img[j, i] = np.inf
                break
            else:
                cropped_img[j, i] = np.inf

    # finding the path between two parts of panorama
    val = np.inf
    for i in range(cropped_img.shape[1] - 1, 0, -1):
        for j in range(cropped_img.shape[0] - 1, 0, -1):
            if cropped_img[j, i] != np.inf:
                if j > 3 * cropped_img.shape[0] / 4:
                    if cropped_img[j, i] < val:
                        [x, y, val] = [j + x_min, i + y_min, cropped_img[j, i]]
                break

    # computing the mask
    for i in range(img1.shape[0] - 1, x - 1, -1):
        for j in range(y):
            mask[i, j, :] = [1, 1, 1]

    while x > 0 and y > 0 and cropped_img[x - x_min, y - y_min] != np.inf:
        m = min(cropped_img[x - 1 - x_min, y - 1 - y_min],
                cropped_img[x - 1 - x_min, y - y_min],
                cropped_img[x - 1 - x_min, y + 1 - y_min])
        if m == cropped_img[x - 1 - x_min, y - 1 - y_min]:
            [x, y] = [x - 1, y - 1]
        elif m == cropped_img[x - 1 - x_min, y - y_min]:
            [x, y] = [x - 1, y]
        else:
            [x, y] = [x - 1, y + 1]
        for j in range(y):
            mask[x, j, :] = [1, 1, 1]

    for i in range(x):
        for j in range(y):
            mask[i, j, :] = [1, 1, 1]

    out_img = mask * fr1 + (1 - mask) * fr2

    return out_img

