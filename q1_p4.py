import utils
import video_processing
from scipy import stats
import tqdm
import read_config
import numpy as np

path = read_config.path_config()
res_path = path['res_path']
homography_frames_path = path['homography_frames_path']

params = read_config.param_config('param_config_q1')
NUM_FRAMES = params['num_frames']


def p4(flag=False):
    """
    part four
    I used the mode for computing the value of background pixels
    but the RAM of the laptop does not let us to directly compute the mode for all of
    the pixels of an image,
    so I made some batches and computed mode for them.
    for RAM 8, 100 is a good value for running and for RAM 12 250 works correctly.
    with this approach and computing mode for RGB separately, we see a lot of black pixels
    in the corners than happens because of camera moving.
    to solve this I take a sum of RGB and consider all the values less than a special threshold
    like 40 as black pixels and remove them from mode computing. now we have a better background
    image in the corners.

    but how do I remove black pixels from the mode:
    we know that all the images have the values from 0 to 255, so for each black point
    in an image I assigned a value more than 255 and different from others to black pixels
    and now because of the difference we are sure that the mode is not any of these pixels.

    Inputs:
    --> flag: if true saving results, else plotting
    Outputs:
    ==> Nothing, just saving or plotting results
    """

    batch = 100
    thresh = 40
    frame = utils.get_img(homography_frames_path + str(1).zfill(3) + '.jpg')
    frames = np.zeros((NUM_FRAMES, frame.shape[0], batch, 3))

    background_img = np.zeros(frame.shape)

    for j in tqdm.tqdm(range(0, frame.shape[1] - batch, batch)):
        temp = np.zeros((frame.shape[0], batch, 3)) + 256
        for i in range(NUM_FRAMES):
            frame = utils.get_img(homography_frames_path + str(i + 1).zfill(3) + '.jpg')
            a = frame[:, j:j + batch, :]
            s = np.sum(a, axis=2)
            s = s < thresh
            (x, y) = np.nonzero(s)
            a[x, y, :] = temp[x, y, :]
            temp[x, y, :] = temp[x, y, :] + 1
            frames[i, :, :, :] = a
        background_img[:, j:j + batch, :] = stats.mode(frames, axis=0).mode

    # the last batch
    frames = np.zeros((NUM_FRAMES, frame.shape[0], frame.shape[1] % batch, 3))
    j = int(frame.shape[1] / batch) * batch
    temp = np.zeros((frame.shape[0], frame.shape[1] % batch, 3)) + 256
    for i in range(NUM_FRAMES):
        frame = utils.get_img(homography_frames_path + str(i + 1).zfill(3) + '.jpg')
        a = frame[:, j:, :]
        s = np.sum(a, axis=2)
        s = s < thresh
        (x, y) = np.nonzero(s)
        a[x, y, :] = temp[x, y, :]
        temp[x, y, :] = temp[x, y, :] + 1
        frames[i, :, :, :] = a
    background_img[:, j:, :] = stats.mode(frames, axis=0).mode

    if flag:
        utils.save_img(background_img, res_path + 'res06-background-panorama.jpg', True)
    else:
        utils.plot_img(background_img)


video_processing.capture_frames(res_path + 'res05-reference-plane.mp4', homography_frames_path)
p4(True)
