import cv2
import os
import read_config

params = read_config.param_config('param_config_q1')
NUM_FRAMES = params['num_frames']


def capture_frames(path, save_path):
    """
    Save NUM_FRAMES video frames in a folder

    Inputs:
    --> path: the filename of the video
    --> save_path: the folder name of the saving frames
    Outputs:
    ==> Nothing, frames will be saved with 3 digit names
    """
    os.mkdir(save_path)
    vid = cv2.VideoCapture(path)

    for i in range(NUM_FRAMES):
        ret, frame = vid.read()
        if not ret:
            break
        cv2.imwrite(save_path + str(i + 1).zfill(3) + '.jpg', frame)

    vid.release()
    cv2.destroyAllWindows()


def capture_frames_low_res(path, save_path):
    """
    Save NUM_FRAMES video frames in a folder with low resolution
    we down sample each dim to half of its previous form
    so we decrease resolution to quarter of the original

    Inputs:
    --> Nothing
    Outputs:
    ==> Nothing, frames will be saved with 3 digit names
    """
    os.mkdir(save_path)
    vid = cv2.VideoCapture(path)

    for i in range(NUM_FRAMES):
        ret, frame = vid.read()
        if not ret:
            break
        size = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
        new_frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path + str(i + 1).zfill(3) + '.jpg', new_frame)

    vid.release()
    cv2.destroyAllWindows()
