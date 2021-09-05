import cv2
import numpy as np
import read_config

path = read_config.path_config()
resource_path = path['resource_path']

params = read_config.param_config('param_config_q2')
BOARD_SIZE = (params['board_size_x'], params['board_size_y'])
a = params['a']


def get_gray_img(path):
    """
    Read the image file from the path and change it to the gray one

    Inputs:
    --> path: path for the image
    Outputs:
    ==> gray_img: the gray image
    """
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def calibration_camera(start, end, flags=0):
    """
    for this implementation I used the code for the opencv source of camera calibration,
    we have two important parameters object_points and image_points,
    object_points are the grid ones which was made by np.mgrid, showing the boxes
    with their depth to be zero,
    img_points are showing the corners which was made by np.findChessboardCorners.
    we give this inputs to cv2.calibrateCamera and get the K matrix at the end.
    the other parameters and this matrix are used to find out whether our approximation
    works well or not.

    Inputs:
    --> start: num of the first image for camera calibration
    --> end: num of the last image for camera calibration
    --> flags: if zero the principal point is in the center of the table,
    if cv2.CALIB_FIX_PRINCIPAL_POINT the principal point is the center of the whole image
    Outputs:
    ==> err: the re-projection error, which gives a good estimation of
    how exact the found parameters are
    ==> mat: the parameters matrix, we named it K in out course
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    object_points = []
    img_points = []

    obj_point = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    obj_point = obj_point * 22  # multiply a to obj_points

    for i in range(start, end + 1):
        img = get_gray_img(resource_path + 'im' + str(i).zfill(2) + '.jpg')
        ret, corners = cv2.findChessboardCorners(img, BOARD_SIZE,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            object_points.append(obj_point)
            # not necessary
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

    _, mat, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, img.shape[::-1], None, None,
                                                     flags=flags)
    mean_error = 0
    for i in range(len(object_points)):
        img_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mat, dist)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        mean_error += error
    err = mean_error / len(object_points)

    return err, mat


err1, m1 = calibration_camera(1, 10)
err2, m2 = calibration_camera(6, 15)
err3, m3 = calibration_camera(11, 20)
err4, m4 = calibration_camera(1, 20)

print(err1, '\n', m1)
print(err2, '\n', m2)
print(err3, '\n', m3)
print(err4, '\n', m4)

# err1, m1 = calibration_camera(1, 10, cv2.CALIB_FIX_PRINCIPAL_POINT)
# err2, m2 = calibration_camera(6, 15, cv2.CALIB_FIX_PRINCIPAL_POINT)
# err3, m3 = calibration_camera(11, 20, cv2.CALIB_FIX_PRINCIPAL_POINT)
# err4, m4 = calibration_camera(1, 20, cv2.CALIB_FIX_PRINCIPAL_POINT)
#
# print(err1, '\n', m1)
# print(err2, '\n', m2)
# print(err3, '\n', m3)
# print(err4, '\n', m4)
