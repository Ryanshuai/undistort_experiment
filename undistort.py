import cv2
import os
import numpy as np

image_path_list = list()
for root, dirs, files in os.walk("original_images"):
    for file in files:
        file_path = os.path.join(root, file)
        if file_path.endswith(".png"):
            image_path_list.append(file_path)

cameraMatrix = np.array([[1258.18, -0.0541007, 913.243], [0, 1258.07, 400], [0, 0, 1]])
# cameraMatrix = np.array([[1258.18, 0, 913.243], [0, 1258.07, 638.875], [0, 0, 1]])

image_path = "original_images/980591041.png"
# image_path = "original_images/001.png"
src = cv2.imread(image_path)

k12 = np.array([-0.198351, 0])

distCoeffs = np.array([k12[0], k12[1], 9.93501e-05, -5.2079e-05])

# cameraMatrix = np.array([[1258.18, 0.0541007, 913.243], [0, 1258.07, 638.875], [0, 0, 1]])
# distCoeffs = np.array([9.93501e-05, -5.2079e-05, 0, 0, 1.68351, 0.181557, 2.51051, 1.03601])
#

# cameraMatrix = None
# distCoeffs = np.array([9.93501e-05, -5.2079e-05, 0, 0, 1.68351, 0.181557, 2.51051, 1.03601])
# distCoeffs = np.array([1.68351, 0.181557, 9.93501e-05, -5.2079e-05, 2.51051, 1.03601, 0, 0])
# distCoeffs = np.array([1.68351, 0.181557, 9.93501e-05, -5.2079e-05, 2.51051, 1.03601, 2.51051, 0])
# distCoeffs = None

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (1824, 948), 1, (3584, 1896))

mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newcameramtx, (3584, 1896), 5)
dst = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)

# cv2.imwrite(
#     image_path.replace("original_images", "undistorted_images").replace("980591041", "{}_{}".format(i, j)),
#     dst)write(
dst = cv2.resize(dst, (2000, 1000))

cv2.imshow("", dst)
cv2.waitKey()
