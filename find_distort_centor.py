import cv2
import os
import numpy as np


def pixel_xy_to_center_xy(xys, center=np.array([0.5, 0.5])):
    xys = np.array(xys)
    float_xys = xys / np.array([3584, 1896])
    center_float_xys = float_xys - center
    return center_float_xys


def center_xy_to_pixel_xy(xys):
    xys = np.array(xys)
    xys = xys + np.array([0.5, 0.5])
    xys = xys * np.array([3584, 1896])
    return xys


def image_to_biValue(image):
    image = np.where(image > 0, 255, 0)
    image_b, image_g, image_r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    image = (image_b & image_g & image_r)[:, :, np.newaxis]
    image = np.concatenate((image, image, image), axis=2)
    biValue = image.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    biValue = cv2.morphologyEx(biValue, cv2.MORPH_OPEN, kernel, iterations=5)
    biValue = cv2.morphologyEx(biValue, cv2.MORPH_CLOSE, kernel, iterations=3)
    biValue = np.where(biValue > 0, 255, 0).astype(np.uint8)
    return biValue


def scan_one_column(col):
    indices = np.nonzero(col)
    start = np.min(indices)
    end = np.max(indices)
    length = end - start
    return start, end, length


def step_scan_column(image):
    delta = 520
    upper_points = list()
    lower_points = list()
    for x in range(delta, 3584 - delta, 10):
        col = image[:, x, 0]
        start, end, length = scan_one_column(col)
        mid = (start + end) // 2

        upper_points.append([x, start])
        lower_points.append([x, end])
    return upper_points, lower_points


def find_the_upper_line(u_points):
    y_upper = -0.33808017
    line_points = np.ones_like(u_points) * y_upper
    line_points[:, 0] = u_points[:, 0] * y_upper / u_points[:, 1]
    return line_points


def find_the_lower_line(u_points):
    y_lower = 0.25856616
    line_points = np.ones_like(u_points) * y_lower
    line_points[:, 0] = u_points[:, 0] * y_lower / u_points[:, 1]
    return line_points


def calculate_r(xys):
    diff_xy = xys
    diff_xy_2 = np.power(diff_xy, 2)
    r_2 = diff_xy_2[:, 0] + diff_xy_2[:, 1]
    r = np.sqrt(r_2)
    return r


if __name__ == '__main__':
    image = cv2.imread("001.jpg")
    im_h, im_w, im_c = image.shape
    print("image.shape", im_h, im_w)

    biValue_depict = image_to_biValue(image)
    biValue_original = biValue_depict.copy()
    upper_U_points, lower_U_points = step_scan_column(biValue_original)

    upper_U_points = np.array(upper_U_points)
    lower_U_points = np.array(lower_U_points)

    for y in np.arange(0.47, 0.472, 0.00001):
        print("y:{}\t".format(y), end="")
        upper_U_points_center_float = pixel_xy_to_center_xy(upper_U_points, center=np.array([0.5, y]))
        lower_U_points_center_float = pixel_xy_to_center_xy(lower_U_points, center=np.array([0.5, y]))

        upper_line_points = find_the_upper_line(upper_U_points_center_float)
        lower_line_points = find_the_lower_line(lower_U_points_center_float)

        upper_distort_r = calculate_r(upper_line_points)
        lower_distort_r = calculate_r(lower_line_points)

        upper_ideal_r = calculate_r(upper_U_points_center_float)
        lower_ideal_r = calculate_r(lower_U_points_center_float)

        distort_r = upper_distort_r
        ideal_r = upper_ideal_r

        mat_B = distort_r / ideal_r

        ideal_r = ideal_r[:, np.newaxis]
        mat_A = np.concatenate(
            (np.ones_like(ideal_r), np.power(ideal_r, 2), np.power(ideal_r, 4), np.power(ideal_r, 6)), axis=-1)
        C1 = np.linalg.pinv((mat_A.T @ mat_A)) @ mat_A.T @ mat_B
        # print(list(C1))

        distort_r = lower_distort_r
        ideal_r = lower_ideal_r

        mat_B = distort_r / ideal_r

        ideal_r = ideal_r[:, np.newaxis]
        mat_A = np.concatenate(
            (np.ones_like(ideal_r), np.power(ideal_r, 2), np.power(ideal_r, 4), np.power(ideal_r, 6)), axis=-1)
        C2 = np.linalg.pinv((mat_A.T @ mat_A)) @ mat_A.T @ mat_B
        print(np.max(np.abs(C2 - C1)), (C2 - C1))

    # for line_xy, U_xy in zip(center_xy_to_pixel_xy(upper_line_points), upper_U_points):
    #     ux, uy = U_xy
    #     lx, ly = line_xy
    #     cv2.circle(biValue_depict, (int(ux), int(uy)), 2, (255, 0, 0), 3)
    #     cv2.circle(biValue_depict, (int(lx), int(ly)), 2, (0, 255, 0), 3)
    #     cv2.line(biValue_depict, (3584 // 2, 1896 // 2), (int(ux), int(uy)), (0, 0, 255), 1, 8)
    #     cv2.imshow(",", biValue_depict.astype(np.uint8))
    #     cv2.waitKey()
    #
    # for line_xy, U_xy in zip(center_xy_to_pixel_xy(lower_line_points), lower_U_points):
    #     ux, uy = U_xy
    #     lx, ly = line_xy
    #     cv2.circle(biValue_depict, (int(ux), int(uy)), 2, (255, 0, 0), 3)
    #     cv2.circle(biValue_depict, (int(lx), int(ly)), 2, (0, 255, 0), 3)
    #     cv2.line(biValue_depict, (3584 // 2, 1896 // 2), (int(ux), int(uy)), (0, 0, 255), 1, 8)
    #     cv2.imshow(",", biValue_depict.astype(np.uint8))
    #     cv2.waitKey()
