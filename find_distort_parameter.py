import cv2
import os
import numpy as np


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
    length = 0
    start = 0
    end = len(col)
    for idx, (upper, lower) in enumerate(zip(col[:-1], col[1:])):
        if upper == 0 and lower == 255:
            start = idx
        if upper == 255 and lower == 0:
            end = idx
        if upper == 255:
            length += 1

    return start, end, length


def step_scan_column(image, depict):
    delta = 520
    upper_points = list()
    lower_points = list()
    for x in range(delta, 3584 - delta, 10):
        col = image[:, x, 0]
        start, end, length = scan_one_column(col)
        mid = (start + end) // 2

        # cv2.circle(depict, (x, start), 2, (255, 0, 0), 3)
        # cv2.circle(depict, (x, mid), 2, (0, 255, 0), 3)
        # cv2.circle(depict, (x, end), 2, (0, 0, 255), 3)

        upper_points.append([x, start])
        lower_points.append([x, end])
    return upper_points, lower_points


def find_center_1(image, depict):
    im_h, im_w, im_c = image.shape
    col = image[:, im_w // 2, 0]
    start, end, length = scan_one_column(col)
    mid = (start + end) // 2
    cv2.circle(depict, (im_w // 2, mid), 5, (255, 0, 0), 2)
    c_y, c_x = mid, im_w // 2
    return c_x, c_y


def find_the_intersection_with_image_edge(center_point, point2):
    x_c, y_c = center_point
    x2, y2 = point2
    if y_c > y2:  # upper points
        y_edge = 0
        k = -(y2 - y_c) / (x2 - x_c)  # minus because image y is inverse
        x_diff = y_c / k
        x_edge = x_c + x_diff

    else:  # lower points
        image_height = 1896
        y_edge = image_height  # image height
        k = (y2 - y_c) / (x2 - x_c)
        x_diff = (image_height - y_c) / k
        x_edge = x_c + x_diff

    return x_edge, y_edge


def calculate_r_2(center_xy, xys):
    diff_xy = xys - np.array(center_xy)
    diff_xy = diff_xy / 1896
    diff_xy_2 = np.power(diff_xy, 2)
    r_2 = diff_xy_2[:, 0] + diff_xy_2[:, 1]
    return r_2


def check_mapping(image, center_xy, xys):
    r_2 = calculate_r_2(center_xy, xys)
    scale = r_2


if __name__ == '__main__':
    image = cv2.imread("001.jpg")
    im_h, im_w, im_c = image.shape
    print("image.shape", im_h, im_w)

    biValue_depict = image_to_biValue(image)
    biValue_original = biValue_depict.copy()
    c_x, c_y = find_center_1(biValue_original, biValue_depict)
    upper_points, lower_points = step_scan_column(biValue_original, biValue_depict)

    # show the points pare found
    for upper_point in lower_points:
        x, y = upper_point
        x_edge, y_edge = find_the_intersection_with_image_edge((c_x, c_y), (x, y))

        cv2.circle(biValue_depict, (int(x), int(y)), 2, (255, 0, 0), 3)
        cv2.circle(biValue_depict, (int(x_edge), int(y_edge)), 2, (0, 255, 0), 3)
        cv2.line(biValue_depict, (c_x, c_y), (int(x_edge), int(y_edge)), (0, 0, 255), 1, 8)

        cv2.imshow(",", biValue_depict.astype(np.uint8))
        cv2.waitKey()

    points_in_normal = np.array(upper_points + lower_points)
    points_in_distort = [find_the_intersection_with_image_edge((c_x, c_y), point) for point in points_in_normal]
    points_in_distort = np.array(points_in_distort)

    r_2_in_distort = calculate_r_2([c_x, c_y], points_in_distort)
    r_2_in_normal = calculate_r_2([c_x, c_y], points_in_normal)

    r_in_distort = np.sqrt(r_2_in_distort)
    r_in_normal = np.sqrt(r_2_in_normal)

    # mat_B = r_in_normal / r_in_distort
    mat_B = r_in_distort / r_in_normal

    r_2_in_distort = r_2_in_distort[:, np.newaxis]
    mat_A = np.concatenate(
        (r_2_in_distort, np.power(r_2_in_distort, 2) , np.power(r_2_in_distort, 3)),
        axis=-1)

    C = np.linalg.pinv((mat_A.T @ mat_A)) @ mat_A.T @ mat_B
    print(C)


