import time
import cv2
from settings import *
import numpy as np


def nothing(x):
    pass


def preprocess_img(frame):
    print("preprocess_img called")
    # dilates the image to increase white color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('hello', blurred)
    cv2.waitKey()
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                   block_size_big, mean_sub_big)
    cv2.imshow('thresh', thresh)
    cv2.waitKey()
    thresh_not = cv2.bitwise_not(thresh)
    cv2.imshow('threshnot', thresh_not)
    cv2.waitKey()
    kernel_close = np.ones((5, 5), np.uint8)
    # closing is dilate and then erosion
    closing = cv2.morphologyEx(thresh_not, cv2.MORPH_CLOSE, kernel_close)
    cv2.imshow('clsoing', closing)
    cv2.waitKey()
    # Delete space between line
    dilate = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_close)
    cv2.imshow('dilate', dilate)
    cv2.waitKey()

    return dilate


def find_corners(contour):
    top_left = [10000, 10000]
    top_right = [0, 10000]
    bottom_right = [0, 0]
    bottom_left = [10000, 0]
    mean_x = np.mean(contour[:, :, 0])
    mean_y = np.mean(contour[:, :, 1])
    for j in range(len(contour)):
        x, y = contour[j][0]
        if x > mean_x:  # On right
            if y > mean_y:  # On bottom
                bottom_right = [x, y]
            else:
                top_right = [x, y]
        else:
            if y > mean_y:  # On bottom
                bottom_left = [x, y]
            else:
                top_left = [x, y]
    return [top_left, top_right, bottom_right, bottom_left]


def get_corners(preprocessed_img):
    # here we find the biggest contours and then we draw contors and find 4 corners and circle the 4 corners
    img_contours = preprocessed_img.copy()
    contours, _ = cv2.findContours(
        preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contours = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    biggest_area = cv2.contourArea(contours[0])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < smallest_area_allow:
            break
        if area > biggest_area / ratio_lim:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, approx_poly_coef * peri, True)
            if len(approx) == 4:
                best_contours.append(approx)
    print("best contours")
    print(best_contours)
    if not best_contours:
        return None
    corners = []
    for best_contour in best_contours:
        corners.append(find_corners(best_contour))
    print("corners")
    print(corners)
    for best_contour in best_contours:
        print("best cont1")
        cv2.drawContours(img_contours, [best_contour], 0, (0, 255, 255), 3)
        for corner in corners:
            print("best cont2")
            for point in corner:
                print("best cont3")
                x, y = point
                cv2.circle(img_contours, (x, y), 10, (255, 0, 0), 3)

    cv2.imshow('bestcntro', img_contours)
    cv2.waitKey()
    return corners


def undistorted_grids(frame, extreme_points):
    print("undistorted_grids")
    undistorted = []
    true_points_grids = []
    transform_matrix = []
    for points_grid in extreme_points:
        points_grid = np.array(points_grid, dtype=np.float32)
        final_pts = np.array(
            [[0, 0], [target_w_grid - 1, 0],
             [target_w_grid - 1, target_h_grid - 1], [0, target_h_grid - 1]],
            dtype=np.float32)
        M = cv2.getPerspectiveTransform(points_grid, final_pts)
        undistorted.append(cv2.warpPerspective(
            frame, M, (target_w_grid, target_h_grid)))
        cv2.imshow("test", undistorted[-1])
        cv2.waitKey()
        true_points_grids.append(points_grid)
        transform_matrix.append(np.linalg.inv(M))
    print("undistorted")
    print(len(undistorted))
    print("true_points_grids")
    print(len(true_points_grids))
    print("transform_matrix")
    print(len(transform_matrix))

    return undistorted, true_points_grids, transform_matrix


def main_grid_detector(frame):

    preprocessed_img = preprocess_img(frame)
    # here we have black and white preprocessed image
    print("getting lines and corners")
    extreme_points = get_corners(
        preprocessed_img)
    print(extreme_points)
    if extreme_points is None:
        print("contours cannot be detected")
        return None, None, None
    grids_final, points_grids, transform_matrix = undistorted_grids(
        frame, extreme_points)
    return grids_final, points_grids, transform_matrix
