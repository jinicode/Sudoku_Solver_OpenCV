import time
import cv2
from settings import *
import numpy as np


def nothing(x):
    pass


def preprocess_im(frame):
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

    # kernel_open = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh_not, cv2.MORPH_OPEN, kernel_open)  # Denoise
    kernel_close = np.ones((5, 5), np.uint8)
    # Delete space between line
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
    # contour_x = sorted(contour,key = lambda c:c[0][0])
    # contour_y = sorted(contour,key = lambda c:c[0][1])
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


def look_for_corners(img_lines):

    img_contours = cv2.cvtColor(img_lines.copy(), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        cv2.drawContours(img_contours, [best_contour], 0, (0, 0, 255), 3)
        for corner in corners:
            print("best cont2")
            for point in corner:
                print("best cont3")
                x, y = point
                cv2.circle(img_contours, (x, y), 10, (255, 0, 0), 3)

    cv2.imshow('bestcntro', img_contours)
    cv2.waitKey()
    return corners


def get_lines_and_corners(img, edges, use_hough=False):
    img_lines = edges.copy()
    return look_for_corners(img_lines)


def undistorted_grids(frame, points_grids, ratio):
    print("undistorted_grids")
    undistorted = []
    true_points_grids = []
    transfo_matrix = []
    for points_grid in points_grids:
        points_grid = np.array(points_grid, dtype=np.float32) * ratio
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
        transfo_matrix.append(np.linalg.inv(M))
    print("undistorted")
    print(undistorted)
    print("true_points_grids")
    print(true_points_grids)
    print("transfo_matrix")
    print(transfo_matrix)
    return undistorted, true_points_grids, transfo_matrix


def main_grid_detector_img(frame, resized=True, use_hough=False):
    ratio = frame.shape[0] / frame.shape[0]
    prepro_im_edges = preprocess_im(frame)

    extreme_points_biased = get_lines_and_corners(
        frame.copy(), prepro_im_edges, use_hough=use_hough)
    print(extreme_points_biased)
    if extreme_points_biased is None:
        print("contours cannot be detected")
        return None, None, None
    grids_final, points_grids, transfo_matrix = undistorted_grids(
        frame, extreme_points_biased, ratio)
    return grids_final, points_grids, transfo_matrix
