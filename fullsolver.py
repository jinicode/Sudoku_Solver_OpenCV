import argparse
import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from settings import *
save_folder = "images_save/"
images_extension = [".jpg", ".jpeg", ".png"]
font = cv2.FONT_HERSHEY_SIMPLEX


def recreate_img_filled(frame, im_grids, pointGrid, transform_matrix, ratio=None):
    target_h, target_w = frame.shape[:2]
    if ratio:
        im_final = frame.copy()
        for i, points_grid in enumerate(pointGrid):
            pointGrid[i] = np.array(points_grid, dtype=np.float32) * ratio
    else:
        im_final = frame
    new_im = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
    for im_grid, points_grid, transform_matrix in zip(im_grids, pointGrid, transform_matrix):
        if im_grid is None:
            for point in points_grid:
                x, y = point
                cv2.circle(new_im, (x, y), 6, (255, 0, 0), 3)
        else:
            if ratio:
                init_pts = np.array([[0, 0], [target_w_grid - 1, 0], [target_w_grid - 1, target_h_grid - 1],
                                     [0, target_h_grid - 1]], dtype=np.float32)
                transform_matrix = cv2.getPerspectiveTransform(
                    init_pts, points_grid)
            new_im = cv2.add(new_im, cv2.warpPerspective(
                im_grid, transform_matrix, (target_w, target_h)))
    _, mask = cv2.threshold(cv2.cvtColor(
        new_im, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    im_final = cv2.bitwise_and(im_final, im_final, mask=cv2.bitwise_not(mask))
    im_final = cv2.add(im_final, new_im)
    return im_final


def write_solved_grids(frames, grids_matrix, solved_grids):
    ims_filled_grid = []
    for frame, grid_init, solved_grid in zip(frames, grids_matrix, solved_grids):
        if solved_grid is None:
            ims_filled_grid.append(None)
            continue
        im_filled_grid = np.zeros_like(frame)
        h_im, w_im = frame.shape[:2]
        for y in range(9):
            for x in range(9):
                if grid_init[y, x] != 0:
                    continue
                true_y, true_x = int((y + 0.5) * h_im /
                                     9), int((x + 0.5) * w_im / 9)
                digit = str(solved_grid[y, x])
                (text_width, text_height) = cv2.getTextSize(
                    digit, font, fontScale=1.2, thickness=2)[0]
                cv2.putText(im_filled_grid, digit,
                            (true_x - int(text_width / 2),
                             true_y + int(text_height / 2)),
                            font, 1.2, (0, 3, 0), 2 * 3)
                cv2.putText(im_filled_grid, digit,
                            (true_x - int(text_width / 2),
                             true_y + int(text_height / 2)),
                            font, 1.2, (255, 0, 0), 2)
        ims_filled_grid.append(im_filled_grid)
    return ims_filled_grid


def is_affected(x1, y1, x2, y2):
    if x1 == x2:
        return True
    if y1 == y2:
        return True
    if x1 // 3 == x2 // 3 and y1 // 3 == y2 // 3:
        return True
    return False


class Sudoku:
    def __init__(self, sudo=None, grid=None):
        self.possible_values_grid = np.empty((9, 9), dtype=list)
        if sudo is None:
            self.grid = np.zeros((9, 9), dtype=int)
            self.count_possible_grid = np.zeros((9, 9), dtype=int)
            self.init_sudo(grid)
        else:
            self.grid = sudo.grid.copy()
            for y in range(9):
                for x in range(9):
                    self.possible_values_grid[y,
                                              x] = sudo.possible_values_grid[y, x].copy()
            self.count_possible_grid = sudo.count_possible_grid.copy()

    def __str__(self):
        string = "-" * 18
        for y in range(9):
            string += "\n|"
            for x in range(9):
                string += str(self.grid[y, x]) + "|"
        string += "\n"
        string += "-" * 18

        return string

    def apply_hypothesis_value(self, x, y, value):
        self.grid[y, x] = value
        self.possible_values_grid[y, x] = []
        self.count_possible_grid[y, x] = 0

        for y2 in range(9):
            for x2 in range(9):
                if is_affected(x, y, x2, y2) and self.grid[y2, x2] == 0:
                    list_possible_values = self.possible_values_grid[y2, x2]
                    if value in list_possible_values:
                        list_possible_values.remove(value)
                        new_len = len(list_possible_values)
                        self.count_possible_grid[y2, x2] = new_len

    def init_sudo(self, grid):
        for y in range(9):
            for x in range(9):
                value = grid[y][x]
                self.grid[y, x] = value
                if value == 0:
                    self.possible_values_grid[y, x] = [
                        1, 2, 3, 4, 5, 6, 7, 8, 9]
                    self.count_possible_grid[y, x] = 9
                else:
                    self.possible_values_grid[y, x] = []
        self.get_possible_values()

    def is_filled(self):
        return 0 not in self.grid

    def get_possible_values(self):
        for y in range(9):
            for x in range(9):
                if self.grid[y, x] != 0:

                    continue
                possible_values = self.get_1_possible_values(x, y)
                self.possible_values_grid[y, x] = possible_values
                self.count_possible_grid[y, x] = len(possible_values)

    def get_1_possible_values(self, x, y):
        possible_values = self.possible_values_grid[y, x]
        self.check_line(y, possible_values)
        self.check_column(x, possible_values)
        self.check_square(x, y, possible_values)
        return possible_values

    def check_line(self, y, possible_values):
        line = self.grid[y, :]
        for value in reversed(possible_values):
            if value in line:
                possible_values.remove(value)

    def check_column(self, x, possible_values):
        column = self.grid[:, x]
        for value in reversed(possible_values):
            if value in column:
                possible_values.remove(value)

    def check_square(self, x, y, possible_values):
        x1 = 3 * (x // 3)
        y1 = 3 * (y // 3)
        x2, y2 = x1 + 3, y1 + 3
        square = self.grid[y1:y2, x1:x2]
        for value in reversed(possible_values):
            if value in square:
                possible_values.remove(value)

    def apply_and_actualize(self, x, y, value):
        self.grid[y, x] = value
        self.possible_values_grid[y, x] = []
        self.count_possible_grid[y, x] = 0

        for y2 in range(9):
            for x2 in range(9):
                if is_affected(x, y, x2, y2) and self.grid[y2, x2] == 0:
                    list_possible_values = self.possible_values_grid[y2, x2]
                    if value in list_possible_values:
                        list_possible_values.remove(value)
                        new_len = len(list_possible_values)
                        if new_len == 0:
                            return False
                        self.count_possible_grid[y2, x2] = new_len
        return True

    def apply_unique_possibilities(self):
        for y in range(9):
            for x in range(9):
                if self.grid[y, x] == 0 and self.count_possible_grid[y, x] == 1:
                    value = self.possible_values_grid[y, x][0]
                    if not self.apply_and_actualize(x, y, value):
                        return False

        return True

    def verify_new_result(self, my_zip):
        for x, y in my_zip:
            val = self.grid[y, x]
            self.grid[y, x] = 0
            line = self.grid[y, :]
            column = self.grid[:, x]
            x1 = 3 * (x // 3)
            y1 = 3 * (y // 3)
            x2, y2 = x1 + 3, y1 + 3
            square = self.grid[y1:y2, x1:x2]
            test = val in line or val in column or val in square
            self.grid[y, x] = val
            if test:
                return False

        return True

    def should_make_hypothesis(self):
        return 1 not in self.count_possible_grid

    def best_hypothesis(self):
        count_less_options = 9
        best_x = 0
        best_y = 0
        for y in range(9):
            for x in range(9):
                if self.grid[y, x] != 0:
                    continue
                if self.count_possible_grid[y, x] == 2:
                    return x, y, self.possible_values_grid[y, x]
                elif self.count_possible_grid[y, x] < count_less_options:
                    best_x, best_y = x, y
                    count_less_options = self.count_possible_grid[y, x]
                    if count_less_options == 0:
                        return None, None, []

        return best_x, best_y, self.possible_values_grid[best_y, best_x]

    def verify_result(self):
        for y in range(9):
            for x in range(9):
                grid = self.grid.copy()
                grid[y, x] = 0
                line = grid[y, :]
                column = grid[:, x]
                x1 = 3 * (x // 3)
                y1 = 3 * (y // 3)
                x2, y2 = x1 + 3, y1 + 3
                square = grid[y1:y2, x1:x2]
                val = self.grid[y, x]
                if val in line or val in column or val in square:
                    return False

        return True


def solve_grid(sudo):
    while not sudo.is_filled():
        if sudo.should_make_hypothesis():
            x, y, possible_values_hyp = sudo.best_hypothesis()
            if not possible_values_hyp:
                return False, None
            for val in possible_values_hyp:
                new_sudo = Sudoku(sudo=sudo)
                new_sudo.apply_hypothesis_value(x, y, val)
                ret, solved_sudo = solve_grid(new_sudo)
                if ret:
                    return True, solved_sudo
                else:
                    del new_sudo
            return False, None
        else:
            ret = sudo.apply_unique_possibilities()
            if ret is False:

                del sudo
                return False, None
    return True, sudo


def main_solve_grid(grid):
    if grid is None:
        return None
    sudo = Sudoku(grid=grid)
    ret, finished_sudo = solve_grid(sudo)
    if ret:
        return finished_sudo.grid
    else:
        return None


def solve_grids(grids):
    finished_grids = []
    for grid in grids:
        finished_grids.append(main_solve_grid(grid))
    if all(elem is None for elem in finished_grids):
        return None
    return finished_grids


def extract_digits_single(img, model, display=False):
    h_im, w_im = img.shape[:2]
    im_prepro, gray_enhance = processing_im_grid(img)
    im_contours = img.copy()
    contours, _ = cv2.findContours(
        im_prepro, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_digits = []
    loc_digits = []
    i = 1
    j = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        y_true, x_true = y + h / 2, x + w / 2
        if x_true < lim_bord or y_true < lim_bord or x_true > w_im - lim_bord or y_true > h_im - lim_bord:
            print("i", i)
            i = i+1
            continue
        if thresh_h_low < h < thresh_h_high and thresh_area_low < w * h < thresh_area_high:
            # if there is digit inside a box grid
            print("j", j)
            j = j+1
            if True:
                cv2.drawContours(im_contours, [cnt], -1, (0, 255, 0), 1)
            y1, y2 = y - offset_y, y + h + offset_y
            border_x = max(1, int((y2 - y1 - w) / 2))
            x1, x2 = x - border_x, x + w + border_x

            digit_cut = gray_enhance[max(y1, 0):min(
                y2, h_im), max(x1, 0):min(x2, w_im)]
            _, digit_thresh = cv2.threshold(digit_cut,
                                            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_digits.append(cv2.resize(digit_thresh, (28, 28),
                                         interpolation=cv2.INTER_NEAREST).reshape(28, 28, 1))
            loc_digits.append([y_true, x_true])
    cv2.imshow("im_contours", im_contours)
    cv2.waitKey()
    print("img_digits", img_digits[0][0])
    img_digits_np = np.array(img_digits) / 255.0
    print("img_digits_np", img_digits_np[0][0])
    preds_proba = model.predict(img_digits_np)
    print("preds_proba", preds_proba)
    preds = []
    nbr_digits_extracted = 0
    adapted_thresh_conf_cnn = thresh_conf_cnn
    for pred_proba in preds_proba:
        arg_max = np.argmax(pred_proba)
        if pred_proba[arg_max] > adapted_thresh_conf_cnn and arg_max < 9:
            preds.append(arg_max + 1)
            nbr_digits_extracted += 1
        else:
            preds.append(-1)
    if nbr_digits_extracted < min_digits_extracted:

        cv2.imshow("im_contours", im_contours)
        cv2.waitKey()
        return None
    print("preds")
    print(preds)
    grid = fill_grid(preds, loc_digits, h_im, w_im)
    return grid


def extract_digits(img_grids, model):
    print("Extracting")
    print(img_grids)
    grids = []
    for img in img_grids:
        grids.append(extract_digits_single(img, model))
    print("grids")
    print(grids)
    return grids


def processing_im_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_enhance = (gray - gray.min()) * int(255 / (gray.max() - gray.min()))
    blurred = cv2.GaussianBlur(gray_enhance, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, block_size_grid, mean_sub_grid)
    return thresh, gray_enhance


def fill_grid(preds, loc_digits, h_im, w_im):
    grid = np.zeros((9, 9), dtype=int)
    for pred, loc in zip(preds, loc_digits):
        if pred > 0:
            y, x = loc
            true_y = int(9 * y // h_im)
            true_x = int(9 * x // w_im)
            grid[true_y, true_x] = pred

    return grid


def undistorted_grids(frame, extreme_points):
    undistorted = []
    true_pointGrid = []
    transform_matrix = []
    print("extreme_points")
    print(extreme_points)
    for points_grid in extreme_points:
        print("points grid")
        print(points_grid)
        points_grid = np.array(points_grid, dtype=np.float32)
        print(points_grid)
        final_pts = np.array(
            [[0, 0], [target_w_grid - 1, 0],
             [target_w_grid - 1, target_h_grid - 1], [0, target_h_grid - 1]],
            dtype=np.float32)
        print("final_pts")
        print(final_pts)
        M = cv2.getPerspectiveTransform(points_grid, final_pts)
        print("M")
        print(M)
        undistorted.append(cv2.warpPerspective(
            frame, M, (target_w_grid, target_h_grid)))
        print("undistorted")
        print(undistorted[-1])
        cv2.imshow("test", undistorted[-1])
        cv2.waitKey()
        true_pointGrid.append(points_grid)
        transform_matrix.append(np.linalg.inv(M))
    return undistorted, true_pointGrid, transform_matrix


def find_corners(contour):
    top_left = [10000, 10000]
    top_right = [0, 10000]
    bottom_right = [0, 0]
    bottom_left = [10000, 0]
    mean_x = np.mean(contour[:, :, 0])
    mean_y = np.mean(contour[:, :, 1])
    for j in range(len(contour)):
        x, y = contour[j][0]
        if x > mean_x:
            if y > mean_y:
                bottom_right = [x, y]
            else:
                top_right = [x, y]
        else:
            if y > mean_y:
                bottom_left = [x, y]
            else:
                top_left = [x, y]
    return [top_left, top_right, bottom_right, bottom_left]


def get_corners(preprocessed_img):
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
    if not best_contours:
        return None
    print("\n best_contours")
    print(best_contours)
    corners = []
    for best_contour in best_contours:
        print("\n best_contour")
        print(best_contour)
        corners.append(find_corners(best_contour))
    print("\n cornrs")
    print(corners)
    i = 1
    j = 1
    for best_contour in best_contours:
        print("best contour", i)
        print(best_contour)
        i = i+1

        cv2.drawContours(img_contours, [best_contour], 0, (0, 0, 255), 3)
        for corner in corners:
            print("corner", j)

            print(corner)
            for point in corner:
                print(point, j)
                j = j+1
                x, y = point
                cv2.circle(img_contours, (x, y), 10, (255, 0, 0), 3)
    cv2.imshow('bestcntro', img_contours)
    cv2.waitKey()
    return corners


def preprocess_img(frame):
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
    closing = cv2.morphologyEx(thresh_not, cv2.MORPH_CLOSE, kernel_close)
    cv2.imshow('clsoing', closing)
    cv2.waitKey()
    dilate = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_close)
    cv2.imshow('dilate', dilate)
    cv2.waitKey()
    return dilate


def grid_detector(frame):
    preprocessed_img = preprocess_img(frame)
    # get 4 most extreme points by contours
    extreme_points = get_corners(
        preprocessed_img)
    if extreme_points is None:
        return None, None, None
    # undistorted_grids will return the image in small with only useful portion
    grids_final, pointGrid, transform_matrix = undistorted_grids(
        frame, extreme_points)
    return grids_final, pointGrid, transform_matrix


def main_img(im_path, model, save=False):
    frame = cv2.imread(im_path)
    if frame is None:
        sys.exit(3)
    imgGridsFinal, pointGrid, transform_matrix = grid_detector(
        frame)
    if imgGridsFinal is None:
        sys.exit(3)
    grids_matrix = extract_digits(imgGridsFinal, model)
    if all(elem is None for elem in grids_matrix):
        sys.exit(3)
    print("grids_matrix", grids_matrix)
    grids_solved = solve_grids(grids_matrix)
    if grids_solved is None:
        cv2.imshow('grid_extract', imgGridsFinal[0])
        cv2.imwrite(save_folder + os.path.splitext(os.path.basename(im_path))
                    [0] + "_failed.jpg", imgGridsFinal[0])
        cv2.waitKey()
        sys.exit(3)
    print("grids_solved", grids_solved)
    ims_filled_grid = write_solved_grids(
        imgGridsFinal, grids_matrix, grids_solved)
    print("ims_filled_grid", ims_filled_grid)
    im_final = recreate_img_filled(
        frame, ims_filled_grid, pointGrid, transform_matrix)
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        cv2.imwrite(save_folder + os.path.splitext(os.path.basename(im_path))
                    [0] + "_solved.jpg", im_final)
    if len(ims_filled_grid) == 1:
        cv2.imshow('imgabc', frame)
        cv2.imshow('grid_extract123', imgGridsFinal[0])
        cv2.imshow('grid_filled123', ims_filled_grid[0])
    cv2.imshow('im_final123', im_final)
    cv2.waitKey()


def setting_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--i_path",
                        help="Path of the input image",
                        )
    parser.add_argument("-mp", "--model_path", type=str,
                        default='model/my_model.h5')
    parser.add_argument("-s", "--save", type=int, choices=[1], default=1)
    args = parser.parse_args()
    if args.i_path is None:
        sys.exit(3)
    try:
        model = load_model(args.model_path)
    except OSError:
        sys.exit(3)
    return args, model


def main_function():
    args, model = setting_args()
    if args.i_path.endswith(tuple(images_extension)):
        main_img(args.i_path, model, args.save)
    else:
        sys.exit(3)


if __name__ == '__main__':
    main_function()
