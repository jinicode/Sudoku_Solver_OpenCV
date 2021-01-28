import cv2
import numpy as np

from settings import *
from Sudoku import verify_viable_grid


def extract_digits(img_grids, model):
    grids = []

    for img in img_grids:
        grids.append(extract_digits_single(img, model
                                           ))
        # cv2.waitKey(0)
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


def extract_digits_single(img, model, display=False):
    h_im, w_im = img.shape[:2]
    im_prepro, gray_enhance = processing_im_grid(img)
    im_contours = img.copy()
    contours, _ = cv2.findContours(
        im_prepro, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_digits = []
    loc_digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        y_true, x_true = y + h / 2, x + w / 2
        if x_true < lim_bord or y_true < lim_bord or x_true > w_im - lim_bord or y_true > h_im - lim_bord:
            continue
        if thresh_h_low < h < thresh_h_high and thresh_area_low < w * h < thresh_area_high:
            if display:
                cv2.drawContours(im_contours, [cnt], -1, (0, 255, 0), 1)
                # print(w*h)
            y1, y2 = y - offset_y, y + h + offset_y
            border_x = max(1, int((y2 - y1 - w) / 2))
            x1, x2 = x - border_x, x + w + border_x
            # digit = im_prepro[y1:y2, x1:x2]
            digit_cut = gray_enhance[max(y1, 0):min(
                y2, h_im), max(x1, 0):min(x2, w_im)]
            _, digit_thresh = cv2.threshold(digit_cut,
                                            0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # digit_w_border = cv2.copyMakeBorder(digit, l_border, l_border, l_border, l_border,
            #                                     cv2.BORDER_CONSTANT, None, 255)
            img_digits.append(cv2.resize(digit_thresh, (28, 28),
                                         interpolation=cv2.INTER_NEAREST).reshape(28, 28, 1))
            loc_digits.append([y_true, x_true])
    print("img digits")
    cv2.imshow("im_contours", im_contours)
    cv2.waitKey()
    img_digits_np = np.array(img_digits) / 255.0
    preds_proba = model.predict(img_digits_np)

    # preds = np.argmax(preds_proba, axis=1) + 1
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
    # for i in range(len(preds)):
    #     y, x = loc_digits[i]
    #     cv2.imshow('pred_{} - {:.6f} - x/y : {}/{}'.format(preds[i], 100 * max(preds_proba[i]), int(x), int(y)),
    #                img_digits[i])
    #     cv2.waitKey()
    if nbr_digits_extracted < min_digits_extracted:

        cv2.imshow("im_contours", im_contours)
        cv2.waitKey()
        return None
    grid = fill_grid(preds, loc_digits, h_im, w_im)
    if verify_viable_grid(grid):
        print("verify_viable_grid")
        return grid
    else:
        return None
