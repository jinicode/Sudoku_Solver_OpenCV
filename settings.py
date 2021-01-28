import os

import tensorflow
from cv2 import FONT_HERSHEY_SIMPLEX
from numpy import pi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

# ----PREPRO BIG IMAGE----#
block_size_big = 41
mean_sub_big = 15

# ----GRID COUNTOURS----#
ratio_lim = 2
smallest_area_allow = 75000
approx_poly_coef = 0.1
target_h_grid, target_w_grid = 450, 450

# ----PREPRO IMAGE DIGIT----#
block_size_grid = 29  # 43
mean_sub_grid = 25


# ----DIGITS EXTRACTION----#
thresh_conf_cnn = 0.98
thresh_conf_cnn_high = 0.99


lim_bord = 10
thresh_h_low = 15
thresh_h_high = 50
thresh_area_low = 210
thresh_area_high = 900
l_case = 45
l_border = 1
offset_y = 2
min_digits_extracted = 13
