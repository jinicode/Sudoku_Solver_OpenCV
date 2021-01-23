import sys
import time
import cv2
from tensorflow.keras.models import load_model
from settings import *
from extract_digits import extract_digits
from grid_detector import grid_detector
from grid_solver import solve_grids
from new_img_generator import *
from os import walk

save_folder = "images_save/"


def main_img(im_path, model, save=False):
    init = time.time()
    frame = cv2.imread(im_path)
    init0 = time.time()
    if frame is None:
        print("This path doesn't lead to a frame")
        sys.exit(3)
    print("")
    imgGridsinal, pointGrid, transform_matrix = grid_detector(
        frame)
    print("imgGridsinal")
    print(imgGridsinal)
    print(len(imgGridsinal[0]))
    print("pointGrid")
    print(len(pointGrid[0]))
    print("transform_matrix")
    print(len(transform_matrix[0]))
    found_grid_time = time.time()
    if imgGridsinal is None:
        print("No grid found")
        sys.exit(3)
    print("Grid(s) found")
    grids_matrix = extract_digits(imgGridsinal, model
                                  )
    if all(elem is None for elem in grids_matrix):
        print("Failed during digits extraction")
        sys.exit(3)
    print("Extraction done")
    extract_time = time.time()
    grids_solved = solve_grids(grids_matrix)
    print("Solving done")
    if grids_solved is None:
        print("grids not solved ")
        print(grids_matrix)
        cv2.imshow('grid_extract', imgGridsinal[0])
        cv2.imwrite(save_folder + os.path.splitext(os.path.basename(im_path))
                    [0] + "_failed.jpg", imgGridsinal[0])
        cv2.waitKey()
        sys.exit(3)

    solve_time = time.time()
    ims_filled_grid = write_solved_grids(
        imgGridsinal, grids_matrix, grids_solved)
    im_final = recreate_img_filled(
        frame, ims_filled_grid, pointGrid, transform_matrix)
    final_time = time.time()
    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        cv2.imwrite(save_folder + os.path.splitext(os.path.basename(im_path))
                    [0] + "_solved.jpg", im_final)
    total_time = final_time - init

    load_time = init0 - init
    print("Load Image\t\t\t{:03.1f}% - {:05.2f}ms".format(100 *
                                                          load_time / total_time, 1000 * load_time))
    founding_time = found_grid_time - init0
    print(
        "Grid Research \t\t{:03.1f}% - {:05.2f}ms".format(100 * founding_time / total_time, 1000 * founding_time))
    extraction_duration = extract_time - found_grid_time
    print(
        "Digits Extraction \t{:03.1f}% - {:05.2f}ms".format(100 * extraction_duration / total_time,
                                                            1000 * extraction_duration))
    solving_duration = solve_time - extract_time
    print(
        "Grid Solving \t\t{:03.1f}% - {:05.2f}ms".format(100 * solving_duration / total_time, 1000 * solving_duration))
    recreation_duration = final_time - solve_time
    print(
        "Image recreation \t{:03.1f}% - {:05.2f}ms".format(100 * recreation_duration / total_time,
                                                           1000 * recreation_duration))
    print("PROCESS DURATION \t{:.2f}s".format(final_time - init0))
    print("EVERYTHING DONE \t{:.2f}s".format(total_time))
    # print(grid)
    # print(grid_solved)
    if len(ims_filled_grid) == 1:
        cv2.imshow('imgabc', frame)
        cv2.imshow('grid_extract123', imgGridsinal[0])
        cv2.imshow('grid_filled123', ims_filled_grid[0])
    cv2.imshow('im_final123', im_final)
    cv2.waitKey()
