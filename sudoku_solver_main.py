import argparse
import os
import sys
from tensorflow.keras.models import load_model
images_extension = [".jpg", ".jpeg", ".png", ".bmp", ".ash"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parser_generation():
    print("checking args")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--i_path",
                        help="Path of the input image",
                        )
    parser.add_argument("-p", "--profile", type=int, choices=[1], default=1)
    parser.add_argument("-mp", "--model_path", type=str,
                        default='model/my_model.h5')
    parser.add_argument("-s", "--save", type=int, choices=[1], default=1)
    parser.add_argument("-d", "--display",
                        help="display output detail", action="store_true")
    args = parser.parse_args()
    return args


def setting_args():  # Function to read parameter settings
    args = parser_generation()
    if args.i_path is None:
        print("image not provided")
        sys.exit(3)
    try:
        model = load_model(args.model_path)
    except OSError:
        print("model not found")
        sys.exit(3)
    return args, model


def main_function():
    args, model = setting_args()
    if args.i_path.endswith(tuple(images_extension)):
        print("calling main_img")
        from main_img import main_img
        main_img(args.i_path, model, args.save)
    else:
        print("\nCannot identify File type\nLeaving...")
        sys.exit(3)


if __name__ == '__main__':
    main_function()
