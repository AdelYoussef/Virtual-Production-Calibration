import cv2
import numpy as np
from VP_Calib import VP_Calib


def main():
    Camera_Type = "VP"
    id = 0
    camera_name  = "A6400"
    calibrate = VP_Calib("config.yaml")

    image_folder = calibrate.Mono_Calibration_Capture(Camera_Type, id, camera_name)

    calibrate.Mono_Calib(image_folder)


if __name__ == '__main__':
    main()