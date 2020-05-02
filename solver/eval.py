import sys

import cv2
import numpy as np

sys.path.append("..")
from generator.Generator import Generator, chunker
from generator import base_numbers
from solver.solver_models import get_model
from solver.utils import solve_sudoku

from ocr.ocr_decoder import img_to_grid
from ocr.ocr_detector import get_detector
from ocr.ocr_recognizer import get_recognizer
from glob import glob


if __name__ == "__main__":
    solver = get_model()

    solver.load_weights("solver.h5")

    detector_model_h5 = "ocr_detector.h5"
    detector_model = get_detector()
    detector_model.load_weights(detector_model_h5)

    recognizer_model_h5 = "ocr_recognizer.h5"
    recognizer_model = get_recognizer()
    recognizer_model.load_weights(recognizer_model_h5)

    for img_path in glob('eval/*.png'):
        print(img_path)

        img = cv2.imread(img_path)

        grid = img_to_grid(img, detector_model, recognizer_model, plot_path="plot.png", print_result=False)

        for l in grid:
            print(l)

        x_ocr = [[a for a in x] for x in grid]

        pred_gen = solve_sudoku(x_ocr, solver)

        print("OCR Prediction solved ?", pred_gen.board.is_solved())
        print(pred_gen.board)
