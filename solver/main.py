import sys

import numpy as np
import cv2


sys.path.append("..")
from generator.Generator import Generator
from generator import base_numbers
from solver.solver_models import get_model, predict
from ocr.ocr_decoder import img_to_grid
from ocr.ocr_detector import get_detector
from ocr.ocr_recognizer import get_recognizer


if __name__ == "__main__":
    solver = get_model()

    solver.load_weights("solver.h5")

    detector_model_h5 = "ocr_detector.h5"
    detector_model = get_detector()
    detector_model.load_weights(detector_model_h5)

    recognizer_model_h5 = "ocr_recognizer.h5"
    recognizer_model = get_recognizer()
    recognizer_model.load_weights(recognizer_model_h5)

    print("Automatically Generated : ")

    gen = Generator(base_numbers)
    print("start solved ?", gen.board.is_solved())

    gen.randomize(np.random.randint(31, 360))

    initial = gen.board.copy()

    print("shuffled solved ?", initial.is_solved())

    gen.reduce_via_logical(np.random.randint(31, 360))

    removed = gen.board.copy()

    print("removed solved ?", removed.is_solved())

    x_in = [a.value for x in removed.rows.values() for a in x]

    pred_gen = predict(x_in, solver)

    print("Prediction solved ?", pred_gen.board.is_solved())

    print("Manually Typed : ")

    x_new = [3, 1, 0, 0, 6, 0, 0, 0, 0,
             0, 0, 9, 0, 0, 0, 6, 2, 0,
             4, 0, 0, 0, 9, 8, 0, 0, 3,
             0, 0, 3, 9, 8, 0, 0, 4, 1,
             0, 7, 8, 1, 0, 0, 0, 5, 0,
             0, 0, 2, 6, 0, 5, 0, 0, 9,
             0, 0, 0, 0, 4, 6, 7, 9, 8,
             5, 6, 0, 0, 0, 9, 1, 0, 0,
             8, 0, 7, 0, 0, 3, 4, 0, 0]

    pred_gen = predict(x_new, solver)

    print("Medium Prediction solved ?", pred_gen.board.is_solved())
    print(pred_gen.board)

    print("From OCR : ")

    img = cv2.imread("example5.png")

    grid = img_to_grid(img, detector_model, recognizer_model, plot_path="plot.png", print_result=False)

    for l in grid:
        print(l)

    x_ocr = [x for a in grid for x in a]

    pred_gen = predict(x_ocr, solver)

    print("OCR Prediction solved ?", pred_gen.board.is_solved())
    print(pred_gen.board)






