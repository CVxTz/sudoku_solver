import sys

sys.path.append("..")
from generator.Board import Board
from solver.utils import backtracking_solve
from ocr.ocr_decoder import img_to_grid
from ocr.ocr_detector import get_detector
from ocr.ocr_recognizer import get_recognizer
import cv2


if __name__ == "__main__":
    detector_model_h5 = "ocr_detector.h5"
    detector_model = get_detector()
    detector_model.load_weights(detector_model_h5)

    recognizer_model_h5 = "ocr_recognizer.h5"
    recognizer_model = get_recognizer()
    recognizer_model.load_weights(recognizer_model_h5)

    img = cv2.imread("example6.png")

    grid = img_to_grid(img, detector_model, recognizer_model, plot_path="plot.png", print_result=False)
    x = [a for x in grid for a in x]

    initial_board = Board(x)

    print(initial_board)

    to_solve_board = initial_board.copy()

    n_iter, _ = backtracking_solve(to_solve_board)

    print(to_solve_board)
    print(to_solve_board.is_solved())
    print("Solved in %s iterations" % n_iter)


