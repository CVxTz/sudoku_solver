import time

import matplotlib.pyplot as plt
import streamlit as st

from generator.Board import Board
from ocr.ocr_decoder import img_to_grid
from ocr.ocr_detector import get_detector
from ocr.ocr_recognizer import get_recognizer
from solver.utils import backtracking_solve, read_from_file, read_img_from_path

detector_model_h5 = "ocr_detector.h5"
detector_model = get_detector()
detector_model.load_weights(detector_model_h5)

recognizer_model_h5 = "ocr_recognizer.h5"
recognizer_model = get_recognizer()
recognizer_model.load_weights(recognizer_model_h5)

st.title("Soduku Solver")

file = st.file_uploader("Upload file", type=["jpg", "png"])

if file:
    img = read_from_file(file)

    grid = img_to_grid(
        img, detector_model, recognizer_model, plot_path="plot.png", print_result=False
    )

    x = [a for x in grid for a in x]

    initial_board = Board(x)

    to_solve_board = initial_board.copy()
    start = time.time()
    n_iter, _ = backtracking_solve(to_solve_board)
    solve_duration = time.time() - start

    st.markdown(
        "<center>"
        + "<h3>Solved in %.5f seconds and %s iterations</h3>" % (solve_duration, n_iter)
        + "</center>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<center>" + to_solve_board.html() + "</center>", unsafe_allow_html=True
    )

    st.markdown("<h3>OCR Soduku</h3>", unsafe_allow_html=True)

    fig = plt.figure()
    plt.imshow(read_img_from_path("plot.png"))
    plt.axis("off")
    st.pyplot()
