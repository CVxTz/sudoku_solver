import time

import matplotlib.pyplot as plt
import streamlit as st

from generator.Board import Board
from ocr.ocr_decoder import img_to_grid
from solver.utils import backtracking_solve, read_from_file, read_img_from_path, load_model, set_initially_available

detector_model, recognizer_model = load_model()

st.sidebar.markdown(
    """<h1>Sudoku Solver</h1>
    <p>
    <h3>Hello !</h3></br>
    This a Sudoku Solver app that uses a custom OCR to detect digits in a cropped screenshot of a sudoku grid and then
    uses backtracking to solve it before displaying the results.
    </p>
    <p>
    Upload an image of a sudoku grid and get the solved state.
    </p>
    <img src='https://raw.githubusercontent.com/CVxTz/sudoku_solver/master/solver/samples/wiki_sudoku.png'
         width="300"></br>
    Image source : <a href='https://en.wikipedia.org/wiki/Sudoku'>Wikipedia Sudoku</a>

    """,
    unsafe_allow_html=True,
)

file = st.file_uploader("Upload Sudoku image", type=["jpg", "png"])

if file:
    img = read_from_file(file)

    grid = img_to_grid(
        img, detector_model, recognizer_model, plot_path=None, print_result=False
    )

    x = [a for x in grid for a in x]

    initial_board = Board(x)
    set_initially_available(initial_board.cells)

    solving_time = st.empty()

    html_board = st.markdown(
        "<center>" + initial_board.html() + "</center>", unsafe_allow_html=True
    )

    time.sleep(0.5)

    to_solve_board = initial_board.copy()
    start = time.time()
    n_iter, _ = backtracking_solve(to_solve_board)
    solve_duration = time.time() - start

    solving_time.markdown(
        "<center>"
        + "<h3>Solved in %.5f seconds and %s iterations</h3>" % (solve_duration, n_iter)
        + "</center>",
        unsafe_allow_html=True,
    )

    html_board.markdown(
        "<center>" + to_solve_board.html() + "</center>", unsafe_allow_html=True
    )

    # st.markdown("<center><h3>OCR Soduku</h3></center>", unsafe_allow_html=True)
    # fig = plt.figure()
    # plt.imshow(read_img_from_path("plot.png"))
    # plt.axis("off")
    # st.pyplot(bbox_inches="tight", pad_inches=0.7)
