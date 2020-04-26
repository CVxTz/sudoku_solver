# Modified from https://github.com/RutledgePaulV/sudoku-generator
from generator.Board import *
import functools


class Generator:

    # constructor for generator, reads in a space delimited
    def __init__(self, base_numbers):

        # constructing board
        self.board = Board(base_numbers.copy())

    # function randomizes an existing complete puzzle
    def randomize(self, iterations):

        # not allowing transformations on a partial puzzle
        if len(self.board.get_used_cells()) == 81:

            # looping through iterations
            for x in range(0, iterations):

                # to get a random column/row
                case = random.randint(0, 4)

                # to get a random band/stack
                block = random.randint(0, 2) * 3

                # in order to select which row and column we shuffle an array of
                # indices and take both elements
                options = list(range(0, 3))
                random.shuffle(options)
                piece1, piece2 = options[0], options[1]

                # pick case according to random to do transformation
                if case == 0:
                    self.board.swap_row(block + piece1, block + piece2)
                elif case == 1:
                    self.board.swap_column(block + piece1, block + piece2)
                elif case == 2:
                    self.board.swap_stack(piece1, piece2)
                elif case == 3:
                    self.board.swap_band(piece1, piece2)
        else:
            raise Exception("Rearranging partial board may compromise uniqueness.")

    # method gets all possible values for a particular cell, if there is only one
    # then we can remove that cell
    def reduce_via_logical(self, cutoff=81):
        cells = self.board.get_used_cells()
        random.shuffle(cells)
        for cell in cells:
            if len(self.board.get_possibles(cell)) == 1:
                cell.value = 0
                cutoff -= 1
            if cutoff == 0:
                break

    # returns current state of generator including number of empty cells and a representation
    # of the puzzle
    def get_current_state(self):
        template = "There are currently %d starting cells.\n\rCurrent puzzle state:\n\r\n\r%s\n\r"
        return template % (len(self.board.get_used_cells()), self.board.__str__())
