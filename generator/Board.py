# Modified from https://github.com/RutledgePaulV/sudoku-generator

from generator.Cell import Cell


class Board:

    # initializing a board
    def __init__(self, numbers=None):

        # we keep list of cells and dictionaries to point to each cell
        # by various locations
        self.rows = {}
        self.columns = {}
        self.boxes = {}
        self.cells = []

        # looping rows
        for row in range(0, 9):
            # looping columns
            for col in range(0, 9):
                # calculating box
                box = 3 * (row // 3) + (col // 3)

                # creating cell instance
                cell = Cell(row, col, box)

                # if initial set is given, set cell value
                if not numbers is None:
                    cell.value = numbers.pop(0)
                else:
                    cell.value = 0

                # initializing dictionary keys and corresponding lists
                # if they are not initialized
                if not row in self.rows:
                    self.rows[row] = []
                if not col in self.columns:
                    self.columns[col] = []
                if not box in self.boxes:
                    self.boxes[box] = []

                # adding cells to each list
                self.rows[row].append(cell)
                self.columns[col].append(cell)
                self.boxes[box].append(cell)
                self.cells.append(cell)

    # returning cells in puzzle that are not set to zero
    def get_used_cells(self):
        return [x for x in self.cells if x.value != 0]

    # returning cells in puzzle that are set to zero
    def get_unused_cells(self):
        unused_cells = [x for x in self.cells if x.value == 0]
        unused_cells.sort(key=lambda x: len(self.get_possibles(x)))
        return unused_cells

    # returning all possible values that could be assigned to the
    # cell provided as argument
    def get_possibles(self, cell):
        all = (
            a
            for x in [
                self.rows[cell.row] + self.columns[cell.col] + self.boxes[cell.box]
            ]
            for a in x
        )
        excluded = set([x.value for x in all if x.value != 0 and x != cell])
        results = [x for x in range(1, 10) if x not in excluded]
        return results

    # calculates the density of a specific cell's context
    def get_density(self, cell):
        all = self.rows[cell.row] + self.columns[cell.col] + self.boxes[cell.box]
        if cell.value != 0:
            all.remove(cell)
        return len([x for x in set(all) if x.value != 0]) / 20.0

    # gets complement of possibles, values that cell cannot be
    def get_excluded(self, cell):
        all = self.rows[cell.row] + self.columns[cell.col] + self.boxes[cell.box]
        excluded = set([x.value for x in all if x.value != 0 and x.value != cell.value])

    # swaps two rows
    def swap_row(self, row_index1, row_index2, allow=False):
        if allow or row_index1 // 3 == row_index2 // 3:
            for x in range(0, len(self.rows[row_index2])):
                temp = self.rows[row_index1][x].value
                self.rows[row_index1][x].value = self.rows[row_index2][x].value
                self.rows[row_index2][x].value = temp
        else:
            raise Exception("Tried to swap non-familial rows.")

    # swaps two columns
    def swap_column(self, col_index1, col_index2, allow=False):
        if allow or col_index1 // 3 == col_index2 // 3:
            for x in range(0, len(self.columns[col_index2])):
                temp = self.columns[col_index1][x].value
                self.columns[col_index1][x].value = self.columns[col_index2][x].value
                self.columns[col_index2][x].value = temp
        else:
            raise Exception("Tried to swap non-familial columns.")

    # swaps two stacks
    def swap_stack(self, stack_index1, stack_index2):
        for x in range(0, 3):
            self.swap_column(stack_index1 * 3 + x, stack_index2 * 3 + x, True)

    # swaps two bands
    def swap_band(self, band_index1, band_index2):
        for x in range(0, 3):
            self.swap_row(band_index1 * 3 + x, band_index2 * 3 + x, True)

    # copies the board
    def copy(self):
        b = Board()
        for row in range(0, len(self.rows)):
            for col in range(0, len(self.columns)):
                b.rows[row][col].value = self.rows[row][col].value
        return b

    # returns string representation
    def __str__(self):
        output = []
        for index, row in self.rows.items():
            my_set = list(map(str, [x.value for x in row]))
            new_set = []
            for x in my_set:
                if x == "0":
                    new_set.append("_")
                else:
                    new_set.append(x)
            output.append("|".join(new_set))
        return "\r\n".join(output)

    # exporting puzzle to a html table for prettier visualization
    def html(self):
        html = (
            "<table style='height:512px; width:512px; font-size: x-large; text-align:center;"
            " border: 4px solid black;'>"
        )
        for index, row in self.rows.items():
            values = []
            row_string = (
                "<tr>"
                if index % 3 != 0
                else "<tr style='border-top: 4px solid black;'>"
            )
            for col, x in enumerate(row):
                values.append(x.value)
                if col % 3 != 0:
                    row_string += (
                        "<td style='background-color:#9b9a9e'>%s</td>"
                        if x.initially_available or x.value == 0
                        else "<td style='background-color:#5bc0de'><b>%s</b></td>"
                    )
                else:
                    row_string += (
                        "<td style='background-color:#9b9a9e; border-left: 4px solid black;'>%s</td>"
                        if x.initially_available or x.value == 0
                        else "<td style='background-color:#5bc0de; border-left: 4px solid black;'><b>%s</b></td>"
                    )

            row_string += "</tr>"
            html += row_string % tuple(values)
        html += "</table>"
        return html

    def is_solved(self):

        solved = True

        for x, v in self.rows.items():
            v = [x.value for x in v if x.value != 0]
            solved = solved and len(set(v)) == 9

        for x, v in self.columns.items():
            v = [x.value for x in v if x.value != 0]
            solved = solved and len(set(v)) == 9

        for x, v in self.boxes.items():
            v = [x.value for x in v if x.value != 0]
            solved = solved and len(set(v)) == 9

        return solved
