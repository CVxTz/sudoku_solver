from generator.Generator import *

difficulties = {
    'easy': (35, 0),
    'medium': (81, 5),
    'hard': (81, 10),
    'extreme': (81, 15)
}
diff = "medium"

gen = Generator("base.txt")

# applying 100 random transformations to puzzle
gen.randomize(100)

# getting a copy before slots are removed
initial = gen.board.copy()

# applying logical reduction with corresponding difficulty cutoff
gen.reduce_via_logical(difficulties[diff][0])