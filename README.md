

[![DOI](https://zenodo.org/badge/259007650.svg)](https://zenodo.org/badge/latestdoi/259007650)

# sudoku_solver
Code for : [https://towardsdatascience.com/building-a-sudoku-solving-application-with-computer-vision-and-backtracking-19668d0a1e2]()

### Run
```
bash run_app.sh
```

## Intro

A Sudoku is a logic-based puzzle that usually comes in the form of a 9x9 grid
and 3x3 sub-grids of 1 to 9 digits. The condition to have a valid solution to
this puzzle is that no digit is used twice in any row, column, or 3x3 sub-grid.

The number of possible 9x9 grids is 6.67×1⁰²¹ so finding a solution can
sometimes be challenging depending on the initial puzzle. In this project, we
will build a Streamlit application that can automatically solve Sudoku puzzles
given a screenshot of one.

We will first build an Object Character Recognition model that can extract
digits from a Sudoku grid image and then work on a backtracking approach to
solve it. The final application will be accessible through an easy to use
Streamlit application.

The Sudoku python representation and the first version of the solver were both
mostly taken and modified from this Repo:
[https://github.com/RutledgePaulV/sudoku-generator](https://github.com/RutledgePaulV/sudoku-generator)

### Object Character Recognition

![](https://cdn-images-1.medium.com/max/800/1*-oZUiyeOi3y-qYChxuxtGQ.png)
<span class="figcaption_hack">  
Image Source:
[https://en.wikipedia.org/wiki/Sudoku](https://en.wikipedia.org/wiki/Sudoku)</span>

Once we have an image of a puzzle we need to extract all the digits that are
written there, as well as their position.

To do that, we will train a digit detector model and then a digit recognizer
model. The first one will tell us where does a digit appears in the image and
the second one will tell us which digit it is. We will also get a data-set for
both of those tasks.

#### Detector Model

The detector model we will use is based on a fully convolutional neural network
with skip connections, very similar to what we used in previous projects like :

* [Vessel Segmentation With Python and
Keras](https://towardsdatascience.com/vessel-segmentation-with-python-and-keras-722f9fb71b21)
* [Fingerprint Denoising and Inpainting using Fully Convolutional
Networks](https://towardsdatascience.com/fingerprint-denoising-and-inpainting-using-fully-convolutional-networks-e24714c3233)

You can read those two posts if you want to learn more about image segmentation.

The objective of this model is to output a binary mask that tells us, for each
pixel of the input image, if it is part of a digit or not.

![](https://cdn-images-1.medium.com/max/800/1*n0f4XJr4CQGZeC2k44-Yvw.png)

#### Recognizer Model

![](https://cdn-images-1.medium.com/max/800/1*CIYBsWqBNv9C9skKILFCLA.png)

<span class="figcaption_hack">Characters extracted from the grid above</span>

The recognizer model’s role is to take as input a single digit and predict which
one it is from the set {1, 2, 3, 4, 5, 6, 7, 8, 9}. It is a mostly convolutional
network but the output is a fully connected layer with softmax activation.

#### Data-set

To train the two networks described above we need annotated data. Instead of
manually annotating a bunch of Sudoku grids we can generate a synthetic data-set
since it does not cost much and hope it works 😉.

To have a realistic data-set we use multiple types of fonts, sizes, background
colors, grid elements …

![](https://cdn-images-1.medium.com/max/800/1*cXmWQWiVwx779lm9EKfYig.png)

<span class="figcaption_hack">Example of generated Image</span>

Since we generate those examples from scratch, we can get all the details about
the position and the class of each digit in the image.

![](https://cdn-images-1.medium.com/max/800/1*CfQT1X4cxMK1eqnJq8ZbbA.png)

<span class="figcaption_hack">Final OCR result</span>

### Backtracking

We will use backtracking to solve the Sudoku. This method allows us to
step-by-step build candidate solutions in a tree-like shape and then prune this
tree if we find out that a sub-tree cannot yield a feasible solution.

The way we will do it in the case of Sudoku is as follows :

* For each cell, we compute the possible values that can be used to fill it given
the state of the grid. We can do this very easily by elimination.
* We sort the cells by their number of possible values, from lowest to greatest.
* We go through the first unfilled cell and assign it one of its possible values,
then to the next one and so on …
* if we end up we a feasible solution we return it, else we go back to the last
cell we assigned a value to and change its state to another possible value.
Kinda like depth-first tree search.

![](https://cdn-images-1.medium.com/max/800/1*SEoISyrZa_RexSPhmt2w_A.png)

<span class="figcaption_hack">Numbers define the order to traversal. Source:

[https://commons.wikimedia.org/wiki/File:Depth-first-tree.svg](https://commons.wikimedia.org/wiki/File:Depth-first-tree.svg)</span>

If after exploring all the possible leaves of this tree we can’t find a solution
then this Sudoku is unsolvable.

The advantage of backtracking is that it is guaranteed to find a solution or
prove that one does not exist. The issue is, while it is generally fast in 9x9
Sudoku grids, its time complexity in the general case is horrendous.

Implementation ( Some operations, like sorting, are performed in the “Board”
class):

    def backtracking_solve(board):
        # Modified from 
        set_initially_available(board.cells)
        to_be_filled = board.get_unused_cells()
        index = 0
        n_iter = 0
        while -1 < index < len(to_be_filled):
            current = to_be_filled[index]
            flag = False
            possible_values = board.get_possibles(current)
            my_range = range(current.value + 1, 10)
            for x in my_range:
                if x in possible_values:
                    n_iter += 1
                    current.value = x
                    flag = True
                    break
            if not flag:
                current.value = 0
                index -= 1
            else:
                index += 1
        if len(to_be_filled) == 0:
            return n_iter, False
        else:
            return n_iter, index == len(to_be_filled)

### The App

We build the app using Streamlit. The app needs to allow us to upload an image,
solve the Sudoku, and display the results.

#### File Upload :

Streamlit provides a simple way to create a file upload widget using
st.file_uploader.

    file = st.file_uploader("Upload Sudoku image", type=["jpg", "png"])

#### OCR :

We apply the detector and recognizer model to create the grid.

    grid = img_to_grid(img, detector_model, recognizer_model, plot_path=None, print_result=False)

#### Solving :

We use backtracking to solve the Sudoku.

    n_iter, _ = backtracking_solve(to_solve_board)

#### Display the results :

We Display the results in a nice looking Html/Css table by specifying
unsafe_allow_html=True.

    html_board.markdown("<center>" + to_solve_board.html() + "</center>", unsafe_allow_html=True)

#### Final result :

![](https://cdn-images-1.medium.com/max/2560/1*v1bArKhF6rA0KvMxRfUg1g.png)

### Conclusion :

In this small project, we build a Sudoku solving application in Streamlit. We
train a custom OCR model along the way and use backtracking to solve the actual
Sudoku grid.


References :
[https://github.com/RutledgePaulV/sudoku-generator](https://github.com/RutledgePaulV/sudoku-generator)

Cite:
```
@software{mansar_youness_2020_4060213,
  author       = {Mansar Youness},
  title        = {CVxTz/sudoku\_solver: v0.3},
  month        = sep,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3},
  doi          = {10.5281/zenodo.4060213},
  url          = {https://doi.org/10.5281/zenodo.4060213}
}
```
