# sudoku-solver
Sudoku image solver program using Open CV, Pytorch and Numpy 

This programs tries to solve a Sudoku puzzle given as input.

### How it works:
1) The picture given as input is resized, converted to grayscale and filtered using Opencv.
2) The main grid is found using contour detection
3) The sudoku grid is transformed through perspective transformation
4) Each cell of the sudoku is extracted
5) OCR is done to get the digit of each cell
6) The reconstructed Sudoku is solved using backtracking

### This project includes:
1) Training of a Convolutional Neural network for ORC using QMNIST dataset
2) A set of multiple sudoku puzzles for examples

### Requirements:
- Numpy
- Matplotlib
- OpenCV
- Pytorch and Pytorchvision

### Execution:
run `python launcher.py [PATH_PICTURE] [PLOT]` 
PATH_PICTURE: optionnal, path of the picture containing the sudoku
PLOT: optionnal, boolean to set to True if you want to see some plots 

### Result:
![alt text](https://github.com/Haha89/sudoku-solver/blob/master/results/Figure_1.png "Picture Preprocessing")
