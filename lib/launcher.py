# -*- coding: utf-8 -*-

"""Script to execute"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from analyzer import picture_processing
from solver import print_board, sudoku_solver
from predict import predict_digit, load_checkpoint
from train import Net


def solver(path):
    IMG = cv.imread(path)
    SQUARES = picture_processing(IMG, plot=True)
    digits = []
    MODEL = load_checkpoint('../results/model.pth')

    for i, el in enumerate(SQUARES):
        digit = predict_digit(el, MODEL)
        digits.append(digit)


    fig = plt.figure(figsize=(13, 13))
    ax = []
    for i, digit in enumerate(digits):
        ax.append(fig.add_subplot(9, 9, i+1))
        ax[-1].set_title(f"Predict : {digit}")
        plt.imshow(SQUARES[i], alpha=0.25)
        plt.axis('off')
    plt.show()

    #Resize to 9*9 and resolves
    board = np.resize(np.array(digits), (9, 9))
    print_board(board)
    print_board(sudoku_solver(board))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No path given. Default picture used")
        solver('..\\inputs\\sudo-med.jpeg')
    else:
        for el in sys.argv[1:]:
            solver(el)
