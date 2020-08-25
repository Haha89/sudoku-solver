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


def solver(path, debug=False):
    IMG = cv.imread(path)
    SQUARES = picture_processing(IMG, plot=debug)
    MODEL = load_checkpoint('../results/model.pth')
    digits = [predict_digit(el, MODEL) for el in SQUARES]

    if debug:
        fig = plt.figure(figsize=(13, 13))
        ax = []
        for i, digit in enumerate(digits):
            ax.append(fig.add_subplot(9, 9, i+1))
            ax[-1].set_title(f"Predict : {digit}")
            plt.imshow(SQUARES[i], alpha=0.25)
            plt.axis('off')
        plt.show()

    board = np.resize(np.array(digits), (9, 9))
    print_board(board)
    print_board(sudoku_solver(board))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No path given. Default picture used")
        solver('..\\inputs\\sudo-med.jpeg')
    elif len(sys.argv) == 2:
        solver(sys.argv[1])
    elif len(sys.argv) == 3:
        solver(sys.argv[1], sys.argv[2])
    else:
        print("To many arguments given. please provide PATH_FILE and DEBUG")
