# -*- coding: utf-8 -*-
"""Contains all the functions related to solving a sudoku"""

import cv2 as cv
import numpy as np

FONT = cv.FONT_HERSHEY_SIMPLEX

def find_empty(board):
    """Find and return the position of first empty value in board"""
    pos_x, pos_y = np.where(board == 0)
    return (pos_x[0], pos_y[0]) if len(pos_x) else None


def solve_step(board):
    """Modifies the board to solve it recursively"""

    miss = find_empty(board)
    if not miss: #Sudoku completed
        return True

    lig, col = miss
    for i in range(1, 10):
        if is_valid(board, i, (lig, col)):
            board[lig, col] = i
            if solve_step(board): #Recursive call
                return True
            board[lig, col] = 0 #Error, reset to 0
    return False


def is_valid(board, num, pos):
    """Check if the number num in position pos is valid in the board"""
    lig, col = pos
    start_x = (lig // 3) * 3
    start_y = (col // 3) * 3

    if num in board[lig, :] or num in board[:, col]:
        return False #Check on line and column

    if num in board[start_x: start_x + 3, start_y: start_y + 3]:
        return False #Check on 3*3 box
    return True


def print_board(board):
    """Function to display the sudoku in a nice way"""
    for i, row in enumerate(board):
        if i%3 == 0 and i:
            print("-------------------")

        for j, num in enumerate(row):
            if j%3 == 0 and j:
                print("|", end="")
            if num == 0:
                num = "."
            print(num) if j == 8 else print(str(num) + " ", end="")

def sudoku_solver(grid):
    """Function that solves the grid and returns it"""
    if not solve_step(grid):
        print("No solution exists")
        return None
    return grid


def display_solved_picture(picture, board):
    """Fills the empty sudoku picture"""
    init_board = board.copy()
    solved_board = sudoku_solver(board)
    backtorgb = cv.cvtColor(picture, cv.COLOR_GRAY2RGB)
    if solved_board is not None:
        for i in range(9):
            for j in range(9):
                x = 34*j + 10
                y = 34*i + 30
                if init_board[i, j] == 0: #Add the number only is initially missing
                    cv.putText(backtorgb, str(solved_board[i, j]), (x, y),
                               FONT, 1, (0,255,0), 2, cv.LINE_AA)
    return backtorgb