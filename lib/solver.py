# -*- coding: utf-8 -*-
"""Contains all the functions related to solving a sudoku"""

import numpy as np

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
    if solve_step(grid):
        print('-'*19)
    else:
        print("No solution exists")
    return grid
