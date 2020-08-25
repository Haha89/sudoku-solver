# -*- coding: utf-8 -*-

"""
    Preprocessing of the sudoku picture:
    Inputs:
        img : array of the picture to analyse
        plot : Put True to get some graphs
    Output:
        list of 81 sub_pictures, each picture contains (or not) a digit
"""

import operator
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

WIDTH = 1000
WIDTH_WRAP = 306
MARGIN = 3

def picture_processing(img, plot=False):
    """
    Parameters
    ----------
    img : Numpy array
        Picture of the sudoku.
    plot : Boolean, optional
        Set to True to plot additional pictures. The default is False.

    Returns
    -------
    squares : list
        List of array. Each array is the picture of one sudoku cell

    """
    #Resizing and filtering
    img = cv.resize(img, (WIDTH, int(img.shape[0]*WIDTH/img.shape[1])))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 11, 17, 17)
    gray = cv.Canny(gray, 30, 200)

    #Find the contours of the sudoku
    contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    polygon = sorted(contours, key=cv.contourArea, reverse=True)[0]
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                                     polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                                 polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                                    polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                                  polygon]), key=operator.itemgetter(1))

    if plot:
        add_circle = lambda x: cv.circle(img, tuple(polygon[x][0]), 10, (0, 0, 255), -1)
        add_circle(top_left), add_circle(top_right)
        add_circle(bottom_left), add_circle(bottom_right)

    #Crop picture and Wrap
    pts1 = np.float32([polygon[top_left][0], polygon[top_right][0],
                       polygon[bottom_left][0], polygon[bottom_right][0]])
    pts2 = np.float32([[0, 0], [WIDTH_WRAP, 0], [0, WIDTH_WRAP],
                       [WIDTH_WRAP, WIDTH_WRAP]])
    transfo = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, transfo, (WIDTH_WRAP, WIDTH_WRAP))

    if plot:
        plt.subplot(121), plt.imshow(img), plt.title('Input')
        plt.subplot(122), plt.imshow(dst), plt.title('Output')
        plt.xticks([]), plt.yticks([])
        plt.show()

    #Cut the picture in 81 squares
    side = WIDTH_WRAP//9
    squares = []
    for i in range(9):
        for j in range(9):
            pic = dst[i*side + MARGIN: (i + 1) *side - MARGIN,
                      j*side + MARGIN: (j + 1) *side - MARGIN]
            binary_pic = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
            binary_pic = cv.threshold(binary_pic, 140, 255, cv.THRESH_BINARY)[1]
            squares.append(~binary_pic/255)
    return squares
