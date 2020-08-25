# -*- coding: utf-8 -*-

"""Functions about predicting the digit contained in an image"""

import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt

try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = '..\\..\\..\\Tesseract\\tesseract'

def load_checkpoint(filepath):
    """Function loading the trained CNN on the QMNIST dataset"""
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model



def predict_digit(img, model, debug=False):
    """FUnction taking the picture of a cell in input and predicting the digit
    contained. If empty cell, returns 0"""

    if np.average(img) <= 0.05: #Empty
        return 0

    if debug:
        plt.imshow(img, alpha=.5)
        plt.show()
        print(img)

    #Preprocess image
    img = img.reshape(1, 28, 28)
    img = torch.tensor(img)
    norm = transforms.Normalize((0.1307,), (0.3081,))
    img = norm(img).view(1, 1, 28, 28)
    prediction = model(img.float())

    if debug:
        print(prediction)
        print(f"Prediction {torch.argmax(prediction).item()}")

    return torch.argmax(prediction).item()


def predict_tesseract(img):
    """Alternative using Tesseract module for predition"""

    final_image = Image.fromarray(img).convert("L")
    output = pytesseract.image_to_data(final_image,
                                       config=("-c tessedit"
                                               "_char_whitelist=0123456789"
                                               " --psm 10"
                                               " -l osd "))
    return output
