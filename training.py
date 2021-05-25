
import tkinter as tk
from tkinter import messagebox
from cv2 import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

def trainer():
    # get training data from recorded face
    data_path = './faces/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # Create arrays for training data and labels
    Training_Data, Labels = [], []

    # Open training images in datapath
    # Create a numpy array for training data
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)

    # Initialize facial recognizer
    model = cv.face.LBPHFaceRecognizer_create() 

    # train model and save to faces data folder
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    model.save('faces data/faces_data.yml')
    #print("Model trained sucessefully")
    messagebox.showinfo('Result','Training face data completed.')
