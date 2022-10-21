import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import zipfile
from itertools import chain, combinations
from PIL import Image

# PREPROCESSING OF IMAGES

# 1. FUNCTION TO DISPLAY 3D IMAGES
def display(path):
  img = plt.imread(path)
  dpi = 80
  height , width , depth = img.shape
  size = (width/float(dpi) , height/float(dpi))
  fig = plt.figure(figsize = size)
  ax = fig.add_axes([0,0,1,1])
  ax.imshow(img , cmap = 'gray')
  ax.axis('off')
  plt.show()
  return

# 2. FUNCTION TO DISPLAY 2D IMAGES
def display2D(path):
  img = plt.imread(path)
  dpi = 80
  height , width = img.shape
  size = (width/float(dpi) , height/float(dpi))
  fig = plt.figure(figsize = size)
  ax = fig.add_axes([0,0,1,1])
  ax.imshow(img , cmap = 'gray')
  ax.axis('off')
  plt.show()
  return

# 3. FUNCTIONS TO PREPROCESS IMAGES
def preprocess(img):
    gray_img = grayscale(img)
    bw_img = binarize(gray_img)
    filter_img = filtered(bw_img)
    sample_img = resize(filter_img)
    return sample_img

def grayscale(img):
    temp = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return temp

def binarize(img):
    thresh , temp = cv2.threshold(img , 110 , 255 , cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return temp

def filtered(img):
    temp = cv2.medianBlur(img , 3)
    return temp

def resize(img):
    temp = cv2.resize(img , (64,128))
    for x in range(128):
      for y in range(64):
        temp[x][y] = temp[x][y]/255
    return temp


# FEATURE EXTRACTION ALGORITHM

# Function to extract all substrings of a given set for columns of dataframe
def get_subsets(fullset):
  listrep = list(fullset)
  n = len(listrep)
  return [[listrep[k] for k in range(n) if i&1<<k] for i in range(2**n)]

string=['a','b','c', 'd', 'e', 'f', 'g', 'h']
columnHeaders = get_subsets(string)
columnHeaders.pop(0)

# FUNCTION TO EXTRACT FEATURES

def amsFeatureExtraction(image, label):
  #converting required file to ndarray
    file = np.array(image)
    # Division of image into two zones
    h, w = file.shape
    h1 = h//2

    zero_array = [i for i in range(255)]
    zero_array.append('LABEL')
    #Counter list to store counts of each subset
    counter = [0 for i in range(255)]

    # for both zones ... zone top..

    for j in range(h1-2):
      for k in range(w-2):
        #if mid value is 1
        if file[j+1][k+1] == 1:
          #list to include combination
          string = []
          #checking each box of sliding window
          for m in range(3):
            for n in range(3):
              if file[j+m][k+n] == 1:
                if m == 2 and n == 2:
                  string.append('a')
                elif m == 2 and n == 1:
                  string.append('b')
                elif m == 2 and n == 0:
                  string.append('c')
                elif m == 1 and n == 0:
                  string.append('d')
                elif m == 0 and n == 0:
                  string.append('e')
                elif m == 0 and n == 1:
                  string.append('f')
                elif m == 0 and n == 2:
                  string.append('g')
                elif m == 1 and n == 2:
                  string.append('h')
                else:
                  continue
          #checking index of found combo in header matrix
          string.sort()
          if string:
            index = columnHeaders.index(string)
            counter[index] = counter[index] + 1

    # for both zones ... zone bottom..
    val = h - h1
    for j in range(val-2):
      for k in range(w-2):
        #if mid value is 1
        idx = j + h1
        if file[idx+1][k+1] == 1:
          #list to include combination
          string = []
          #checking each box of sliding window
          for m in range(3):
            for n in range(3):
              if file[j+h1+m][k+n] == 1:
                if m == 2 and n == 2:
                  string.append('a')
                elif m == 2 and n == 1:
                  string.append('b')
                elif m == 2 and n == 0:
                  string.append('c')
                elif m == 1 and n == 0:
                  string.append('d')
                elif m == 0 and n == 0:
                  string.append('e')
                elif m == 0 and n == 1:
                  string.append('f')
                elif m == 0 and n == 2:
                  string.append('g')
                elif m == 1 and n == 2:
                  string.append('h')
                else:
                  continue
          #checking index of found combo in header matrix
          string.sort()
          if string:
            index = columnHeaders.index(string)
            counter[index] = counter[index] + 1
    # making a temp df out of counter to return
    counter.append(label)
    counter_df = np.reshape(counter, (1, 256))
    temp_df = pd.DataFrame(counter_df, columns = zero_array)
    return temp_df


# THE MAIN

# FUNCTION TO OPEN FILES
path = r"D:\Project\Odia_Char_DataSet-20220622T124105Z-001\Odia_Char_DataSet\Test"
# the main matrix with only columnHeades initialised as columnList
columnList = [i for i in range(255)]
columnList.append('LABEL')
amsfeature_matrix = pd.DataFrame(columns=columnList)

labels = os.listdir(path)
for each in labels:
    n_path = path+'/'+each
    files = os.listdir(n_path)
    for every in files:
        if (every.endswith(".db") == True):
            continue
        img_path = n_path+'/'+every
        img = cv2.imread(img_path)
        prep_img = preprocess(img)
        img_data = amsFeatureExtraction(prep_img , each)
        amsfeature_matrix = amsfeature_matrix.append(img_data , ignore_index=True)

amsfeature_matrix.to_csv('IITBfeatureMatrixTest.csv')
