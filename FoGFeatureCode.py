import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# FUNCTION TO DISPLAY 3D IMAGES
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

# FUNCTION TO DISPLAY 2D IMAGES
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

# FUNCTIONS TO PREPROCESS IMAGES
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

# FUNCTION TO EXTRACT FEATURES
def extractfeatures(img , label):
    #for 32 parts of img
    new = img
    feature_img=[]
    for i in range(15 ,128 , 16):
      for j in range(15 , 64 ,16):
        #Now traversing each
        entry = [0,0,0,0,0]
        for k in range((i-15) , i):
          for l in range((j-15) , j):
            temp = [[new[k][l] , new[k][l+1]] , [new[k+1][l] , new[k+1][l+1]]]
            window = np.array(temp)
            if (((window[0][0]==window[1][1]==0)) and ((window[0][1]==window[1][0]==1))):
              entry[0] = entry[0]+1

            if (((window[0][0]==window[1][1]==1)) and ((window[0][1]==window[1][0]==0))):
              entry[1] = entry[1]+1

            if (((window[0][0]==window[0][1]==1)) and ((window[1][0]==window[1][1]==0)) or ((window[0][0]==window[0][1]==0)) and ((window[1][0]==window[1][1]==1))):
              entry[2] = entry[2]+1

            if (((window[0][0]==window[1][0]==1)) and ((window[0][1]==window[1][1]==0)) or ((window[0][0]==window[1][0]==0)) and ((window[0][1]==window[1][1]==1))):
              entry[3] = entry[3]+1

            if (window[0][0]==window[1][1]==window[0][1]==window[1][0]==1):
              entry[4] = entry[4]+1

        for b in range(5):
          feature_img.append(entry[b])

    feature_img.append(label)
    feature_matrix = np.reshape(feature_img , (1,161))
    column = ['F' , 'B' , 'H' , "V" , 'N']*32
    column.append('LABEL')
    temp_df = pd.DataFrame(feature_matrix , columns=column)
    return temp_df


# FUNCTION TO OPEN FILES
path = "D:\Project\Prank"
column = ['F' , 'B' , 'H' , "V" , 'N']*32
column.append('LABEL')
fog_matrix = pd.DataFrame(columns=column)
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
        img_data = extractfeatures(prep_img , each)
        fog_matrix = fog_matrix.append(img_data , ignore_index=True)

fog_matrix.to_csv('testMatrix.csv')
