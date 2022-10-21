import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def dist(x,y):
  return sum([(x_i - y_i) ** 2 for x_i,y_i in zip(x,y)])**0.5

def most_frequent(l):
  return max(l , key=l.count)

def knn(X_Train , Y_Train , X_Test):
    predicted = []
    for sample in X_Test.itertuples():
      sample_idx = sample[0]
      sample_data = sample[1:161]
      training_distances = []
      for train in X_Train.itertuples():
        train_idx = train[0]
        train_data = train[1:161]
        training_distances.append((dist(train_data , sample_data) , train_idx))

      training_distances.sort()
      candidates = training_distances[:3]
      top_candidate = most_frequent(candidates)
      top_candidate_idx = top_candidate[1]
      # print(Y_Train[top_candidate_idx])
      predicted.append(Y_Train[top_candidate_idx])

    return predicted

def knn_accuracy(predicted , Y_Test):
    correct = 0
    incorrect = 0
    for a,b in zip(predicted , Y_Test):
        if (a==b):
            correct+=1
        else:
            incorrect+=1

    accuracy = ((correct/len(predicted))*100)
    return accuracy

path1 = r"D:\Project\testMatrix.csv"
path2 = r"D:\Project\trainMatrix.csv"

test_images = pd.read_csv(path1)
train_images = pd.read_csv(path2)

Y_Train = train_images["LABEL"]
Y_Test = test_images["LABEL"]
train_images.drop("LABEL" , inplace=True , axis=1)
train_images.drop("INDEX" , inplace=True , axis=1)
test_images.drop("LABEL" , inplace=True , axis=1)
test_images.drop("INDEX" , inplace=True , axis=1)
X_Train = train_images
X_Test = test_images
predicted = knn(X_Train , Y_Train , X_Test)
accuracy = knn_accuracy(predicted , Y_Test)
print(accuracy)
