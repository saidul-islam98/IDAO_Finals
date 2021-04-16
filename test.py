import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import random
import os
from shutil import copyfile
from tqdm import tqdm
import argparse

from model import load_model
from preprocessor import load_image

def load_and_predict(path, model):
    x = load_image(path, rotate=False)
    x = np.expand_dims(x, axis=0)
    x = model.predict(x)
    #x = x[0][0]
    return round(x[0][0][0]), x[1][0][0]

energy_counts = [0,0,0,0,0,0]

#this function returns the energy level(1,3,6,10,20,30) that x is closest to
def closest(x):
    r = 0
    diff = 100
    
    levels = [1,3,6,10,20,30]
    for level in levels:
        if abs(x-level) < diff:
            diff = abs(x-level)
            r = level
    energy_counts[levels.index(r)]+=1
    return r

parser = argparse.ArgumentParser()
# '''
# nargs='?' means 0-or-1 arguments
# const=1 sets the default when there are 0 arguments
# '''
parser.add_argument('--path', nargs='?', const=1, default='sample_test', help='Test image folder', type= str)
parser.add_argument('--weight', nargs='?', const=1, default='weights.hdf5', help='Weight file path', type= str)
parser.add_argument('--csv', nargs='?', const=1, default=True , help='Weight file path', type=bool)

args = parser.parse_args()

root = os.getcwd()
model = load_model(os.path.join(root, 'weights.hdf5'))

class_counts = [0,0]
energy_counts = [0,0,0,0,0,0]
total_loss = 0.0

test_FOLDER = args.path

if args.csv:
    ids = [] 
    class_preds = []
    energy_preds = []

print(test_FOLDER)

for filename in tqdm(os.listdir(os.path.join(root, 'sample_test'))):
    if not filename.endswith('.png'):
        continue

    path = os.path.join(test_FOLDER,filename)
    
    cls, energy = load_and_predict(path, model)
    
    class_counts[int(cls)]+=1
    # nearest = closest(energy)
    # energy_counts[levels.index(nearest)]+=1

    total_loss+=abs(energy-closest(energy))

    if args.csv:
        ids.append(filename[:-4])
        class_preds.append(cls)
        #we round our regression predictions to their nearest energy levels
        #our model seems to have a harder time differentiating between 1kev and 3kev samples
        #so for predictions falling between 1 and 3, we leave them as they are to reduce mean absolute error
        if energy>1 and energy<3:
            energy_preds.append(energy)
        else:
            energy_preds.append(closest(energy))

if args.csv:
    print('\n')
    print('predictions.csv created')
    preds = pd.DataFrame({
            'id': ids,
            'classification_predictions': class_preds,
            'regression_predictions': energy_preds
        })
    preds.to_csv('predictions.csv', index=False)

print('class_counts', class_counts)
print('energy_counts', energy_counts)
print('total loss', total_loss/sum(energy_counts))