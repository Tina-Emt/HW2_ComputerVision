# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:31:00 2020

@author: Lenovo
"""
# Achieving HOG feature and saving it in a file
import cv2
import numpy as np
image = cv2.imread("test_image1.jpg",0)

def hogCalculator(image):
    winSize = (40,56)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 128
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(np.uint8(image), winStride, padding, locations)
    return hist

hist = hogCalculator(image)
np.save("Feature_extracted.npy", hist)
np.savetxt("Feature_extracted.txt", hist, fmt="%s")
data = np.genfromtxt('Feature_extracted.txt', delimiter = '\n')
print(data)

#%%
# Saving Positive Features
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
positive_patches = faces.images
#%%
# Getting non_face images
from skimage import data,color, transform

imgs_to_use = ['camera', 'text', 'coins', 'moon',
               'page', 'clock', 'immunohistochemistry',
               'chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)())
          for name in imgs_to_use]
#%%
# Saving Negative Features
import numpy as np
from sklearn.feature_extraction.image import PatchExtractor

def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                               max_patches=N, random_state=0)
    
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale)
                              for im in images for scale in [0.5, 1.0, 2.0]])

#%%
# Getting Data and Label
from itertools import chain
from skimage import feature

AllData = np.array([feature.hog(im)
                    for im in chain(positive_patches,
                                    negative_patches)])
       
AllLabel = np.zeros(AllData.shape[0])
AllLabel[:positive_patches.shape[0]] = 1

nSample,nFeature=AllData.shape

            
#%%
# Getting train and test data
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

AllData_Shuffled, AllLabel_Shuffled = shuffle_in_unison(AllData, AllLabel)

nTrain = int(0.8*nSample)

TrainData = AllData_Shuffled[0:nTrain]
TrainLabel = AllLabel_Shuffled[0:nTrain]

TestData = AllData_Shuffled[nTrain:]
TestLabel = AllLabel_Shuffled[nTrain:]
#%%
# Fitting the best model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(), TrainData, TrainLabel)

#from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import fbeta_score, make_scorer

grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]}) 

grid.fit(TrainData, TrainLabel)
grid.best_score_

print(grid.best_score_)
print(grid.best_params_)
#%%
# training the model
model = grid.best_estimator_
model.fit(TrainData, TrainLabel)
#%%
#reading test image from file
import cv2
from skimage import data, color, transform
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
#from sklearn.preprocessing import StandardScaler
#test_image = data.astronaut()
test_image = cv2.imread("test_image1.jpg",0)

test_image = color.rgb2gray(test_image)
test_image = transform.rescale(test_image, 0.5)
#test_image = test_image[:160, 40:180]

plt.imshow(test_image, cmap='gray')
plt.axis('off');

#%%
# Achieving patches
from sklearn.svm import SVC
def GettingPatches(img, scale, patch_size=positive_patches[0].shape, 
                   istep=2, jstep=2):
    for Scale in scale:
        print("Scale ", Scale)
        Ni, Nj = (int(Scale * s) for s in patch_size)
        for i in range(0, img.shape[0] - Ni, istep):
            for j in range(0, img.shape[1] - Ni, jstep):
                patch = img[i:i + Ni, j:j + Nj]
                if Scale != 1:
                    patch = transform.resize(patch, patch_size)
                yield (i, j), patch , Scale
#%%                     
scale = [1, 1.5, 2]
indices, patches , scale = zip(*GettingPatches(test_image, scale = scale))
indices = np.array(indices)
print("indices ", indices.shape)
#%%
# Finding hog features
patches_hog = np.array([feature.hog(patch) for patch in patches])
#%%
print("patches_hog ", patches_hog.shape)
# Scores
d = model.decision_function(patches_hog)
print("d ", d.shape)
#labels
labels = model.predict(patches_hog)
print("labels: ", labels.shape)
labels.sum()
#%%
# reducing boxes with less scores
patches_hog1 = patches_hog[ d > 1.5 ]
indices1 = indices[ d > 1.5]
labels1 = labels[ d > 1.5 ]
scale = np.array(scale)
scale = scale[ d > 1.5 ]
    
#%%
TotalScore = []
for x in d:
    if x > 1.5:
        TotalScore.append(x)

#%%
from sklearn.svm import SVC

fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

scale = np.array(scale)
scaleprime = scale[labels1 == 1]



indices = np.array(indices)
counter = 0
boxes = []
for i, j in indices1[labels1 == 1]:
    
    Ni, Nj = (int(scaleprime[counter] * s) for s in positive_patches[0].shape)
    boxes.append([j, j + Nj, i, i + Ni])
    counter = counter + 1
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))

#%%
# non maximum suppression    
import tensorflow as tf
import numpy as np

#def toTensor(arg):
#    arg = tf.convert_to_tensor(arg, dtype = tf.float32)
#    return tf.matmul(arg, arg) + arg

fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

scale = np.array(scale)
scaleprime = scale[labels1 == 1]

#print(boxes)
boxes = tf.convert_to_tensor(boxes, dtype = tf.float32)
TotalScore = tf.convert_to_tensor(TotalScore, dtype = tf.float32)
 
max_output_size = 1;

selected_indicies = tf.image.non_max_suppression(
        boxes , TotalScore, max_output_size = 1, iou_threshold = 0.5, score_threshold = float('-inf'), name = None
        )
SelectedIndex = tf.Session().run(selected_indicies)

boxes_array = tf.Session().run(boxes)
SelectedBox = boxes_array[SelectedIndex[0]]
print(SelectedBox)

ax.add_patch(plt.Rectangle((SelectedBox[0] , SelectedBox[2] ), SelectedBox[1] - SelectedBox[0], SelectedBox[3] - SelectedBox[2], edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))   



    
