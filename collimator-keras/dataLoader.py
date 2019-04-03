import os
from PIL import Image
from skimage.io import imsave, imread
from skimage.transform import rescale
import random
from random import shuffle
import numpy as np
#def create__data__list(data_path):
#    #train_data_path = os.path.join(data_path, 'train')
#    images = os.listdir(data_path)

#    train = []
#    label = []

#    for image_name in images:
#        if 'Label' in image_name:
#            continue
#        image_name = data_path + image_name
#        label_name = image_name.replace('Image', 'Label')
#        train.append(image_name)
#        label.append(label_name)
#    return train, label

def create__data__list(data_path):
    #train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(data_path)

    train = []
    label = []

    for image_name in images:
        if 'Label' in image_name:
            continue
        image_name = data_path + image_name
        label_name = image_name.replace('Image', 'Label')
        train.append(image_name)
        label.append(label_name)

    random.seed(1)
    shuffle(train)

    random.seed(1)
    shuffle(label)

    num = 1830
    num1 = 1798

    TrainData = train[0:num1]
    TrainLabel = label[0:num1]
    ValidateData = train[num1:num]
    ValidateLabel = label[num1:num]
    return TrainData, TrainLabel, ValidateData, ValidateLabel

#def create__data__list(data_path):
#    #train_data_path = os.path.join(data_path, 'train')
#    images = os.listdir(data_path)

#    #random.seed(1)
#    #images = shuffle(images)

#    TrainData = []
#    TrainLabel = []
#    ValidateData = []
#    ValidateLabel = []

#    num = 1830

#    num *= 0.8
#    cnt = 0;
#    for image_name in images:
#        if 'Label' in image_name:
#            continue
#        image_name = data_path + image_name
#        label_name = image_name.replace('Image', 'Label')
#        if(cnt < num):
#          TrainData.append(image_name)
#          TrainLabel.append(label_name)
#          cnt += 1
#        else:
#          ValidateData.append(image_name)
#          ValidateLabel.append(label_name)
#    return TrainData, TrainLabel, ValidateData, ValidateLabel

def read_image(file):
   # im = Image.open(file)
    im = imread(file, as_gray = True)

    #rescale image image
    #im = rescale(im, 1.0 / 2.0, order = 0, preserve_range = True, anti_aliasing = False)
    imarray = np.array(im)
    imarray = imarray.astype('float32')
   # imarray = imarray / 255

   # rows =  imarray.shape[0]
   # cols = imarray.shape[1]
    return imarray

def rescaleImage(im):
    #skimage.traform function
    im = rescale(im, 1.0 / 2.0, order = 0, preserve_range = True, anti_aliasing = False)
    imarray = np.array(im)
    imarray = imarray.astype('float32')
    return imarray
