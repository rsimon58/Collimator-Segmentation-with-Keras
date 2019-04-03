import os
import numpy as np
import random
from random import shuffle
from unet import create_unet, create_unet_v2, create_unet_v3
from rect_net import create_RecNet
from loss_utils import dice_coef_loss, l2_loss
from dataLoader import create__data__list, read_image
from skimage.io import imsave, imread
from skimage.transform import rescale
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
from data import normalize_MeanSD


dataset_name =  'trainNewR3'
dataset_folder = 'G:/BDI/train_data_label/'
batch_size = 32
image_size = [80, 96]
steps_per_epoch = 1100
epochs = 120
learning_rate = 1e-4,
learningRateReducationRate = 0.8
learningRateReducationEpochStep = 200
max_steps =  200
#model_batch_normalization = True
loss_mode ='dice'
modelFile = ''
bUseUNET = False

# learning rate decay
learningRateReducationRate = 0.8
learningRateReducationEpochStep = 200


def train():

    #print("Get U-Net")
    #retrainModelStr = ''
    ## Load the model from the file
    #if (modelFile != '' and os.path.exists(modelFile)):
    #    from keras.models import load_model  # noqa
    #    from keras import backend as K
    #    import re
    #    m = re.search('(?<=weights.)[0-9]+', os.path.basename(modelFile))
    #    retrainModelStr = '-M{0}'.format(m.group(0))  # -M0022
    #    print('----------------------------------------------')
    #    print('Loading model from file: {0}'.format(modelFile))
    #    print('----------------------------------------------')
    #    model = load_model(modelFile,
    #                       custom_objects={'dice_coef_loss': dice_coef_loss,
    #                                       'dice_coef': dice_coef})
    #    K.set_value(model.optimizer.lr, learningrate)
    ## build the model
    #else:
    if(bUseUNET):
        print("Using UNET")
        model = create_unet_v3(image_size[0], image_size[1], model_batch_normalization)  # noqa # image size(it must be same as the model inputs)
    else:
        print("Using RECT fitting")
        model = create_RecNet(image_size[0], image_size[1])

        # optimizer
    optimizer = Adam(lr=learning_rate)
        #optimizer = RMSprop(lr=learning_rate)

    if (loss_mode == 'l2'):
         model.compile(optimizer=optimizer, loss=l2_loss, metrics=['accuracy'])
    elif (loss_mode == 'dice'):
         model.compile(optimizer=optimizer, loss = dice_coef_loss, metrics=['accuracy'])
    elif (loss_mode == 'mean_squared_error'):
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    elif (loss_mode == 'binary_cross_entropy'):
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:  # default
         model.compile(optimizer=optimizer, loss=loss_mode, metrics=['accuracy'])

    model.summary()

     #read file names into a list and shuffle
    #data_list_image, data_list_label = create__data__list(dataset_folder)
    train_data_list, train_label_list, validate_data_list, validate_label_list = create__data__list(dataset_folder)

    random.seed(1)
    shuffle(train_data_list)

    random.seed(1)
    shuffle(train_label_list)

    data_size = len(train_data_list)
    epoch_num = int(data_size / batch_size)

    max_steps = epoch_num * 200

    print('Fitting model...')
    # get train generator

    for step in range(0, max_steps):
      batch_idx = step % epoch_num
      
      #create list of file names for 
      batch_train_data_list = train_data_list[int(batch_idx * batch_size) : int((batch_idx + 1) * batch_size)]
      batch_train_label_list = train_label_list[int(batch_idx * batch_size) : int((batch_idx + 1) * batch_size)]
      
      # Load batch data.
      batch_train_data = np.array([read_image(line) for line in batch_train_data_list])
      batch_train_label = np.array([read_image(line) for line in batch_train_label_list])

      #normalize data
      normalize_MeanSD(batch_train_data)

      batch_train_label /= 255.

      #add channel (batch_size, row, cols) -> (batch_size, row, cols, 1)
      batch_train_data = np.expand_dims( batch_train_data,3)
      batch_train_label = np.expand_dims( batch_train_label,3)
      bsize = batch_train_data.shape[0]
      loss, acc = model.train_on_batch(batch_train_data, batch_train_label)

      print('loss : %s acuuracy : %s' % (-loss, acc))

      epoch = (step / epoch_num)

      if batch_idx == 0:
        # Shuffle data at each epoch.
        random.seed(1)
        shuffle(train_data_list)
        random.seed(1)
        shuffle(train_label_list)
        print('Epoch Number: %s' % int(epoch))
  
      ##calculate loss for validation data
      if(epoch % 1 == 0):
        #print('validation')
        batch_validate_data = np.array([read_image(line) for line in validate_data_list])
        batch_validate_label = np.array([read_image(line) for line in validate_label_list])

        normalize_MeanSD(batch_validate_data)

        batch_validate_label /= 255.

        batch_validate_data = np.expand_dims( batch_validate_data,3)
        batch_validate_label = np.expand_dims( batch_validate_label,3)

        #size = shape.batch_train_data[0]
        loss, acc = model.evaluate(x=batch_validate_data, y=batch_validate_label, batch_size = 32,verbose=0)
        print('validation loss : %s accuracy : %s \n' % (-loss, acc))

      ##predict the last batch
      if (epoch > 1 and epoch % 50 == 0):
        # Run a batch of images	
        print('!!!!!!!!! predicting  train!!!!!!!!!!!!!!!!!!!')
        prediction = model.predict_on_batch(batch_train_data) 

        for i in range(0, prediction.shape[0]):
          #print('%s ' % i)
          file_name = batch_train_data_list[i]
          file_name = file_name.replace('G:/BDI/train_data_label','G:/BDI/pred')
          file_name = file_name.replace('.tif', '_pred.tif')
          pred = array_to_img(prediction[i,:, :, :], scale = True)
          pred.save(file_name)

      if (epoch > 1 and epoch % 50 == 0):
        # Run a batch of images	
        print('!!!!!!!!! predicting validation!!!!!!!!!!!!!!!!!!!\n')
        batch_validate_data = np.array([read_image(line) for line in validate_data_list])

        normalize_MeanSD(batch_validate_data)

        batch_validate_data = np.expand_dims( batch_validate_data,3)

        prediction = model.predict_on_batch(batch_validate_data) 

        for i in range(0, prediction.shape[0]):
          file_name = validate_data_list[i]
          file_name = file_name.replace('G:/BDI/train_data_label','G:/BDI/validate')
          file_name = file_name.replace('.tif', '_pred.tif')
          pred = array_to_img(prediction[i,:, :, :], scale = True)
          pred.save(file_name)

      ### Save model weights
      if (epoch > 1 and epoch % 50 == 0):
          model.save_weights('G:/BDI/validate/unet_wts.h5')
      ###  checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      ###  saver.save(sess, checkpoint_path, global_step=step)
if __name__ == '__main__':
  
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
  train()
  #predict()
  #file_name = 'G:/BDI/train_data_label/Labeldata40x48_374_1.tif'
  #im = imread(file_name, as_gray = True)
  #im = rescale(im, 1.0 / 2.0, order = 0, preserve_range = True, anti_aliasing = False)
  #im = np.expand_dims(im,3)
  #imarray = np.array(im)
  #imarray = imarray.astype('float32')

  #  #rescale image image
  ##im = rescale(im, 1.0 / 2.0, order = 0, preserve_range = True, anti_aliasing = False)
  #imOut = array_to_img(imarray, scale = False)
  #file_name = 'G:/BDI/train_data_label/test.tif'

  #imOut.save(file_name)
  
 