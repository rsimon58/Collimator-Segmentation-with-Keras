import numpy as np
import scipy as sp
from PIL import Image
#from scipy import misc


def imread(filename):
  """Read image from file.
  Args:
    filename: .
  Returns:
    im_array: .
  """
  im = sp.misc.imread(filename)
  # return im / 255.0
  return im / 127.5 - 1.0



def load_test_data(filename):
    #if os.path.isfile('proj_test.npy'):
    #    imgs_p = np.load('proj_test.npy')
    #    imgs_s = np.load('scatter_test.npy')
    #    return imgs_p, imgs_s

    # Load projection data 
    nRows = 256
    nCols = 256
    size = nRows*nCols      
    projf = np.empty(shape=(nRows, nCols), dtype=np.float32)
    with open(filename, 'rb') as fp:     
         proj = np.fromfile(fp, dtype=np.float32, count=size)
         projf = np.reshape(proj, (nRows, nCols))
    fp.close()
    return projf

def write_data(filename, np_image):


    # Load projection data 
    nRows = 256
    nCols = 256
    size = nRows*nCols      
    with open(filename, 'wb') as fp:     
         np_image.tofile(fp)
    fp.close()

def imsave(np_image, filename):
  """Save image to file.
  Args:
    np_image: .
    filename: .
  """
  # im = sp.misc.toimage(np_image, cmin=0, cmax=1.0)
  im = sp.misc.toimage(np_image, cmin=-1.0, cmax=1.0)
  im.save(filename)

def imwrite(filename, np_image):
  """Save image to file.
  Args:
    filename: .
    np_image: .
  """
  # im = sp.misc.toimage(np_image, cmin=0, cmax=1.0)
  im = sp.misc.toimage(np_image, cmin=-1.0, cmax=1.0)
  im.save(filename)

def imwrite_batch(filenames, np_images):
  """Save batch images to file.
  Args:
    filenames: 
  """
  #TODO
  pass 

def imresize(np_image, new_dims):
  """Image resize similar to Matlab.

  This function resize images to the new dimension, and properly handles
  alaising when downsampling.
  Args:
    np_image: numpy array of dimension [height, width, 3]
    new_dims: A python list containing the [height, width], number of rows, columns.
  Returns:
    im: numpy array resized to dimensions specified in new_dims.
  """
  # im = np.uint8(np_image*255)
  im = np.uint8((np_image+1.0)*127.5)
  im = Image.fromarray(im) 
  new_height, new_width = new_dims
  im = im.resize((new_width, new_height), Image.ANTIALIAS)
  return np.array(im)
