# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.misc import imsave
import os
import gzip
from six.moves import urllib
import numpy

# train_images_idx3_ubyte_file = 'mnist/train-images.idx3-ubyte'
# train_labels_idx1_ubyte_file = 'mnist/train-labels.idx1-ubyte'
# test_images_idx3_ubyte_file = 'mnist/t10k-images.idx3-ubyte'
# test_labels_idx1_ubyte_file = 'mnist/t10k-labels.idx1-ubyte'



data_dir = 'data'

def maybe_download(filename):
    zip_filename = filename+ '.gz'
    if os.path.exists(data_dir) == False:
        os.mkdir(data_dir)
    zip_filepath = os.path.join(data_dir,zip_filename)
    filepath= os.path.join(data_dir,filename)
    if  os.path.exists(zip_filepath):
        zip_filepath, _ = urllib.request.urlretrieve('https://storage.googleapis.com/cvdf-datasets/mnist/' + zip_filename, zip_filepath)
        print('Successfully downloaded', filename)
    with gzip.open(zipped_filepath, 'rb') as f_in, \
            gzip.open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zip_filepath)
    return filepath



def extract_data(filename, num_images):

  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(28* 28 * num_images * 1)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (255 / 2.0)) / 255
    # data = data.reshape(num_images, 28, 28, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels



train_data_filename = maybe_download('train-images-idx3-ubyte')
train_labels_filename = maybe_download('train-labels-idx1-ubyte')
test_data_filename = maybe_download('t10k-images-idx3-ubyte')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte')

# Extract it into numpy arrays.
train_images_idx3_ubyte_file = extract_data(train_data_filename, 60000)
train_labels_idx1_ubyte_file = extract_labels(train_labels_filename, 60000)
test_images_idx3_ubyte_file = extract_data(test_data_filename, 10000)
test_labels_idx1_ubyte_file = extract_labels(test_labels_filename, 10000)


def decode_idx3_ubyte(idx3_ubyte_file):

    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('have convert %d' % (i + 1))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):

    bin_data = open(idx1_ubyte_file, 'rb').read()


    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ( 'have convert %d' % (i + 1) )
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):

    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):

    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):

    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):

    return decode_idx1_ubyte(idx_ubyte_file)

def run():
    traindir="train"
    if os.path.exists(traindir) == False:
        os.mkdir(traindir)
    testdir="test"
    if os.path.exists(testdir) == False:
        os.mkdir(testdir)
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print (train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    print('done')
	
	
    for i in range (train_images.shape[0]):
        image = 'train/' + str(int(train_labels[i])) + '_' + str(i) + '.jpg'
        imsave(image, train_images[i])
    for i in range (test_images.shape[0]):
        image = 'test/' + str(int(test_labels[i])) + '_' + str(i) + '.jpg'
        imsave(image, test_images[i])
	
	
	
    root=os.getcwd()
    traindir="train"
    testdir="test"
    train_imgs=os.listdir(traindir)
    test_imgs=os.listdir(testdir)
    f = open('train_lable.txt','w')
    for i in train_imgs:
        train_img=i
        train_lab=i.split("_")[0]
        f.write(root+'\\'+traindir+'\\'+train_img+' '+train_lab+'\n')
    print ("genarate train lable")
    f = open('test_lable.txt','w')
    for i in test_imgs:
        test_img=i
        test_lab=i.split("_")[0]
        f.write(root+'\\'+testdir+'\\'+test_img+' '+test_lab+'\n')
    print ("genarate test lable")




if __name__ == '__main__':
    run()
