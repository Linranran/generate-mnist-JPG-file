# Copyright 2019312 . All Rights Reserved.
# Prerequisites:
# Python 2.7
# gzip, subprocess, numpy
#
# ==============================================================================
"""Functions for downloading and uzip MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import subprocess
import os
import numpy
from six.moves import urllib
import struct
import numpy as np
from scipy.misc import imsave


def maybe_download(filename, data_dir, SOURCE_URL):
	filepath = os.path.join(data_dir, filename)
	if not os.path.exists(filepath):
		filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
def uzip_data(data_dir, key ):
	target_path = os.path.join(data_dir, key + '.gz')
	gz_file = gzip.GzipFile(target_path)
	ungz_filename = os.path.join(data_dir,key)
	open(ungz_filename, "wb").write(gz_file.read())
	gz_file.close()
	os.remove(target_path)

def download_unzip(root,dataset_name):
	data_dir=os.path.join(root,dataset_name)
	if os.path.exists (data_dir):
		# os.remove(data_dir)
		# os.mkdir(data_dir)
		print(data_dir)
		print('dir mnist already exist.')
	else:
		os.mkdir(data_dir)
	SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
	data_keys = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
	for key in data_keys:
		if os.path.isfile(os.path.join(data_dir, key)):
			print("[warning...]", key, "already exist.")
		else:
			maybe_download(key, data_dir, SOURCE_URL)
	# uzip the mnist data.
	uziped_data_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']
	for key in uziped_data_keys:
		if os.path.isfile(os.path.join(data_dir, key)):
			print("[warning...]", key, "already exist.")
		else:
			uzip_data(data_dir, key)

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
def gen_lables(traindir,testdir):
	train_imgs = os.listdir(traindir)
	test_imgs = os.listdir(testdir)
	f = open('train_lable.txt', 'w')
	for i in train_imgs:
		train_img = i
		train_lab = i.split("_")[0]
		f.write(train_img + ' ' + train_lab + '\n')
	print("genarate train lable")
	f = open('test_lable.txt', 'w')
	for i in test_imgs:
		test_img = i
		test_lab = i.split("_")[0]
		f.write( test_img + ' ' + test_lab + '\n')
	print("genarate test lable")


def gen_jpg(root,dataset_name):

	train_images_idx3_ubyte_file = os.path.join(root,dataset_name, 'train-images-idx3-ubyte')
	train_labels_idx1_ubyte_file = os.path.join(root,dataset_name, 'train-labels-idx1-ubyte')
	test_images_idx3_ubyte_file = os.path.join(root,dataset_name, 't10k-images-idx3-ubyte')
	test_labels_idx1_ubyte_file = os.path.join(root,dataset_name,'t10k-labels-idx1-ubyte')
	train_images =decode_idx3_ubyte(train_images_idx3_ubyte_file)
	test_images=decode_idx3_ubyte(test_images_idx3_ubyte_file)
	train_labels=decode_idx1_ubyte(train_labels_idx1_ubyte_file)
	test_labels=decode_idx1_ubyte(test_labels_idx1_ubyte_file)
	train_jpg_dir =os.path.join(root, "train")
	if os.path.exists(train_jpg_dir) == False:
		os.mkdir(train_jpg_dir)
	test_jpg_dir = os.path.join(root,"test")
	if os.path.exists(test_jpg_dir) == False:
		os.mkdir(test_jpg_dir)
	for i in range(train_images.shape[0]):
		image = os.path.join(root,'train',str(int(train_labels[i])) + '_' + str(i) + '.jpg')
		imsave(image, train_images[i])
	print( "gen train jpg images")
	for i in range(test_images.shape[0]):
		image = os.path.join(root,'test',str(int(test_labels[i])) + '_' + str(i) + '.jpg')
		imsave(image, test_images[i])
	print("gen train jpg images")
	gen_lables(train_jpg_dir,test_jpg_dir)

if __name__ == '__main__':
	print("===== running - input_data() script =====")
	root=os.path.split(os.path.realpath(__file__))[0]
	dataset_name="mnist"
	download_unzip(root,dataset_name)
	gen_jpg(root,dataset_name)
	print("=============   =============")