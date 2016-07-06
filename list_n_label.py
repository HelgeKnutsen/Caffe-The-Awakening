from os import listdir
from os.path import isfile, join
import random


def save_n_label(path, savefolder, label, ratio):
    train = open(savefolder + 'train.txt', 'a') # open file and append image names and labels on separate lines - training images
    test = open(savefolder + 'test.txt', 'a') # - test images
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]  # list the files in mypath

    n = len(onlyfiles) # number of images total in folder
    num_test = n/(ratio+1) # number of images saved in test-file
    #j = 0 # current number of images in test

    for i in xrange(n-num_test):
        train.write(path + '/' + onlyfiles[i] + ' ' + label + '\n')
    for j in xrange(num_test):
        test.write(path + '/' + onlyfiles[i] + ' ' + label + '\n')
    #for i in xrange(n):  # iterate over file names
        #if (random.randint(0,ratio) < ratio or j >= num_test) and i >= num_test:
            #train.write(path + '/' + onlyfiles[i] + ' ' + label + '\n')
        #else:
            #test.write(path + '/' + onlyfiles[i] + ' ' + label + '\n')
            #j += 1
    train.close()  # close train-file
    test.close() # close test-file

    return train.closed and test.closed


caffe_root = '/home/helge/caffe/' # caffe_root
path = caffe_root + 'data/test 2016.06.29/10m, cropped/2016-06-29 Bakgrunn 10m' # folder with image files
savefolder = caffe_root + 'myData/2016-06-29 10m/' # folder where image names and labels are saved

ratio = 5 # ratio of images saved in train-file vs test-file i.e. ratio = i means 1 of i images are saved in test-file
# Note the following label system is applied
# label = '0' for person
# label = '1' for background
label = '1'
print save_n_label(path, savefolder, label, ratio)

