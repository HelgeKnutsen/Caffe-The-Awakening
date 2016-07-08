import pygame.camera
import pygame.surfarray
import time
import caffe
import sys
import numpy as np
import collections


mod_def = 'myData/net_modified.prototxt' # path to model definitions-file 'models/bvlc_reference_caffenet/deploy.prototxt'
weights = 'savedWeights/2016-06-29 5m, weights/weights.pretrained.caffemodel' # path to weights-file, reference weights at 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
label_path = 'myData/test_names.txt' # 'data/ilsvrc12/synset_words.txt'

# Choose cpu or gpu mode
def set_caffe_mode(gpu):
    if gpu == 0: # cpu mode
        caffe.set_mode_cpu()
    else: # gpu mode
        caffe.set_device(0)
        caffe.set_mode_gpu()
    return 0

# Write usr_name in order to set correct caffe_root '/home/usr_name/caffe/'
def set_caffe_root(user):
    return '/home/'+ user +'/caffe/'

# Set the caffe-net from bvlc_reference_caffenet
def set_caffe_net(caffe_root, mod_def = mod_def, weights = weights):
    model_def = caffe_root + mod_def
    model_weights = caffe_root + weights

    return caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

def mean_imageNet_image(caffe_root):
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    return mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values


def transform_input(mean, net, name = 'data'):
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({name: net.blobs[name].data.shape})

    transformer.set_transpose(name, (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean(name, mean)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale(name, 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap(name, (2, 1, 0))  # swap channels from RGB to BGR
    return transformer

def load_labels(caffe_root, label_path = label_path):
    labels_file = caffe_root + label_path
    return np.loadtxt(labels_file, str, delimiter='\t')

# Set up caffe with initial settings
def caffeSetup(gpu, user, name = 'data'):
    set_caffe_mode(gpu)
    caffe_root = set_caffe_root(user)
    sys.path.insert(0, caffe_root + 'python')
    net = set_caffe_net(caffe_root)
    labels = load_labels(caffe_root)
    mu = mean_imageNet_image(caffe_root)
    transformer = transform_input(mu, net)

    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs[name].reshape(50,        # batch size
                            3,         # 3-channel (BGR) images
                            227, 227)  # image size is 227x227
    caffe_init = collections.namedtuple('caffe_init', ['labels', 'transformer', 'net'])
    return caffe_init(labels, transformer, net)


def classify(image, transformer, net, name = 'data'):
    transformed_image = transformer.preprocess(name, image)  # transform the image
    net.blobs[name].data[...] = transformed_image

    ### perform classification
    output = net.forward()
    # NB! Might want to change the return statement to include more than first image in the batch
    return output['probs'][0]  # the output probability vector for the first image in the batch

def printClassPred(output_prob, labels, m): # m is number of items
    print 'predicted class is:', output_prob.argmax()
    print 'output label:', labels[output_prob.argmax()]
    print '\n'
    top_inds = output_prob.argsort()[::-1][:m]  # reverse sort and take m largest items
    print 'probabilities and labels:'
    for i in range(m):
        print output_prob[top_inds[i]], labels[top_inds[i]]
    print '\n'
    return 0


