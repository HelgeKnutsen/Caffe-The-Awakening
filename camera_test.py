import pygame.camera
import pygame.surfarray
import time
import caffe
import sys
import numpy as np

### Setting up Caffe
# The caffe module needs to be on the Python path:
caffe_root = '/home/helge/caffe/'
proj_root = '/home/helge/PycharmProjects/camera/'
sys.path.insert(0, caffe_root + 'python')
# Obs. The caffenet should be at caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

# Choose cpu or gpu mode:

# For cpu uncomment the line below
caffe.set_mode_cpu()

# For gpu uncomment the two lines below
# caffe.set_device(0)  # if we have multiple GPUs, pick the first one
# caffe.set_mode_gpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


### Setting up Camera
#Initialize image size on screen
screen = pygame.display.set_mode((1920, 1080))

#Start camera
pygame.camera.init()
pygame.camera.list_cameras()
cam = pygame.camera.Camera("/dev/video0", (1920, 1080))
cam.start()

n = 3 # number of pictures
for x in range(0, n):
    #Take picture
    img = cam.get_image()

    #Show image on screen
    screen.blit(img,(0,0))
    pygame.display.flip()

    # Save image to file (comment out if you take many images), by default pygame only supports uncompressed .bmp files
    # pygame.image.save(img, "pygame" + str(x) + ".bmp")

    image = pygame.surfarray.array3d(img)

    #image = caffe.io.load_image(proj_root + 'pygame' + str(x) + '.bmp')  # import the image
    #print type(image)
    # print "Image shape:", image.shape
    transformed_image = transformer.preprocess('data', image)  # transform the image

    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

    print 'predicted class is:', output_prob.argmax()

    labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'

    labels = np.loadtxt(labels_file, str, delimiter='\t')

    print 'output label:', labels[output_prob.argmax()]

    m = 5  # number of items
    top_inds = output_prob.argsort()[::-1][:m]  # reverse sort and take five largest items

    print 'probabilities and labels:'

    for i in range(0, m):
        print output_prob[top_inds[i]], labels[top_inds[i]]

    time.sleep(1)

#Stop the camera
cam.stop()