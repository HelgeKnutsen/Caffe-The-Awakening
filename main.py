import pygame.camera
import time
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import ceil, sqrt

from camera import startCamera, takePicture, dispPicture, loadImage
from selectiveSearch import selSearch
from caffe_functions import caffeSetup, classify, printClassPred

caffe_init = caffeSetup(0, 'ogn')
labels, transformer, net = caffe_init

cam = startCamera()

plt.ion()

n = 2
b = 1
while (b == 1):

    tic = time.time()

    image = takePicture(cam)

    imDisp, imHighQ = image
    #dispPicture(imDisp)

    image = loadImage('/home/ogn/caffe/data/DroneFlight1/20m/', 'left0575.jpg')

    print 'Searching..'
    height = 20
    candidates = selSearch(image, height)


    toc = time.time()

    print 'Searching took ', toc - tic, 'sec'

    tic = time.time()

    #print 'Whole image: '
    #prob = classify(imLowQ, transformer, net)
    #printClassPred(prob, labels, n)


    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40, 40))
    ax.imshow(image)
    for x, y, w, h in candidates:
    #print x, y, w, h
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.pause(0.00000001)
    plt.imshow(image, interpolation='nearest')
    plt.draw()
    plt.show()


    #f = plt.figure()

    i = 0
    for elem in candidates:
    #for i in range (0, len(candidates)):

        #elem = candidates.pop()

        cropped_im = image[elem[1]: elem[1] + elem[3], elem[0] : elem[0] + elem[2]]

        # f.add_subplot(ceil(sqrt(len(candidates))),ceil(sqrt(len(candidates))),i+1)
        # plt.imshow(cropped_im, interpolation='nearest')
        # plt.draw()
        #
        print 'Figure number', i+1
        prob = classify(cropped_im, transformer, net)
        printClassPred(prob, labels, n)
        i = i + 1

    #print 'Found', i, 'squares in the picture'

    #print 'Classifying..'

    toc = time.time()

    print 'Classifying took ', toc - tic, 'sec'


    b = 0
    #time.sleep(1)


cam.stop()

plt.show()