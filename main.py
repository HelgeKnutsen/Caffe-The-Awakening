import pygame.camera
import time
import collections
import matplotlib.pyplot as plt

from camera import startCamera, takePicture, dispPicture, selSearch
from caffe_functions import caffeSetup, classify, printClassPred

caffe_init = caffeSetup(0, 'ogn')
labels, transformer, net = caffe_init

cam = startCamera()

b = 1
while (b == 1):

    tic = time.time()

    image = takePicture(cam)
    #dispPicture(image.disp)

    print 'Searching..'
    candidates = selSearch(image.classify)
    print candidates
    elem = candidates.pop()
    print elem

    #cropped_im = image.classify.crop(elem)

    #cropped_im.show()

    toc = time.time()

    print 'Searching took ', toc - tic, 'sec'

    tic = time.time()

    print 'Classifying..'
    prob = classify(image.classify, transformer, net)

    toc = time.time()

    print 'Classifying took ', toc - tic, 'sec'

    n = 10
    printClassPred(prob, labels, n)

    b = 0
    #time.sleep(1)


plt.show()

cam.stop()