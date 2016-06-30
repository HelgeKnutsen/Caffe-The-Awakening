import pygame.camera
import time
import collections
import matplotlib.pyplot as plt

from camera import startCamera, takePicture, dispPicture, selSearch
from caffe_functions import caffeSetup, classify, printClassPred

caffe_init = caffeSetup(0, 'ogn')
labels, transformer, net = caffe_init

cam = startCamera()

n = 10
b = 1
while (b == 1):

    tic = time.time()

    image = takePicture(cam)
    dispPicture(image.disp)

    print 'Searching..'
    candidates = selSearch(image.classify)


    toc = time.time()

    print 'Searching took ', toc - tic, 'sec'

    tic = time.time()

    print 'Whole image: '
    prob = classify(image.classify, transformer, net)
    printClassPred(prob, labels, n)

    f = plt.figure()

    i = 0
    for elem in candidates:
    #for i in range (0, len(candidates)):

        #elem = candidates.pop()

        cropped_im = image.classify[elem[1]: elem[1] + elem[3], elem[0] : elem[0] + elem[2]]

        f.add_subplot(4,4,i+1)
        plt.imshow(cropped_im, interpolation='nearest')
        plt.draw()

        print 'Figure number', i+1
        prob = classify(cropped_im, transformer, net)
        printClassPred(prob, labels, n)
        i = i + 1

    #print 'Classifying..'

    toc = time.time()

    #print 'Classifying took ', toc - tic, 'sec'


    b = 0
    #time.sleep(1)


cam.stop()

plt.show()