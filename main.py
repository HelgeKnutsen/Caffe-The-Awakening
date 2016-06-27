import pygame.camera
import time
import collections

from camera import startCamera, takePicture, dispPicture
from caffe_functions import caffeSetup, classify, printClassPred

caffe_init = caffeSetup(0, 'ogn')
labels, transformer, net = caffe_init

cam = startCamera()

while (1):

    image = takePicture(cam)
    dispPicture(image.disp)

    prob = classify(image.classify, transformer, net)

    n = 5
    printClassPred(prob, labels, n)

    time.sleep(1)

cam.stop()
